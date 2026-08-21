# Server-runbook: AIME 4xA100 (delt med Kovax)

Operativ referanse: tilkobling, status, launch/resume, omstart. Oppdatert 2026-08-21.

## Tilkobling

```
ssh -p 21031 admin@gate05.aime.info
```
- Nøkkel: `~/.ssh/id_ed25519` (samme som Hetzner-boxen, ingen passphrase).
- Fra Windows: koble fra **WSL** (git-bash har egen known_hosts og feiler første gang;
  alternativt `-o StrictHostKeyChecking=accept-new`).
- **Delt boks.** Kovax kjører egne jobber — sjekk `nvidia-smi` før du tar GPU-er.
  Vår tmux/byobu-sesjon heter `lepned`; hans er `1`. TB-porter: 6006–6008 er hans, **6009 vår**.

## Kataloger

| hva | hvor |
|---|---|
| repo | `~/repos/CeresTrain` |
| venv | `~/cerestrain-env` (aktiveres ikke — launcher finner torchrun selv) |
| outputs | `~/ceres_out/{configs,nets,tblogs,logs}` |
| T91 (22 shards) | `/mnt/lepned/T91` |
| puzzles 4-shard (BRUK DENNE) | `/mnt/lepned/puzzles_2600up_v3/tpg4` |
| puzzles original 5-shard (urørt, for andre brukere) | `/mnt/lepned/puzzles_2600up_v3/tpg` |

## Status-sjekk

```
grep -a '^TRAIN:' ~/ceres_out/logs/<ID>_launch.log | tail -1        # posisjon
pgrep -af "train.py"                                                # prosesser
nvidia-smi                                                          # GPU
grep -a '^cpu MHz' /proc/cpuinfo | awk '$4<1500' | wc -l            # throttle-sjekk
```
- Rate: mål over **>=180 s** (60 s-vinduer lyver pga checkpoint-pauser).
- Throttle-historikk: 399 MHz-låsing var BIOS-relatert, fikset 2026-08-18. Enkeltkjerner
  som dipper forbigående er normalt; feilen var MANGE kjerner, VEDVARENDE, samme kjerner.

## Launch

```
cd ~/repos/CeresTrain
CERES_NUM_DATASET_WORKERS=<W> nohup bash scripts/server/launch_ddp.sh \
  <ID> ~/ceres_out <NPROC> <GPU-liste> <PORT> > /tmp/l_<ID>.out 2>&1 &
```
- Configs må ligge i `~/ceres_out/configs/<ID>_ceres_{net,opt,data,exec,monitoring}.json`
  (launcher foretrekker denne over repo-kopien).
- **Preflight-krav: shards >= NPROC x workers PER korpus.**
  4 ranks -> `CERES_NUM_DATASET_WORKERS=1` (puzzle-korpuset har 4 shards).
  2 ranks -> workers=2 OK.
- Porter: 29500 (og 29600 for arm nr. 2).
- **Formregel:** én stor kjøring = 1 jobb x 4 GPU-er (halv veggklokke).
  A/B-par = 2+2 på NVLink-parene (GPU 0,1 / 2,3) — ~15 % bedre samlet.
  Lite nett (256x10) på 4 GPU-er = KOLLAPS (comm-bundet); aldri den formen.
- Batch: `BatchSizeForwardPass` er GLOBAL og splittes på ranks. 512-nett: fwd 4096 på
  4 ranks (1024/rank ~ 23 GB/GPU = trygt; 2048/rank = OOM på 40 GB).

## Resume / omstart av kjøring

Sett i `<ID>_ceres_opt.json`:
```
"CheckpointResumeFromFileName": "/home/admin/ceres_out/nets/ckpt_a4000-21bn11_<ID>_<numpos>"
```
og launch som vanlig. Gjenopprettes: vekter, optimizer-tilstand (Muon-momentum), num_pos
(-> LR-skjema + ckpt-kadens fortsetter), datastrøm spoles frem.
- **EMA-skyggen persisteres IKKE** — re-varmes over ~EMAMaxN perioder (~20M pos). Ikke
  sammenlign EMA-eksporter de første ~20M etter resume.
- Aux-bredde-vakt: 137-nett krever `CERES_AUX_FEATURES_PER_SQUARE=0` i miljøet.
- Første eksport etter resume er vekt-identisk med kilde-checkpointen (god sanity-sjekk).

## Omstart av SERVEREN (reboot)

Etter reboot er ingenting persistent av vårt:
1. Sjekk klokker: `grep -a '^cpu MHz' /proc/cpuinfo | sort -n | head` (alle >= 1500).
2. Relaunch trening med resume fra siste ckpt (se over).
3. TB ved behov: `nohup tensorboard --logdir ~/ceres_out/tblogs --port 6009 --host 127.0.0.1 &`
   og tunnel fra din maskin: `ssh -p 21031 -L 6009:localhost:6009 admin@gate05.aime.info -N`
   -> http://localhost:6009. Bruk `ServerAliveInterval=60` — tunneler dør ellers stille om natten.

## Eksport-fallgruver (viktig)

- ONNX-eksporter valideres med `onnxruntime.InferenceSession(path)` — 1 sekund, grønn/rød.
  TRT bygger gjerne fra en DEFEKT graf uten feilmelding (subtilt feil motor -> "spiller rart").
- Cast/FP16-buggen er fikset i `save_model.py` (b2c6af6) — men en KJØRENDE prosess startet
  før en fix har gammel kode i minnet: dens eksporter må re-eksporteres etterpå med
  `python recover_export.py <ID> ~/ceres_out <numpos>` (kopier først ckpt til
  `ckpt_lepdev_<ID>_<numpos>` — recover_export forventer lepdev-prefiks).
- Tidlige checkpoints fra varm-LR-kjøringer (512 @ 1.2e-3, 100-200M) er FP16-skjøre:
  OK i TRT, søppel i strikt fp16. Ikke bruk dem i ORT-baserte målinger.

## Diverse

- `EMAPeriodSteps` teller OPTIMIZERSTEG; 1 steg = BatchSizeBackwardPass posisjoner UANSETT
  rank-antall. Preflight-advarselen om å dele på nproc er FEIL — ignorer den.
- zstd-CLI finnes ikke på serveren; bruk python-modulen `zstandard` fra venv-en.
- Disk: `/` (916 GB) = våre outputs; `/mnt` (7,3 TB, ~99 % full) = korpus. Spør Kovax
  før noe legges på `/mnt`.
