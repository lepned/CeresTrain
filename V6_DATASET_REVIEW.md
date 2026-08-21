# Code review: c37804e DirectFromV6 (v6_dataset.py) — 2026-08-21

Full adversarial review (max effort): 10 finder-vinkler, C#-ground-truth fra Ceres-forken
builden faktisk refererer, empirisk dekoding av ekte T80-tars OG en produksjons-TPG-shard
(D:\t91_skip1_v2_surv), kjørte krasj-reproer, 3-batch ende-til-ende-smoke paa ekte data.

**DOM: ikke klar for 4xA100-kjoeringen ennaa.** Kjernen er byte-eksakt; 15 funn maa
adresseres, flere trigges av NOEYAKTIG det tiltenkte produksjonsoppsettet
(t91_v7_op1, 1896 tars, 4 ranks, isal installert, ingen eksplisitt seed).

## Verifisert RENT
- V6/V7-record-dtypes byte-eksakte vs EncodedPositionEvalMiscInfoV6 / ExtraV7 (struct-sitert,
  itemsize-assertert)
- Plane-bitrekkefoelge (unpackbits big, MSB=a-linja) empirisk riktig; rutelayout (8x13-blokker,
  repetisjon 104-111, rokade-rekkefoelge, rank/file-onehots 121-136), nyeste-foerst-historikk,
  STM-orientering m/ rank-flip — alt matcher C# og ekte shard
- wdl_q/wdl_nondeblundered-algebra, KLDPolicy-kilde, MLH-kilde+0.1*sqrt-dekoding, v7x-tail
- 15-slots yield-paritet med TPGDataset; TPG-stien byte-identisk uberoert
- Trunkerings-/gzip-CRC-haandtering komplett FOR stdlib-gzip-fallbacken

## FUNN (rangert)

### Stille input-paritetsavvik (verste klasse)
1. **EP-flagget paa FEIL RUTE**: Python 40+f (slagfeltet), C# 32+f (motstanderbondens felt,
   TPGSquareRecord.cs:351-361; empirisk bekreftet i produksjonsshard). Python mangler ogsaa
   C#s 4. betingelsesledd (!TheirPawns(48+f), EncodedPositionBoards.cs:137-154) → 6.1% falske
   flagg maalt paa ekte T80-data. (v6_dataset.py:302)
2. **Move50 halv skala + utappet**: Python rule50/100; C# min(count,100)/50
   (TPGRecordEncoding.cs:42-47). Empirisk: TPG-bytes i kanal 116 er alle partall. (:294)
3. **Historikk-fyll divergerer**: LC0 nullpadder manglende historikk; Python koder "tomt brett";
   C# FILL_IN=true repeterer naermeste ekte brett (TPGRecordConverter.cs:536; 0 av 50
   produksjonsrecords har all-zero h7). Rammer alle posisjoner < 8 plies. (:286)
4. **Policy-gulvet mangler**: C# gir HVERT lovlig trekk 0.0005-gulv (soft-legality,
   CompressedPolicyVector.cs:68) og renormaliserer betinget; Python top-92 med eksakte nuller
   og ubetinget renormalisering → annen MCTS-prior enn alle TPG-trente nett. (:319)
5. **Aux-maal**: MLH ikke klemt ved 255 plies (5.9% av ekte data over → maal opp til 4.5 vs
   TPGs [0,2.55]); unc bruker abs(best_q) ved NaN orig_q der C# fyller 0.15;
   played_q_suboptimality bruker record i, C# bruker i-1; qdev-nuller uten guard mot
   LossQDeviationMultiplier>0. (:335)

### Driftsdrepere for produksjonsoppsettet
6. **Rank-divergent shuffle**: random.Random(None) per rank + strided sharding → uten
   CERES_SHUFFLE_SEED: ~32% av korpus ALDRI lest, ~26% dobbeltlest paa 4 ranks (simulert),
   stille. tpg_datasets _default_shuffle_seed + all_reduce-sjekk er droppet — gjenbruk
   _RUN_SHUFFLE_SEED. Fast seed alene gjenoppliver resume-bias (per-worker-rng mangler
   run-komponent). (:183/:358)
7. **fd-utmatting**: _tar_handles aapner 1 fd/tar, lukker aldri; hver workers stride spenner
   ~alle 1896 tars → over ulimit 1024 tidlig i epoke 1; deretter svelges open()-OSError til
   'return None' UTEN logg → ~46% av korpus stille borte hele loepet. Trenger LRU-cap +
   feilteller (+ evt. RLIMIT_NOFILE). (:227)
8. **isal-feil slipper gjennom**: nettet fanger (OSError, EOFError, zlib.error), men
   python-isal (anbefalt fastpath!) kaster IsalError (bare Exception-subklasse) paa korrupt
   deflate — live-probet. En korrupt member → hele DDP-loepet doer. Fang isal_zlib.error
   naar importen lyktes. (:231)
9. **Blandet v6+v7 krasjer** (REPRODUSERT: assert len(carry)==len(arrays) 13 vs 16 rett etter
   flush); i lotteri-tilfellet flipper yield-tuplene stille mellom v7x/ikke-v7x. (:384/:408)
10. **NaN-propagering**: best_q/d, result_q/d, played_q, plies_left brukes raa (probs/orig_q
    ER nan_to_num'd); en NaN-record → NaN-loss → alle vekter NaN. C# har eksplisitte
    NaN-fallbacks for disse feltene. (:324)
11. **Index-cache-race**: alle ranks skriver samme .chunkindex.npz uten tmp+os.replace;
    truncert cache → zipfile.BadZipFile ukjent → krasj ved hver senere start til noen sletter
    filen. + 4x redundant kaldstart-arbeid. (:211/:202)
12. **Sidecar-preflight-konflikt**: survival/stvalue/prov-preflights lister .v7x.zst-filer
    som tar-korpus aldri har → aerlige configs DOER ved start; '=1'-workarounden faar
    survival-hodet til aa trene HELT UTEN supervision, stille. (train.py:1293-1322)
13. **Versjon-skip uten teller**: v3/v4/v5-records droppes uten diagnose; sluttrapport foerst
    etter KOMPLETT pass (aldri paa 1.45B) → feil korpus = GPU-er idle i timevis uten en linje
    output. Fail-fast etter N fullskippede chunks. (:246/:261)
14. **RAM-budsjett ~8-10x for lavt**: kommentarens "50k*8 workers = 3.3 GB" teller bare raa
    records; _flush materialiserer ~70 KB/record transient (~4 GB/worker-spike, synkronisert
    over workers ved foerste flush). (:116)
15. **Stille ignorerte knapper**: KeepDrawProb/CERES_KEEP_DRAW_PROB, POLICY_TARGET_ALPHA,
    FILE_MIRROR_AUG ignoreres av v6-stien MENS tpg_datasets bannere fortsatt hevder de er paa;
    oppskriften dokumenterer V6SkipCount/V6ShufflePool i DATA-config men bootstrap bridger
    kun OPT-config (eksempelverdiene == defaults maskerer no-op'en). (:153)

### Under 15-taket (verifisert, viktig nok til aa nevne)
- C# DataSourceType-enum mangler DirectFromV6 → all C#-tooling kaster JsonException paa configen
- boards_per_batch>1/action-configs: pool-permutasjon oedelegger record-adjacens;
  policy_index_in_parent=-1 indekserer slot 1857 stille
- NumTPGFilesToSkip teller naa GAMES (~5 posisjoner) ikke shards (~millioner)
- Starvation-fallback lar sultne workers lese HELE korpus (duplisering) der TPGDataset raiser
- recover_export CERES_HOST_PREFIX-default 'lepdev' vs train.py's gethostname() → to
  navneskjemaer / FileNotFoundError paa server
- Per-pass re-enumerering av kataloger er droppet (nye tars midt i loepet ignoreres)

## Anbefalt fikserekkefoelge
1. Paritet: EP-rute+betingelse, Move50-skala, historikk-FILL_IN, policy-gulv (funn 1-4)
2. DDP-seed: gjenbruk _RUN_SHUFFLE_SEED + all_reduce-sjekken (funn 6)
3. Robusthet: isal-catch, fd-LRU, NaN-guard, atomisk cache-skriving (funn 8,7,10,11)
4. Resten + validering: paritets-A/B (samme posisjoner gjennom TPG-sti og v6-sti,
   sammenlign tensorene bit for bit) er GULLSTANDARDEN foer serverloep.
