# Server-oppskrift: 512-16 + dual-plane deg4 (beste kjøring 2026-08-20)

Destillat av dp2-kampanjens beste validerte konfigurasjon, tilpasset
serverens 512-16-nett (4xA100, DDP). Kilde-evidens: dpdeg-serien +
speil-matrisen @5M (F:\cout\puzzle_tacs_*_gate.log) og EB-cmp-målinger
(cmp_toolbox / cmp_*_20s.log). Alt under er trent/målt lokalt på 256-10 og
512-15; koden er DDP-validert.

## Hvorfor denne oppskriften

| Funn | Evidens |
|---|---|
| RelDegrees = replikert policy-løft | +52/+29 og +22/+16 in-dist (2 seeds), 0 % kost |
| P-plan-dybde = generaliseringsspak | OOD-pT3 −13 → +30 → +49 for 2→3→4 blokker (monotont) |
| 5. blokk gir kun in-dist (memorering) | +75 in-dist, OOD flat → 4 blokker er riktig dybde |
| P-planet er ~gratis ved 512-skala | EB cmp begge ordrer: ingen målbar EPS-kost (512-15) |
| deg3 tåler anneal bedre (mindre OOD-tap) | deg310 vs dp210-kontroll @10M: +71/+59 in-dist, +19/+8 OOD |

## Net-config (ceres_net.json) — KOMPLETT chassis, ikke bare delta

VIKTIG (laert 2026-08-20 kveld): forste versjon av denne filen listet bare
dual-plane-deltaet — da arvet serveren sine egne defaults for resten
(bl.a. RPE av). Under er HELE chassiset evidensen ble malt pa (lokal
dp2/tacs-familie), tilpasset 512-16:

```json
{
    "ModelDim": 512,
    "NumLayers": 16,
    "NumHeads": 8,
    "PreNorm": false,
    "NormType": "RMSNorm",
    "FFNMultiplier": 6,
    "FFNActivationType": "Mish",
    "HeadsActivationType": "Mish",
    "NonLinearAttention": true,
    "SoftCapCutoff": 100,
    "SmolgenDimPerSquare": 32,
    "SmolgenDim": 256,
    "SmolgenToHeadDivisor": 4,
    "SmolgenActivationType": "Swish",
    "UseRPE": true,
    "UseRPE_V": false,
    "UseRoPE": false,

    "UseVisEdgeBias": false,
    "VisEdgeFamilies": "vis,xray,pinray,check,flight",
    "VisEdgeGates": "",

    "UseDualPlane": true,
    "DualPlanePolicyDecode": true,
    "DualPlaneRelDegrees": true,
    "DualPlaneLayers": 4,
    "DualPlaneSoftMinHeads": 2
}
```

### Komponent-viktighet (malt @5M, dp2-chassis, ablasjoner dp2ns/dp2nr)

| Komponent | Kost ved fjerning | Dom |
|---|---|---|
| Smolgen | P −71, pT3 −59 in-dist; KLD 1.05→1.43 | **PAKREVD** |
| RPE | P −3 in-dist, men OOD −35/−36 | Anbefalt, ikke kritisk |

Kjorer serveren uten RPE (som runen startet 2026-08-20): akseptabelt —
v4r-presedens (2.5B, smolgen + ingen pos-enc = policy-paritet), P-planet
baerer egen geometri (fil/rank-embed + relasjonsbiaser), og OOD-kosten er
malt i puzzle-regimet der den trolig overdriver. Men merk at deg4-deltaene
(+49 OOD osv.) ble malt MED RPE — forste RPE-pa-arm pa server er verdt a
prove nar baseline-deltaet er kjent.

Merknader:
- `UseVisEdgeBias: false` = bar chassis. `VisEdgeFamilies` MÅ likevel stå:
  dual-plane bygger sine 20 relasjonskanaler privat fra familielisten
  (`dp_private_vis`); uten den asserter ceres_net ved oppstart.
- Ingen cf (`VisEdgeGates: ""`): gates koster 26-39 % NPS og er
  puzzle-spesialist — feil bytte for spillkorpus.
- P-planet er fast dp=128/32 tokens uansett trunkbredde → kostnaden ved
  512-16 er malt til ~0 % (EB cmp, begge ordrer, 20 s startpos).
- Zero-init-kontrakt: alle injects/decode-koplinger starter i eksakt
  no-op — warm-start fra eksisterende 512-16-checkpoint er trygt
  (dual-plane-familien ligger i `_AUX_HEAD_PREFIXES` og freshinitialiseres
  ved resume fra ikke-dp-checkpoint).

## Opt-config — anbefalinger

```json
{
    "LearningRateBase": 0.0006
}
```

- **LR 6e-4, ikke 1.2e-3.** To uavhengige grunner: (1) sqrt-bredderegelen
  fra 8e-4@256 gir ~6e-4@512; (2) QK-clip-analysen viste tung klipping på
  512-nettet ved 1.2e-3 — klipping er symptomet på for varm LR for bredden.
  Value-kollapsen 200-500M på server-baseline er konsistent med samme
  diagnose.
- **INGEN FileMirrorAug** (bruker-beslutning, riktig): gevinsten er
  anti-memorering i begrenset-data-regimer; T80-korpuset (4B pos) har ikke
  memoreringsproblemet, kvalifiseringsandelen er lav på spilldata
  (rokaderettigheter), og speiling på spilldata er utestet i skala.
- `BatchSizeForwardPass` er GLOBAL og deles på ranks i DDP — samme verdi
  som eksisterende server-configs, ikke gang opp per GPU.

## Split-LR-status (per 2026-08-20)

Implementert og korrekthetsvalidert, IKKE effektvalidert:
- Kode: `muon.py` (`lr_ratios` per-param), `train.py` (FAMILY-LR-blokk med
  medlemskaps-dump), config-nokler `Opt_LearningRateHeadsRatio` /
  `Opt_LearningRateCouplingsRatio`, `MuonAdamWScope: "ffn-only"`
  (Kovax-partisjonen, AdamW for attention).
- Validering: `tools/splitlr_smoke.py` gronn; mikro-run slr0 trent
  ende-til-ende inkl. checkpoint-resume, banner bekreftet
  (`FAMILY-LR: heads ratio=0.333 (32 params), couplings ratio=2.0 (28 params)`).
- Effekt: H1-H4-armene (fixed-budget 150M, plan i
  F:\cout\Findings\split_lr_plan_2026-08.md) er IKKE kjort enna.

**Anbefaling for forste server-run: la split-LR sta AV** (utelat nøklene).
En variabel om gangen — forst dual-plane-deltaet mot 512-16-baselinen du
allerede har, sa evt. H1 (heads ratio 1/3) som neste arm nar deltaet er
kjent.

## DirectFromV6: tren rett fra LC0-tars/chunks — null TPG-lagring (2026-08-21)

For servere uten diskplass til TPG-shards: data-configen
```json
{
    "SourceType": "DirectFromV6",
    "TrainingFilesDirectory": "/path/til/tars-eller-chunks",
    "V6SkipCount": 30,
    "V6ShufflePool": 50000
}
```
leser LC0 v6- OG v7-records direkte — baade loese .gz-chunks og
UUTPAKKEDE .tar-arkiver (medlemmene leses in-place via seek+read;
indeks caches som <tar>.chunkindex.npz naar katalogen er skrivbar).
v7-halen mater trainerens v7x-forbrukere (censored q/d + z-provenance)
uten sidecar-filer.

Krav/valg:
- `AuxFeaturesPerSquare: 0` (137-kanals modell; qz-ablasjonen viste aux-noeytralitet)
- `LossQDeviationMultiplier: 0` (feltet finnes ikke i v6)
- `pip install isal` (2-3x raskere gunzip; brukes automatisk om installert)
- CERES_NUM_DATASET_WORKERS etter kjerner: ~2,2M pos/h/worker (loese filer,
  native fs) og opptil ~19M pos/h/worker DIREKTE FRA TAR (een filhaandtak,
  ingen per-fil-aapningskost — raskeste sti, maalt 2026-08-21)
- V6ShufflePool: RAM = pool x workers x 8,4 KB (200k x 8 = 13 GB OOM-et
  lokalt; 50k er trygg default)
- Ingen rescore/deblunder i loypa: bruk pre-rescorede tars
- Skip er TILFELDIG per pass (hele korpuset naas over epoker, ulikt
  gen-tpgs permanente utvalg)

Validert ende-til-ende lokalt 2026-08-21: startpos-anker + felt-sanity
(v6 og v7), 5M-trening fra raa chunks pa deg4-chassiset (v6smk: value
1972/1719 = deg4mx-klassen ved samme LR-fase; policy begrenset av
een-dags-korpus som ventet), ORT-load groenn, gate kjoert.

## Value-oscillasjon (observert 400M-1B paa foerste 512-16-deg4-loep) — diagnose og kur

**Symptom:** EB-value alternerer med ~100M-periode fra 400M (500M ned, 600M opp, ...);
TB viser value/value2/unc oscillerende mens policy og qk-clip er friske. Baseline uten
dual-plane: daarligere niva men stabilere.

**Diagnose (to komponenter som komponerer):**
1. DELT-POOL-MEKANISMEN: P-planet (~500k params) deles av policy-decoden (sterke,
   committende gradienter) og value-injectene (svake). Policy omformer kontinuerlig
   representasjonen value leser -> dp gir bedre value-NIVA men hoeyere varians enn
   baseline (trunk-only value = sloev men stabil). At value2 oscillerer i takt tross
   0.04-vekt beviser at det er representasjons-drevet, ikke loss-konkurranse.
2. Aera-nonstasjonaritet i korpuset (T91: remisrate/sluttspilltetthet/eval-skarphet
   drifter med tar-dato; shard-granulaer lesing) gir ~100M-periodisiteten.
   DirectFromV6 mikser paa partiniva og fjerner denne komponenten i senere loep.

**Kur (validert kirurgisk lokalt, commit 18e5b84):**
```json
net-config:  "DualPlanePolicyGradScale": 0.25
opt-config:  "LossValueMultiplier": 2.0
```
- Grad-scale: forward-IDENTITET (null inferenskost, bit-lik funksjon); skalerer kun
  policy-decodens gradienter INN i de delte P-tokene (verifisert eksakt: P-plan-grad
  x0.25, decode-vekter x1.0, value-grad x1.0). Demper policys omformingskraft over
  poolen; 5M-smoke viste moderat tidlig policy-etterslep (-52 P) som forventet nar
  den raskeste laereren dempes — gevinsten er stabilitet paa 400M+-skala.
- Value-masse 2.0: Kovax-adopsjon #1 (hans objektiv er ~50 % value; vaart ~20-25 %) —
  adresserer at trunken flytter seg for policy paa values bekostning.
- Survival-anker: UTGAAR paa server (besluttet 2026-08-21): labelene finnes kun
  som TPG-sidecars, og serverens datasti er chunks/tars (v6/v7) som ikke baerer
  dem — preflighten nekter kombinasjonen by design. Value-familiens reserve er
  dermed den value-private P-blokken alene.

**Anbefalt design (besluttet 2026-08-21 kveld): FRISKT 1B-loep, ikke forlengelses-arm.**
Begrunnelse: (1) en resume-hale senker LR samtidig som stabilisatorene slaas paa —
attribusjonen drukner i s3-loven (value konsoliderer i lav LR uansett); (2) value-
representasjonen DANNES i 100-400M-vinduet — stabilisatorene maa beskytte
formasjonsfasen, ikke bare reparere halen; (3) RPE hoerer til fra steg 0 og kan
ikke varmstartes inn.

Konfig-pakken for det friske loepet (alle punkter med maalt evidens).
DATASTI: DirectFromV6 paa v7-korpusene direkte (IKKE TPG) — de nye v7-feltene
(OppPlayedIndex m.fl.) finnes kun der, og partiniva-miksingen motvirker
aera-komponenten av oscillasjonen. Flere kataloger skilles med ';' —
miksing blir volum-proporsjonal (cv2 ~1/3 av T91 => ~25 % av stroemmen):
```json
data-config: "SourceType": "DirectFromV6",
             "TrainingFilesDirectory": "/path/t91_v7_op1;/path/cv2",
             "V6SkipCount": 30, "V6ShufflePool": 50000,
             "V6MaxResultQDelta": 1.2
net-config:  "DualPlanePolicyGradScale": 0.25
opt-config:  "LossValueMultiplier": 2.0,
             "LossOppPolicyMultiplier": 0.1,
             "AuxFeaturesPerSquare": 0, "LossQDeviationMultiplier": 0
env:         CERES_SHUFFLE_SEED=<fast tall>, CERES_NUM_DATASET_WORKERS=4 per rank,
             CERES_DDP_STATIC_GRAPH=1  (PAAKREVD med LossOppPolicyMultiplier>0
             under torchrun: oppp-hodet er stash-only og usynlig for DDPs
             default-reducer; guard i train.py feiler hoylytt uten. Review
             2026-08-21 funn 1+2 — participation-terms er inne, saa mixed/
             all--1-batcher er trygge.)
```
- Opponent-policy aux (LossOppPolicyMultiplier 0.1): Monroe-ideen (+5 % hos LC0),
  target = v7 OppPlayedIndex (99 % populert i begge korpus), training-only hode,
  -1 maskert. Validert: enhets-smoke + 1M integrert trening
  (opp_policy_loss 6.34 -> 4.80; random-baseline 7.5).
- grad-scale 0.25: kirurgisk verifisert (P-plan-grad x0.25, decode-vekter x1.0,
  value-grad x1.0), null inferenskost
- RPE: DROPPET etter kostmaaling (2026-08-21 kveld): TRT-serving-kost maalt til
  ~11 % EPS paa 256-10 (dp2 vs dp2nr, begge ordrer) og ~10-16 % paa 512-15
  (rpe512-probe vs deg3-512) — kosten krymper IKKE med bredde. Mot +15-36
  OOD-policy (Nodes=1, usikker overfoering til spill) er nettoen marginal-til-
  negativ ved fast tenketid. Arkivert som maalt-og-avvist for serving-nett.
- value-masse 2.0: Kovax-adopsjon #1
- Survival-anker: IKKE i dette loepet (brukerbeslutning) — staar som senere kandidat
- Datasti (revidert 2026-08-21): DirectFromV6 paa v7 DIREKTE (ingen TPG) —
  brukerbeslutning; de nye v7-feltene (OppPlayedIndex, QAfterPlayedMove) finnes
  bare der. Multi-rot: cv2 + T91-v7 i ett TrainingFilesDirectory (';'-separert,
  volumproporsjonal miks). Hele pakken 1M-roeyktestet samlet (run2pkg +
  actsmk-varianter).

GRATIS DIAGNOSTIKK FOER START: kjoer EB paa 900M- og 1B-checkpointene fra det
gamle loepet — deres egen cosinus er allerede nede paa ~1e-4 der. Alternerer
value fortsatt, er lav LR alene bevist utilstrekkelig (stabilisatorene noedvendige);
roer den seg selv, er diagnosen mer LR-koblet enn delt-pool-koblet. En EB-kjoering,
stor informasjonsverdi.

Suksesskriterium for det friske loepet: EB-value-kurven 400M-1B flat/monoton
(ingen niva-alternering), TB value/value2/unc uten sagtann. Reserve hvis dempingen
ikke rekker: value-privat P-blokk (fork etter siste delte blokk, kun value-
gradienter). Action-hodet (q_after_played, ett-stegs lookahead-value) er den
andre value-forsterkeren i koe — BYGD OG VALIDERT 2026-08-21:
- `LossActionPlayedMultiplier` (opt-config, anbefalt 0.1): action-hodet
  ([B,1858,3] WDL per trekk) faar masked soft-CE paa SPILT trekk-slot mot
  WDL fra v7 q/d-after-played (99 % populert i begge korpus, verifisert
  q_after == -next.best_q til MAE 0.000). 1M-smoke: action_played_loss
  0.918 -> 0.834, policy/value upaavirket.
- Hodet EKSPORTERES som ONNX-output "action" — Ceres-TRT gjenkjenner paa navn
  og MCTS konsumerer via ActionWDLForMove.
- TRT-kost MAALT (EB cmp actsmk vs run2pkg, begge ordrer): ~9-12 % EPS.
  Kosten er VALGFRI ved serving: tren med hodet (value-signal inn i trunken),
  strippes ev. fra eksporten for full fart — beslutning per nett, ikke per run.

## Gate-regel (uendret fra Stage C-protokollen)

Sammenlign mot 512-16-baselinen ved samme posisjonstall; value-regelen
≥ +30 @200M rg2700 star. Les OOD (rg2100/2300-klassen) for
generaliseringsdommer og hits i gulvregimet — og for policy: pT3 +
KLD-matrisen, ikke bare P.

## ⚠ EKSPORTBUG FUNNET OG FIKSET (b2c6af6, 2026-08-20 kveld) — les foer du EB-tester dp-nett

**Symptom:** deg4@100M fra serverloepet "spilte daarlig" i EB gameplay. Aarsaken var IKKE
nettet — treningsmetrikkene var friske — men en oedelagt ONNX-eksport:

- `convert_float_to_float16` (save_model) omskriver tensortyper i grafen, men lar
  eksisterende Cast-noders `to`-attributt staa urort. Dual-planes interne `.float()`-kast
  (dynamo → kjedede `_to_copy`-noder) ga dermed en selvmotsigende graf.
- onnxruntime NEKTER aa laste fila ("Type Error: Type (tensor(float16)) of output arg
  (_to_copy_1) ... does not match expected type (tensor(float))").
- TensorRT parser den likevel og bygger en SUBTILT FEIL motor → nett som "virker" men
  spiller svakt. Dette er fella: EB/TRT gir ingen feilmelding.

**Fix (in-tree):** avstemmingspass i `save_model.py` etter fp16-konverteringen — alle
FLOAT<->FLOAT16-Cast-attributter settes til aa matche konvertert utgangstype. Logger
`INFO: ONNX_FP16_CAST_RECONCILED N` naar den bidrar. Systemisk: all fremtidig modulkode
med interne `.float()`/`.half()` er dekket.

**Validering (lokalt, mot server-ckpt deg4@100M):** foer = ORT-load feiler eksakt som over;
etter = laster rent, 0 NaN, og maaler som treningsmetrikkene tilsier
(top1 54.9 % / pTop3 82.3 % / valAcc 91.6 % paa T91-skip1 — paritet med 256ctrl@100M).
⇒ gameplay-dommen over deg4@100M er UGYLDIG; retest med fikset eksport.

**Praktisk:**
- Kjørende treningsprosesser har gammel save_model i minnet: deres eksporter maa
  re-eksporteres med `recover_export.py <ID> <outputs_dir> <numpos>` etter pull.
  Checkpointene er friske; ingen retrening.
- Lastbarhets-test = perfekt CI: `onnxruntime.InferenceSession(path)` groenn/roed pa sekundet.

**Bifunn (gjelder ogsaa uten dual-plane):** 512-BASELINENS tidlige eksporter (100M/200M,
LR 1.2e-3) er FP16-SKJOERE: strikt fp16 (ORT) gir soeppel (top1 11 %!), TRT fungerte fordi
den selv velger fp32 der det trengs. Mekanisme: ekstreme vekter/logiter i den varme fasen
(QK-klipp-regimet) satureres av fp16-konverteringen; fra ~500M er filene friske. EB-tall
(TRT) staar seg, men IKKE bruk tidlige 512@1.2e-3-onnx i ORT-baserte maalinger.
deg4 (LR 0.6e-3) er frisk allerede ved 100M — nok et argument for lavere peak-LR.

