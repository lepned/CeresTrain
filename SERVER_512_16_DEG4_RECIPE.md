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

## Net-config (ceres_net.json) — delta mot server-baseline

```json
{
    "ModelDim": 512,
    "NumLayers": 16,

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

## Gate-regel (uendret fra Stage C-protokollen)

Sammenlign mot 512-16-baselinen ved samme posisjonstall; value-regelen
≥ +30 @200M rg2700 star. Les OOD (rg2100/2300-klassen) for
generaliseringsdommer og hits i gulvregimet — og for policy: pT3 +
KLD-matrisen, ikke bare P.
