"""Writes the example configs of the Fry blocks next to this file (not launched by anything).

  python make_fry_configs.py [dir of the 384 base arm configs]

  parity_768_28_fry117                                      Fry itself in our trainer (for test_fry_parity.py; not a training config)
  prod_384_12_fryfaithful_t80t91_300M                       384 x 12 with all four blocks and his stage-3 style recipe
  prod_384_12_fry_hlg_tpl8qkbt6sqfAnt135notripei_t80t91_300M    base arm + SquareFeatures only
  prod_384_12_fry_hlg_tpl8qkbt6gqaAnt135notripei_t80t91_300M    base arm + grouped-query / narrow-QK attention only

Base arm: prod_384_12_fry_hlg_tpl8qkbt6Ant135notripei_t80t91_300M (flat BT6 trunk, templates, QK-norm, decoder,
HL-Gauss served value). The single-idea arms are the base files with the listed keys added and nothing else changed.
"""
import copy
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = sys.argv[1] if len(sys.argv) > 1 else '/mnt/c/Dev/Chess/lc0ex_work/terms/configs_notripei'
BASE_ID = 'prod_384_12_fry_hlg_tpl8qkbt6Ant135notripei_t80t91_300M'
PARTS = ('net', 'opt', 'data', 'exec', 'monitoring')


def load(cid):
  return {p: json.load(open(os.path.join(BASE_DIR, f'{cid}_ceres_{p}.json'))) for p in PARTS}


def save(cid, cfg):
  cfg = copy.deepcopy(cfg)
  cfg['exec']['ID'] = cid
  for p in PARTS:
    with open(os.path.join(HERE, f'{cid}_ceres_{p}.json'), 'w', newline='\n') as f:
      json.dump(cfg[p], f, indent=2)
      f.write('\n')
  print('wrote', cid)


def drop(d, *keys):
  for k in keys:
    d.pop(k, None)


base = load(BASE_ID)

# ---- single-idea arm: square features on the base arm's embedding
sqf = copy.deepcopy(base)
sqf['net']['SquareFeatures'] = True
save('prod_384_12_fry_hlg_tpl8qkbt6sqfAnt135notripei_t80t91_300M', sqf)

# ---- single-idea arm: 12 query heads on 6 key/value heads, q/k width 8 / 16 / 24 by depth (0.25 / 0.5 / 0.75 of the
#      32-wide values: layers 0-1 / 2-4 / 5-11, the proportions of Fry's 28 layers), no projection bias (kept out:
#      one idea per arm)
gqa = copy.deepcopy(base)
gqa['net'].update({'AttnKVHeads': 6, 'AttnQKDims': [8, 8, 16, 16, 16, 24, 24, 24, 24, 24, 24, 24], 'AttnValueDim': 32})
save('prod_384_12_fry_hlg_tpl8qkbt6gqaAnt135notripei_t80t91_300M', gqa)

# ---- the faithful 384 x 12 arm
QK_384 = [8, 8, 16, 16, 16, 24, 24, 24, 24, 24, 24, 24]
fnet = {
  'TrainOn4BoardSequences': False,
  'ModelDim': 384, 'NumLayers': 12, 'NumHeads': 12, 'LoRARankDivisor': 0,
  'UseQKV': True, 'DualAttentionMode': 'None', 'PreNorm': True, 'NormType': 'RMSNorm', 'NormEps': 1e-5,
  'AttentionMultiplier': 1,
  'FFNMultiplier': 992 / 768, 'FFNActivationType': 'SwiGLU', 'FFNNoBias': True, 'FFNUseGlobalEveryNLayers': 0,
  'HeadsActivationType': 'Mish', 'HeadWidthMultiplier': 4, 'PriorStateDim': 0,
  'NonLinearAttention': False, 'SoftCapCutoff': 0, 'UseQKNorm': True, 'DeepNorm': False, 'DenseFormer': False,
  'SmolgenDimPerSquare': 0, 'SmolgenDim': 0, 'SmolgenToHeadDivisor': 4, 'SmolgenActivationType': 'Swish',
  'UseRPE': False, 'UseRPE_V': False, 'UseRoPE': False, 'UseRelBias': False,
  'SoftMoEConfig': base['net']['SoftMoEConfig'], 'TSBConfig': base['net']['TSBConfig'],
  'UseMoveTokens': False, 'NBTInnerLayers': 0,
  'AttnTemplates': 8, 'AttnTemplateScope': 'layer',
  'NBTInputSkip': True, 'InputSkipLayer0': True,
  'FryInput': True, 'SquareFeatures': True,
  'AttnKVHeads': 6, 'AttnQKDims': QK_384, 'AttnValueDim': 32, 'AttnQKVBias': True,
  'PolicyHeadFry': True,
  'ValueHeadFry': True, 'ValueHLGaussServe': True, 'ValueHLGaussServeDraw': 'wdl',
}
fopt = copy.deepcopy(base['opt'])
fopt.update({
  'Optimizer': 'Muon', 'MuonAdamWScope': 'fry', 'MuonMomentum': 0.95, 'Beta1': 0.9, 'Beta2': 0.999,
  'WeightDecayScheme': 'fry', 'WeightDecayFromSchedule': 0.17, 'WeightDecay': None,
  'MuonHonorNoDecay': False,
  'LearningRateBase': 0.0009, 'LRBeginDecayAtFractionComplete': 1.0, 'LRWarmupPhaseMultiplier': 0.03,
  'LearningRateHeadsRatio': None, 'LearningRateBaseHeads': None,
  'GradientClipLevel': 1.0,
  'LossPolicyMultiplier': 1, 'HLGaussWeight': 0.5, 'HLGaussBuckets': 128,
  'LossValueMultiplier': 0.1,                      # the WDL head, for the served draw probability only (Fry has none)
  'LossValue2Multiplier': 0, 'LossMLHMultiplier': 0, 'LossUNCMultiplier': 0, 'LossQDeviationMultiplier': 0,
  'LossUncertaintyPolicyMultiplier': 0, 'LossValueDMultiplier': 0, 'LossValue2DMultiplier': 0,
  'PolicyPLWeight': 0, 'LossMoveTokenValueOrderMultiplier': 0, 'DepthProbeValueWeight': 0.0, 'DepthProbePolicyWeight': 0.0,
  'EMAPeriodSteps': -1,
})
drop(fopt, 'MuonHonorNoDecayScope', 'MoveTokenValueOrderTopK', 'PolicyPLTopK')
fdata = copy.deepcopy(base['data'])
fdata['FractionQ'] = 0.8                           # value target 0.8 search q + 0.2 result (ours: the deblundered result)
save('prod_384_12_fryfaithful_t80t91_300M', {'net': fnet, 'opt': fopt, 'data': fdata, 'exec': base['exec'], 'monitoring': base['monitoring']})

# ---- Fry itself (parity; fry_parity.FRY_NET / FRY_OPT are the same dictionaries)
sys.path.insert(0, os.path.join(HERE, '..', '..', 'src', 'CeresTrainPy'))
import fry_parity as FP
pexec = copy.deepcopy(base['exec'])
pexec.update({'DeviceType': 'cpu', 'DeviceIDs': [0]})
pdata = {'SourceType': 'DirectFromPositionGenerator', 'TrainingFilesDirectory': '/none', 'FractionQ': 0.8, 'WDLLabelSmoothing': 0}
save('parity_768_28_fry117', {'net': FP.FRY_NET, 'opt': FP.FRY_OPT, 'data': pdata, 'exec': pexec, 'monitoring': base['monitoring']})
