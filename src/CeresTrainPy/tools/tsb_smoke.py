"""TSB smoke test: verify zero-init TSB is bit-identical to non-TSB at step 0.

Standalone script. Constructs a tiny CeresNet twice with the same architecture —
once without TSB, once with TSB enabled — copies the non-TSB weights into the
TSB-enabled instance, runs a fixed batch through both, and asserts the outputs
are exactly equal (atol=0 in fp32; the test tolerates tiny bf16 noise).

Run with the cerestrain-env active:

    cd /mnt/c/Users/lepne/source/repos/CeresTrain/src/CeresTrainPy
    python tools/tsb_smoke.py
"""
import sys
import os
import torch

# Ensure imports resolve from the parent CeresTrainPy directory.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import Configuration, NUM_TOKENS_INPUT, NUM_INPUT_BYTES_PER_SQUARE
from ceres_net import CeresNet


class _StubFabric:
    """Minimal fabric stand-in so CeresNet can be constructed without lightning."""
    is_global_zero = True
    device = torch.device('cpu')


def _make_config(tsb_enabled: bool):
    """Hand-build a Configuration without going through JSON files."""
    cfg = Configuration.__new__(Configuration)
    cfg.id = 'tsb_smoke'

    # Data section (defaults).
    cfg.Data_SourceType = 'DirectFromTPG'
    cfg.Data_PositionGenerator = {}
    cfg.Data_TrainingFilesDirectory = None
    cfg.Data_NumTPGFilesToSkip = 0
    cfg.Data_FractionQ = 1.0
    cfg.Data_WDLLabelSmoothing = 0.0

    # Exec section (defaults).
    cfg.Exec_ID = 'tsb_smoke'
    cfg.Exec_DeviceType = 'cpu'
    cfg.Exec_DeviceIDs = [0]
    cfg.Exec_DataType = 'BFloat16'
    cfg.Exec_UseFP8 = False
    cfg.Exec_DropoutRate = 0
    cfg.Exec_DropoutDuringInference = False
    cfg.Exec_EngineType = 'CSharpViaTorchscript'
    cfg.Exec_SaveNetwork1FileName = None
    cfg.Exec_SaveNetwork2FileName = None
    cfg.Exec_ActivationMonitorDumpSkipCount = 0
    cfg.Exec_SupplementaryStat = None
    cfg.Exec_TrackFinalLayerIntrinsicDimensionality = False
    cfg.Exec_MonitorActivationStats = False
    cfg.Exec_ExportOnly = False
    cfg.Exec_TestFlag = False
    cfg.Exec_TestValue = 0

    # Opt section.
    cfg.Opt_LoRARankDivisor = 0
    cfg.Opt_LoRARestrictPolicyValueOnly = False
    cfg.Opt_LoRARestrictValueOnly = False
    cfg.Opt_NumTrainingPositions = 1000
    cfg.Opt_BatchSizeForwardPass = 4
    cfg.Opt_BatchSizeBackwardPass = 4
    cfg.Opt_Optimizer = 'AdamW'
    cfg.Opt_CheckpointResumeFromFileName = None
    cfg.Opt_CheckpointFrequencyNumPositions = 1000000
    cfg.Opt_PyTorchCompileMode = None
    cfg.Opt_WeightDecay = 0.005
    cfg.Opt_LearningRateBase = 1e-4
    cfg.Opt_LRBeginDecayAtFractionComplete = 0.5
    cfg.Opt_Beta1 = 0.95
    cfg.Opt_Beta2 = 0.999
    cfg.Opt_Beta3 = 0.9999
    cfg.Opt_Alpha = 5
    cfg.Opt_GradientClipLevel = 1
    cfg.Opt_LossValueMultiplier = 1.0
    cfg.Opt_LossValue2Multiplier = 0.0
    cfg.Opt_LossValueDMultiplier = 0
    cfg.Opt_LossValue2DMultiplier = 0
    cfg.Opt_LossUncertaintyPolicyMultiplier = 0.01
    cfg.Opt_LossActionMultiplier = 0
    cfg.Opt_LossActionUncertaintyMultiplier = 0
    cfg.Opt_LossQDeviationMultiplier = 0.02
    cfg.Opt_LossPolicyMultiplier = 1.0
    cfg.Opt_LossMLHMultiplier = 0
    cfg.Opt_LossUNCMultiplier = 0.01
    cfg.Opt_KLAnchorRefCheckpoint = None
    cfg.Opt_KLAnchorPolicyWeight = 0.0
    cfg.Opt_KLAnchorValueWeight = 0.0
    cfg.Opt_TestValue = 0

    # NetDef section — small but realistic SwiGLU net.
    cfg.NetDef_TrainOn4BoardSequences = False
    cfg.NetDef_ModelDim = 64
    cfg.NetDef_NumLayers = 2
    cfg.NetDef_UsePieceRelationBias = False
    cfg.NetDef_LoopCount = 1
    cfg.NetDef_NumHeads = 4
    cfg.NetDef_UseQKV = True
    cfg.NetDef_DualAttentionMode = 'None'
    cfg.NetDef_PreNorm = False
    cfg.NetDef_NormType = 'RMSNorm'
    cfg.NetDef_AttentionMultiplier = 1
    cfg.NetDef_NonLinearAttention = False
    cfg.NetDef_FFNMultiplier = 4
    cfg.NetDef_FFNActivationType = 'SwiGLU'
    cfg.NetDef_FFNUseGlobalEveryNLayers = 0
    cfg.NetDef_HeadsActivationType = 'Mish'
    cfg.NetDef_PriorStateDim = 0
    cfg.NetDef_DeepNorm = False
    cfg.NetDef_DenseFormer = False
    cfg.NetDef_SmolgenDimPerSquare = 0
    cfg.NetDef_SmolgenDim = 0
    cfg.NetDef_SmolgenToHeadDivisor = 0
    cfg.NetDef_SmolgenActivationType = 'None'
    cfg.NetDef_HeadWidthMultiplier = 2
    cfg.NetDef_UseRPE = False
    cfg.NetDef_UseRPE_V = True
    cfg.NetDef_UseRelBias = False
    cfg.NetDef_UseQKNorm = False
    cfg.NetDef_SoftCapCutoff = 0
    cfg.NetDef_TestValue = 0
    cfg.NetDef_SoftMoE_MoEMode = 'None'
    cfg.NetDef_SoftMoE_OnlyForAlternatingLayers = False
    cfg.NetDef_SoftMoE_NumExperts = 0
    cfg.NetDef_SoftMoE_NumSlotsPerExpert = 0
    cfg.NetDef_SoftMoE_UseNormalization = False
    cfg.NetDef_SoftMoE_UseBias = False

    # TSB section — only difference between the two configs.
    cfg.NetDef_TSB_Enabled = tsb_enabled
    cfg.NetDef_TSB_FFNMultiplier = 1
    cfg.NetDef_TSB_GateBiasInit = -4.0
    cfg.NetDef_TSB_GateMLPHiddenDivisor = 8

    return cfg


def _make_net(cfg):
    fabric = _StubFabric()
    return CeresNet(fabric, cfg,
                    policy_loss_weight=cfg.Opt_LossPolicyMultiplier,
                    value_loss_weight=cfg.Opt_LossValueMultiplier,
                    moves_left_loss_weight=cfg.Opt_LossMLHMultiplier,
                    unc_loss_weight=cfg.Opt_LossUNCMultiplier,
                    value2_loss_weight=cfg.Opt_LossValue2Multiplier,
                    q_deviation_loss_weight=cfg.Opt_LossQDeviationMultiplier,
                    value_diff_loss_weight=cfg.Opt_LossValueDMultiplier,
                    value2_diff_loss_weight=cfg.Opt_LossValue2DMultiplier,
                    action_loss_weight=cfg.Opt_LossActionMultiplier,
                    uncertainty_policy_weight=cfg.Opt_LossUncertaintyPolicyMultiplier,
                    action_uncertainty_loss_weight=cfg.Opt_LossActionUncertaintyMultiplier,
                    q_ratio=cfg.Data_FractionQ)


def main():
    torch.manual_seed(42)

    cfg_off = _make_config(tsb_enabled=False)
    cfg_on = _make_config(tsb_enabled=True)

    net_off = _make_net(cfg_off).eval()
    net_on = _make_net(cfg_on).eval()

    # Copy non-TSB weights into the TSB-enabled net so non-TSB params are identical.
    src = net_off.state_dict()
    dst = dict(net_on.state_dict())
    for k, v in src.items():
        if k in dst:
            dst[k].copy_(v)
    net_on.load_state_dict(dst, strict=False)

    # Spot-check that TSB params are present and zero where they need to be.
    tsb_param_names = [n for n, _ in net_on.named_parameters() if 'tactical_ffn' in n or 'tactical_gate' in n]
    print(f"TSB param tensors: {len(tsb_param_names)}")
    for n, p in net_on.named_parameters():
        if 'tactical_ffn_linear3' in n:
            assert torch.all(p == 0), f"{n} not zero-initialized"
        if 'tactical_gate_fc2.weight' in n:
            assert torch.all(p == 0), f"{n} not zero-initialized"
        if 'tactical_gate_fc2.bias' in n:
            assert torch.allclose(p, torch.full_like(p, -4.0)), f"{n} not -4 init"
    print("TSB init invariants verified (linear3=0, gate_fc2 weight=0, bias=-4).")

    # Forward both nets on identical input.
    torch.manual_seed(0)
    B = 2
    squares = torch.randn(B, NUM_TOKENS_INPUT, NUM_INPUT_BYTES_PER_SQUARE)

    with torch.no_grad():
        out_off = net_off(squares, None)
        out_on = net_on(squares, None)

    # Compare key outputs (policy_out at index 0, value_out at index 1).
    policy_off, value_off = out_off[0], out_off[1]
    policy_on, value_on = out_on[0], out_on[1]

    pol_max_diff = (policy_off - policy_on).abs().max().item()
    val_max_diff = (value_off - value_on).abs().max().item()

    print(f"Policy max abs diff: {pol_max_diff:.3e}")
    print(f"Value  max abs diff: {val_max_diff:.3e}")

    # In fp32 with zero-init linear3, the additive residual contributes exactly 0,
    # so outputs must be bit-identical.
    assert torch.equal(policy_off, policy_on), \
        f"Policy outputs differ: max abs diff {pol_max_diff:.3e}"
    assert torch.equal(value_off, value_on), \
        f"Value outputs differ: max abs diff {val_max_diff:.3e}"
    print("PASS: TSB-enabled forward is bit-identical to non-TSB forward at init.")


if __name__ == '__main__':
    main()
