﻿#region License notice

/*
  This file is part of the CeresTrain project at https://github.com/dje-dev/cerestrain.
  Copyright (C) 2023- by David Elliott and the CeresTrain Authors.

  Ceres is free software under the terms of the GNU General Public License v3.0.
  You should have received a copy of the GNU General Public License
  along with CeresTrain. If not, see <http://www.gnu.org/licenses/>.
*/

#endregion

#region Using directives

using System.Text.Json.Serialization;

#endregion

namespace CeresTrain.Trainer
{
  public enum OptimizerType
  {
    /// <summary>
    /// Stochastic gradient descent optimizer.
    /// </summary>
    SGD,

    /// <summary>
    ///  Adam optimizer.
    /// </summary>
    AdamW,

    /// <summary>
    /// Adam optimizer with 8-bit quantization.
    /// </summary>
    AdamW8bit,

    /// <summary>
    /// Nadam optimizer (with decoupled weight decay).
    /// 
    /// The paper "Benchmarking Neural Network Training Algorithms" Dahl et al. 2023
    /// notes that NAdam matches our outperforms all other tested optimizers in all configurations
    /// (for transformer network).
    /// </summary>
    NAdamW,

    /// <summary>
    /// Schedule-free Adam optimizer.
    /// See "The Road Less Scheduled by Defazio et. al. at https://arxiv.org/abs/2405.15682)
    /// Code from official implementation at https://github.com/facebookresearch/schedule_free/blob/main/schedulefree/adamw_schedulefree.py.
    /// </summary>
    AdamWScheduleFree,

    /// <summary>
    /// AdEMAMix optimizer (Pagliardini et al. 2024) https://arxiv.org/abs/2409.03137
    /// See: https://github.com/nanowell/AdEMAMix-Optimizer-Pytorch
    /// </summary>
    AdEMAMix,

    /// <summary>
    /// AdEMAMix-Shampoo optimizer (Pagliardini et al. 2024) https://arxiv.org/abs/2409.03137
    /// See: https://github.com/nanowell/AdEMAMix-Optimizer-Pytorch
    /// </summary>
    AdEMAMixShampoo,

    /// <summary>
    /// "SOAP: Improving and Stabilizing Shampoo using Adam" by Vyal et al. (https://arxiv.org/abs/2409.11321)
    /// Code from official implementation at https://github.com/nikhilvyas/SOAP/blob/main/soap.py
    /// </summary>
    SOAP,

    /// <summary>
    /// "Muon: An optimizer for the hidden layers of neural networks"
    /// Code from the author's implementation at https://github.com/KellerJordan/Muon.
    /// </summary>
    Muon
  }


  /// <summary>
  /// Parameters related to optimization.
  /// </summary>
  public readonly record struct ConfigOptimization
  {
    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="numTrainingPositions"></param>
    public ConfigOptimization(long numTrainingPositions) : this()
    {
      NumTrainingPositions = numTrainingPositions;
    }

    /// <summary>
    /// Default constructor for deserialization.
    /// </summary>
    [JsonConstructor]
    public ConfigOptimization()
    {
    }

    /// <summary>
    /// If nonzero, LoRA fine-tuning mode is enabled.
    /// See: "LoRA: Low-Rank Adaptation of Large Language Models" by Hu et. al. (https://arxiv.org/abs/2106.09685)
    /// The model parameters are frozen except for some inserted LoRA layers.
    /// The rank of the layer is some divisor of the rank of the input dimension.
    /// </summary>
    public readonly int LoRARankDivisor { get; init; } = 0;


    /// <summary>
    /// Number of training positions to use before halting training.
    /// </summary>
    public readonly long NumTrainingPositions { get; init; } = int.MaxValue;

    /// <summary>
    /// Batch size used each forward pass.
    /// </summary>
    public readonly int BatchSizeForwardPass { get; init; } = 2048;

    /// <summary>
    /// Batch size used for each backward pass (possibly larger than BatchSizeForwardPass if accumulating gradients).
    /// </summary>
    public readonly int BatchSizeBackwardPass { get; init; } = 2048;

    /// <summary>
    /// Type of optimizer.
    /// </summary>
    public readonly OptimizerType Optimizer { get; init; } = OptimizerType.AdamW;

    /// <summary>
    /// The (approximate) number of positions between successive checkpoints.
    /// </summary>
    public readonly int CheckpointFrequencyNumPositions { get; init; } = 200_000_000;

    /// <summary>
    /// Optional name of file containing the starting checkpoint from which training will be resumed,
    /// or none if training is to start from scratch.
    /// </summary>
    public readonly string CheckpointResumeFromFileName { get; init; }

    /// <summary>
    /// String to be used for model argument of the PyTorch compile method (or null for no compile).
    /// Valid values: "default", "reduce-overhead", or "max-autotune" or "max-autotune-no-cudagraphs"
    /// with "max-autotune" often giving best training speed but slowing down initialization.
    /// </summary>
    public readonly string PyTorchCompileMode { get; init; } = "default";

    /// <summary>
    /// Weight decay coefficient for the optimizer.
    /// A small weight decay may help stabilize training or possibly enhance generalization.
    /// </summary>
    public readonly float WeightDecay { get; init; } = 0.01f;

    /// <summary>
    /// Maximum learning rate to be used during optimization.
    /// Learning rate typically needs to be lower for small batch sizes and/or larger networks.
    /// </summary>
    public readonly float LearningRateBase { get; init; } = 6E-4f;

    /// <summary>
    /// Fraction complete (between 0 and 1) at which scaling down of the LearningRateBase begins 
    /// (linearly from LearningRateBase to a fixed minimum value of 0.10x starting rate).
    /// For short(long)-duration training values of 0.6 (0.4) may be good choices.
    /// </summary>
    public readonly float LRBeginDecayAtFractionComplete { get; init; } = 0.6f;

    /// <summary>
    /// Multiplier applied to learning rate base during the warmup phase.
    /// TODO: Currently PyTorch version does not respect this, and uses a different warmup shape.
    /// </summary>
    public readonly float LRWarmupPhaseMultiplier { get; init; } = 0.1f;

    /// <summary>
    /// Beta 1 coefficient used with optimizers such as Adam, AdamW, or NAdamW.
    /// </summary>
    public readonly float Beta1 { get; init; } = 0.95f;

    /// <summary>
    /// Beta 2 coefficient used with optimizers such as Adam, AdamW, or NAdamW.
    /// </summary>
    public readonly float Beta2 { get; init; } = 0.95f;

    /// <summary>
    /// Beta 3 coefficient used with AdEMAMix optimizers.
    /// </summary>
    public readonly float Beta3 { get; init; } = 0.9999f;

    /// <summary>
    /// Alpha coefficient used with AdEMAMix optimizers.
    /// </summary>
    public readonly float Alpha { get; init; } = 5;


    /// <summary>
    /// Value at which gradients are clipped on each optimizer step 
    /// (clipping is disabled if this value is 0.0).
    /// 
    /// Clipping may stabilize training.
    /// Contrary to intuition, low values (such as 0.5) 
    /// may actually increase speed of convergence, see:
    /// "Why gradient clipping accelerates training: A theoretical justification for adaptivity" Zhang et. al.
    /// https://arxiv.org/abs/1905.11881
    /// </summary>
    public readonly float GradientClipLevel { get; init; } = 1.0f;


    #region Loss multipliers

    /// <summary>
    /// Scaling multiplier to be applied to primary value loss term.
    /// </summary>
    public readonly float LossValueMultiplier { get; init; } = 1.0f;

    /// <summary>
    /// Scaling multiplier to be applied to secondary value loss term.
    /// Typically a lower coefficient is used here because it is very noisy.
    /// </summary>
    public readonly float LossValue2Multiplier { get; init; } = 0.04f;

    /// <summary>
    /// Scaling multiplier to be applied to policy loss term.
    /// </summary>
    public readonly float LossPolicyMultiplier { get; init; } = 1.5f;

    /// <summary>
    /// Scaling multiplier to be applied to MLH loss term.
    /// </summary>
    public readonly float LossMLHMultiplier { get; init; } = 0.0f;

    /// <summary>
    /// Scaling multiplier to be applied to value head uncertainty loss term.
    /// Coefficient typically small due to low importance in gameplay and relatively high noise.
    /// </summary>
    public readonly float LossUNCMultiplier { get; init; } = 0.005f;

    /// <summary>
    /// Scaling multiplier to be applied to estimates of lower and upper deviation bounds of forward Q.
    /// Coefficient typically small due to low importance in gameplay and relatively high noise.
    /// </summary>
    public readonly float LossQDeviationMultiplier { get; init; } = 0.01f;

    /// <summary>
    /// Scaling multiplier to be applied to policy uncertainty term.
    /// </summary>
    public readonly float LossUncertaintyPolicyMultiplier { get; init; } = 0.02f;


    /// <summary>
    /// Scaling multiplier to be applied to difference in value scores between consecutive positions.
    /// Only available when TrainOn4BoardSequences is true.
    /// This acts as consistency regularizer.
    /// Seems to have small positive benefit, especially for action head accuracy.
    /// </summary>
    public readonly float LossValueDMultiplier { get; init; } = 0.1f;

    /// <summary>
    /// Scaling multiplier to be applied to difference in value2 scores between consecutive positions.
    /// Only available when TrainOn4BoardSequences is true.
    /// This acts as consistency regularizer.
    /// Benefit is unclear for Value2 because this training target is so noisy.
    /// </summary>
    public readonly float LossValue2DMultiplier { get; init; } = 0.0f;

    /// <summary>
    /// Loss weight applied to error in action prediction (relative to actual value2 from position).
    /// Only available when TrainOn4BoardSequences is true.
    /// </summary>
    public readonly float LossActionMultiplier { get; init; } = 0.3f;

    /// <summary>
    /// Scaling multiplier to be applied to action value uncertainty term.
    /// </summary>
    public readonly float LossActionUncertaintyMultiplier { get; init; } = 0.01f;

    #endregion

    /// <summary>
    /// Reserved value used for debugging/experimentation to turn on a possible ad hoc test/diagnostic feature.
    /// </summary>
    public readonly float TestValue { get; init; } = 0;
  }
}
