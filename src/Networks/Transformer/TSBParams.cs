#region License notice

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

namespace CeresTrain.Networks.Transformer
{
  /// <summary>
  /// Parameters that define the Tactical SwiGLU Bypass (TSB) feature.
  ///
  /// TSB adds a per-block parallel SwiGLU FFN beside the original FFN of each
  /// transformer block, combined via a per-block scalar gate using an additive
  /// residual: output = sp_ffn + g * tactical_ffn. The tactical branch is zero-
  /// initialized so the network's forward at step 0 is bit-identical to a
  /// non-TSB network. Training (typically with the body and heads frozen) lets
  /// the gate selectively open in tactical layers, producing a strong
  /// tactical-puzzle responder while preserving the original general-play
  /// behavior anywhere the gate stays closed.
  /// </summary>
  public readonly record struct TSBParams
  {
    /// <summary>
    /// Default constructor for deserialization.
    /// </summary>
    [JsonConstructor]
    public TSBParams()
    {
    }

    /// <summary>
    /// Constructor with explicit values for all fields.
    /// </summary>
    public TSBParams(bool enabled, int ffnMultiplier, float gateBiasInit, int gateMLPHiddenDivisor)
    {
      Enabled = enabled;
      FFNMultiplier = ffnMultiplier;
      GateBiasInit = gateBiasInit;
      GateMLPHiddenDivisor = gateMLPHiddenDivisor;
    }

    /// <summary>
    /// If true, each transformer block instantiates a parallel TSBSwiGLU branch
    /// alongside the original FFN. When false, no TSB code paths execute and
    /// existing checkpoints/runs are unaffected.
    /// </summary>
    public readonly bool Enabled { get; init; } = false;

    /// <summary>
    /// Inner-dimension multiplier for the tactical FFN (relative to ModelDim).
    /// 1 = same size as ModelDim hidden, 2 = double, etc. Smaller multipliers
    /// reduce inference cost (the parallel FFN is the dominant added compute).
    /// Default 1: minimal capacity, ~+15% inference cost over baseline.
    /// </summary>
    public readonly int FFNMultiplier { get; init; } = 1;

    /// <summary>
    /// Initial bias of the gate's final linear layer. Default -4.0 produces
    /// sigmoid(-4) ~ 0.018, so the gate starts essentially closed. Combined
    /// with the zero-initialized tactical-FFN output, this guarantees the
    /// network's forward is bit-identical to a non-TSB network at step 0.
    /// </summary>
    public readonly float GateBiasInit { get; init; } = -4.0f;

    /// <summary>
    /// Divisor for the gate MLP's hidden dimension: gate_hidden = ModelDim /
    /// GateMLPHiddenDivisor. Default 8. The gate MLP is tiny (~37K params per
    /// layer at ModelDim 544), so this knob has negligible compute impact.
    /// </summary>
    public readonly int GateMLPHiddenDivisor { get; init; } = 8;
  }
}
