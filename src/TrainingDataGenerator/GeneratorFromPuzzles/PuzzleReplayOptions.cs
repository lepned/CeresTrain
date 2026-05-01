#region License notice

/*
  This file is part of the CeresTrain project at https://github.com/dje-dev/cerestrain.
  Copyright (C) 2023- by David Elliott and the CeresTrain Authors.

  Ceres is free software under the terms of the GNU General Public License v3.0.
  You should have received a copy of the GNU General Public License
  along with CeresTrain. If not, see <http://www.gnu.org/licenses/>.
*/

#endregion

using System;
using System.IO;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace CeresTrain.TrainingDataGenerator.GeneratorFromPuzzles
{
  /// <summary>
  /// Configuration for a single end-to-end puzzle action-replay run (mine -> label -> TPG).
  /// </summary>
  public sealed class PuzzleReplayOptions
  {
    public string LichessCsvPath { get; set; }

    public string NetSpec { get; set; }

    public string Device { get; set; } = "GPU:0";

    /// <summary>
    /// Optional separate NetSpec for the action-head teacher used by `enrich-action-head`.
    /// When null, falls back to <see cref="NetSpec"/>. The configured net MUST have an
    /// action head (HasAction=true) — `enrich-action-head` throws otherwise.
    /// Example: "ONNX_TRT16:C:/Dev/Chess/Networks/CeresNet/C3-768-30-pre3-I8.onnx".
    /// </summary>
    public string ActionNetSpec { get; set; }

    /// <summary>
    /// Optional separate Device for <see cref="ActionNetSpec"/>. When null, falls back to <see cref="Device"/>.
    /// </summary>
    public string ActionDevice { get; set; }

    /// <summary>
    /// `enrich-action-head`: number of non-solver legal moves to emit OAIS records for, per parent.
    /// If a parent has fewer than this many non-solver legal moves, all are emitted.
    /// Default 2 (lowered from 4 on 2026-04-28: OAIS records only teach
    /// "any deviation = punished"; 2 random samples per parent gives sufficient
    /// off-path supervision without inflating the dataset 2× for diminishing returns).
    /// </summary>
    public int OAISSamplesPerParent { get; set; } = 2;

    /// <summary>
    /// `enrich-action-head`: ε margin for the rank-1 nudge that forces the solver move's
    /// action[L] to dominate the cross-move L array. Set <see cref="ActionNetSpec"/>=null
    /// to disable action-head enrichment entirely. Default 0.03 (matches PuzzleSoftLabeler convention).
    /// </summary>
    public float RankOneEpsilon { get; set; } = 0.03f;

    public int MinRating { get; set; } = 1800;
    public int MaxRating { get; set; } = 3200;

    public string ThemeIncludeAny { get; set; }
    public string ThemeExcludeAny { get; set; }

    public int MaxPuzzlesToRead { get; set; } = int.MaxValue;

    /// <summary>
    /// If &gt; 0, randomly sub-samples this many hard records before labeling.
    /// Useful for controlling label-stage wall time without re-mining.
    /// </summary>
    public int MaxRecordsToLabel { get; set; } = 0;

    /// <summary>Seed for reproducible sub-sampling (used only when MaxRecordsToLabel &gt; 0).</summary>
    public int LabelSubsampleSeed { get; set; } = 42;

    /// <summary>
    /// If true, the label stage reads puzzle positions directly from the Lichess CSV
    /// (expanding each puzzle to all solver-to-move positions) instead of from hard.jsonl.
    /// Used when training a specialist from scratch on ALL puzzles, not just adversarial
    /// ones against a specific student. Mining step is skipped when this is true.
    /// </summary>
    public bool SkipMining { get; set; } = false;

    /// <summary>
    /// If true and existing labeled.jsonl / rejected.jsonl files are found in OutDir,
    /// resume labeling by skipping any input already represented (keyed on PuzzleId+FEN).
    /// Output files are opened in append mode. Safe to use on interrupted runs.
    /// </summary>
    public bool ResumeFromCheckpoint { get; set; } = false;

    /// <summary>
    /// If &gt; 0, limits the number of records read from labeled.jsonl during eval-labeled.
    /// Useful for quick smoke tests. 0 = evaluate all records.
    /// </summary>
    public int MaxEvalRecords { get; set; } = 0;

    /// <summary>
    /// If true, eval-labeled only processes puzzle-starting positions (where PriorUciMoves
    /// contains exactly the setup move) — matches EB's test setup that evaluates the first
    /// solver move of each puzzle.
    /// </summary>
    public bool EvalStartingPositionsOnly { get; set; } = false;

    /// <summary>
    /// Optional rating-bin thresholds used to oversample harder puzzles during TPG generation.
    /// N thresholds define N+1 bins: (-∞, T0), [T0, T1), ..., [T_{N-1}, +∞).
    /// Each bin's multiplicity comes from RatingBinWeights at the matching index.
    /// Null or empty = no stratification (every record emitted exactly once).
    /// Example: Thresholds=[1400,1800,2200,2600], Weights=[1,1,3,8,16].
    /// </summary>
    public int[] RatingBinThresholds { get; set; }

    /// <summary>
    /// Parallel to RatingBinThresholds — one more entry than Thresholds.
    /// Weight N means each record in that bin is written N times into the TPG.
    /// Null or empty = no stratification.
    /// </summary>
    public int[] RatingBinWeights { get; set; }

    public int TeacherNodes { get; set; } = 100;
    public int MineBatchSize { get; set; } = 512;
    public int TeacherWorkerThreads { get; set; } = 4;

    /// <summary>oppdef-deepen-smoke: re-search OppDef records with |TeacherV|&lt; this at DeepenNodes.</summary>
    public float DeepenQThreshold { get; set; } = 0.2f;
    /// <summary>oppdef-deepen-smoke: how many filtered OppDef records to re-search.</summary>
    public int DeepenSampleN { get; set; } = 1000;
    /// <summary>oppdef-deepen-smoke: search-node budget for the deeper re-search.</summary>
    public int DeepenNodes { get; set; } = 400;

    public string OutDir { get; set; }

    [JsonIgnore]
    public string HardJsonlPath => Path.Combine(OutDir, "hard.jsonl");

    /// <summary>
    /// Optional override for the labeled JSONL input path used by puzzles-to-tpg
    /// and eval-labeled. When null, defaults to &lt;OutDir&gt;/labeled.jsonl.
    /// Set to e.g. "labeled_enriched.jsonl" to use the multi-sided value-enriched
    /// file produced by `enrich-value-labels`.
    /// </summary>
    public string LabeledJsonlFileName { get; set; }

    [JsonIgnore]
    public string LabeledJsonlPath =>
      Path.Combine(OutDir, string.IsNullOrWhiteSpace(LabeledJsonlFileName) ? "labeled.jsonl" : LabeledJsonlFileName);

    [JsonIgnore]
    public string RejectedJsonlPath => Path.Combine(OutDir, "rejected.jsonl");

    [JsonIgnore]
    public string TpgOutDir => Path.Combine(OutDir, "tpg");

    public static PuzzleReplayOptions Load(string jsonPath)
    {
      string json = File.ReadAllText(jsonPath);
      return JsonSerializer.Deserialize<PuzzleReplayOptions>(json,
        new JsonSerializerOptions { PropertyNameCaseInsensitive = true });
    }

    public void Validate()
    {
      if (string.IsNullOrWhiteSpace(LichessCsvPath) || !File.Exists(LichessCsvPath))
        throw new ArgumentException("LichessCsvPath missing or not found: " + LichessCsvPath);
      if (string.IsNullOrWhiteSpace(NetSpec))
        throw new ArgumentException("NetSpec is required");
      if (string.IsNullOrWhiteSpace(OutDir))
        throw new ArgumentException("OutDir is required");
      if (MinRating < 0 || MaxRating < MinRating)
        throw new ArgumentException($"Invalid rating range [{MinRating},{MaxRating}]");
      if (TeacherNodes < 1) throw new ArgumentException("TeacherNodes must be >= 1");
      Directory.CreateDirectory(OutDir);
    }
  }
}
