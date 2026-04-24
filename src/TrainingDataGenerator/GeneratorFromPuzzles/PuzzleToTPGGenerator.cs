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
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;

using Ceres.Base.Math;
using Ceres.Chess;
using Ceres.Chess.EncodedPositions;
using Ceres.Chess.EncodedPositions.Basic;
using Ceres.Chess.MoveGen;
using Ceres.Chess.MoveGen.Converters;
using Ceres.Chess.NNEvaluators.Ceres.TPG;
using Ceres.Chess.Positions;

using CeresTrain.TPG.TPGGenerator;

namespace CeresTrain.TrainingDataGenerator.GeneratorFromPuzzles
{
  /// <summary>
  /// Converts labeled.jsonl into TPG shards using the existing TrainingPositionWriter.
  /// Mirrors the pattern in TrainingRecordFromTablebase, substituting teacher-search
  /// derived targets for tablebase-derived targets.
  /// </summary>
  public static class PuzzleToTPGGenerator
  {
    const int LC0_DATA_VERSION = 6;
    const int LC0_INPUT_FORMAT = 1;
    const byte LC0_INVARIANCE_INFO = 32;


    public static EmitStats Run(PuzzleReplayOptions opts)
    {
      opts.Validate();
      if (!File.Exists(opts.LabeledJsonlPath))
        throw new FileNotFoundException("labeled.jsonl not found — run label stage first", opts.LabeledJsonlPath);

      Directory.CreateDirectory(opts.TpgOutDir);
      string outFile = Path.Combine(opts.TpgOutDir, "puzzles_" + DateTime.UtcNow.ToString("yyyyMMdd_HHmmss") + ".tpg");

      // Resolve rating stratification weights (1 = no oversampling; null/empty array = all 1x).
      int[] thresholds = opts.RatingBinThresholds ?? Array.Empty<int>();
      int[] weights = opts.RatingBinWeights ?? Array.Empty<int>();
      bool stratify = weights.Length > 0 && weights.Length == thresholds.Length + 1;
      if (weights.Length > 0 && weights.Length != thresholds.Length + 1)
      {
        throw new ArgumentException(
          $"RatingBinWeights ({weights.Length}) must have exactly one more entry than RatingBinThresholds ({thresholds.Length}).");
      }
      if (stratify)
      {
        Console.Write($"[to-tpg] Stratifying by rating. Bins: (-inf");
        for (int i = 0; i < thresholds.Length; i++) Console.Write($",{thresholds[i]}");
        Console.Write($",+inf)  Weights: [{string.Join(",", weights)}]");
        Console.WriteLine();
      }

      // First pass: count how many records will actually produce a valid TPG pair,
      // accounting for per-record stratification weights.
      // TrainingPositionWriter requires totalNumPositionsToBeWritten to be an exact
      // multiple of batch size, so we need the exact count up front.
      long exactCount = 0;
      long distinctValid = 0;
      foreach (LabeledPuzzleRecord rec in JsonlIO.Read<LabeledPuzzleRecord>(opts.LabeledJsonlPath))
      {
        if (TryBuild(rec, out _, out _))
        {
          distinctValid++;
          exactCount += stratify ? RatingWeight(rec.Rating, thresholds, weights) : 1;
        }
      }
      Console.WriteLine($"[to-tpg] Counted {distinctValid:N0} distinct valid records -> {exactCount:N0} TPG rows after stratification.");

      if (exactCount == 0)
      {
        Console.WriteLine("[to-tpg] No valid records to write.");
        return new EmitStats { Emitted = 0, Skipped = 0, ElapsedSec = 0 };
      }

      // Pick a batch size that divides exactCount cleanly. Prefer larger batches
      // (closer to 2048) for better write throughput, but fall back to 1 if needed.
      int batchSize = ChooseBatchSize(exactCount, preferredMax: 2048);
      const int NUM_WORKER_THREADS = 1;
      // EMIT_HISTORY must be true: when false, TPGRecordConverter.ConvertToTPGRecordSquares
      // overwrites all 7 history positions with the current position right before writing
      // the per-square bytes, which strips the real history we reconstructed via
      // SetFromSequentialPositions. Inference via EB supplies real history, so the trained
      // net must see real history during training too.
      const bool EMIT_HISTORY = true;

      TrainingPositionWriter writer = new TrainingPositionWriter(
        outFile, NUM_WORKER_THREADS, TPGGeneratorOptions.OutputRecordFormat.TPGRecord,
        true, System.IO.Compression.CompressionLevel.Optimal,
        exactCount, null, null, null, batchSize, true, EMIT_HISTORY, true);

      long emitted = 0, skipped = 0;
      Stopwatch sw = Stopwatch.StartNew();

      // With emitPlySinceLastMovePerSquare=true (required for Python TPG dataloader),
      // the writer dereferences indexLastMoveBySquares; cannot be null. Puzzles have no
      // move history, so pass a 64-element zeros array (64 squares).
      short[] zeroLastMoveBySquares = new short[64];

      foreach (LabeledPuzzleRecord rec in JsonlIO.Read<LabeledPuzzleRecord>(opts.LabeledJsonlPath))
      {
        if (TryBuild(rec, out EncodedTrainingPosition etp, out TPGTrainingTargetNonPolicyInfo target))
        {
          int times = stratify ? RatingWeight(rec.Rating, thresholds, weights) : 1;
          for (int k = 0; k < times; k++)
          {
            writer.Write(in etp, target, 0, zeroLastMoveBySquares,
                         CompressedPolicyVector.DEFAULT_MIN_PROBABILITY_LEGAL_MOVE, 0);
            emitted++;
          }
        }
        else
        {
          skipped++;
        }

        if ((emitted + skipped) % 5000 == 0)
          Console.WriteLine($"[to-tpg] emitted={emitted:N0} skipped={skipped:N0}");
      }

      writer.Shutdown();
      sw.Stop();

      EmitStats stats = new EmitStats { Emitted = emitted, Skipped = skipped, ElapsedSec = sw.Elapsed.TotalSeconds };
      Console.WriteLine($"[to-tpg] Done. Emitted={stats.Emitted:N0} Skipped={stats.Skipped:N0} " +
                        $"-> {outFile}");
      return stats;
    }


    /// <summary>
    /// Builds the TPG pair (EncodedTrainingPosition, TPGTrainingTargetNonPolicyInfo) from a labeled record.
    /// Does NOT flip Black-to-move positions — matches what the production TPG generator
    /// (TrainingPositionGenerator.PreparePosition) does, and what Ceres at inference expects.
    /// The inference path encodes from the side-to-move's perspective via SetFromPosition's
    /// `desiredFromSidePerspective` argument; we do the same here.
    /// </summary>
    public static bool TryBuild(LabeledPuzzleRecord rec,
                                out EncodedTrainingPosition etp,
                                out TPGTrainingTargetNonPolicyInfo targetInfo)
    {
      etp = default;
      targetInfo = default;

      Position pos;
      try { pos = Position.FromFEN(rec.FEN); }
      catch { return false; }

      MGPosition mgPos = pos.ToMGPosition;

      // A record is "value-only" if it has no policy target (no SolutionUci,
      // no TeacherPolicy). In that case we emit it with an all-zero policy
      // target so training-time policy loss is zero for that record, while the
      // value head still learns from TeacherW/D/L. Standard and OppDefence
      // records both carry policy targets (OppDefence's policy = opp's puzzle
      // defence move) and go through the full policy+value path.
      bool hasPolicy = rec.TeacherPolicy != null && rec.TeacherPolicy.Count > 0
                       && !string.IsNullOrEmpty(rec.SolutionUci);

      // A Standard record is required to have a policy target; if it doesn't,
      // reject it (data corruption / legacy format).
      if (rec.Kind == PuzzlePositionKind.Standard && !hasPolicy) return false;

      if (!hasPolicy)
      {
        return TryBuildValueOnly(rec, in pos, in mgPos, out etp, out targetInfo);
      }

      // Build dict of teacher-visited move-index → visit-fraction
      Dictionary<int, float> visitedIdxToProb = new Dictionary<int, float>(rec.TeacherPolicy.Count);
      foreach (PolicyEntry e in rec.TeacherPolicy)
      {
        if (e.P <= 0f) continue;
        MGMove mgMove;
        try { mgMove = MGMoveFromString.ParseMove(in mgPos, e.Uci); }
        catch { continue; }
        if (mgMove == default) continue;

        // MGChessMoveToEncodedMove internally handles the Black-to-move flip
        // (line 229-232 of ConverterMGMoveEncodedMove.cs) — we don't need to.
        int idx = ConverterMGMoveEncodedMove.MGChessMoveToEncodedMove(mgMove).IndexNeuralNet;
        if (!visitedIdxToProb.ContainsKey(idx)) visitedIdxToProb[idx] = 0f;
        visitedIdxToProb[idx] += e.P;
      }

      if (visitedIdxToProb.Count == 0) return false;

      // Enumerate ALL legal moves in the position and include each in the policy target
      // at minimum DEFAULT_MIN_PROBABILITY_LEGAL_MOVE (0.05%). Teacher-visited moves keep
      // their higher teacher probability. This is the key fix: without ALL legal moves in
      // the target, the training-time loss/accuracy mask (`target.greater(0)`) only treats
      // teacher-visited moves as "legal", leaving gradient signal undefined for other legal
      // moves — so at inference (which masks over ALL legal moves) those untrained logits
      // can randomly dominate. Production TAR training data has all legal moves by construction;
      // puzzle teacher-policy doesn't, so we must add them explicitly here.
      MGMoveList allLegalMoves = new MGMoveList();
      MGMoveGen.GenerateMoves(in mgPos, allLegalMoves);

      List<int> indices = new List<int>();
      List<float> probs = new List<float>();
      HashSet<int> addedIdx = new HashSet<int>();

      foreach (MGMove legalMove in allLegalMoves.MovesArray.AsSpan(0, allLegalMoves.NumMovesUsed))
      {
        int idx = ConverterMGMoveEncodedMove.MGChessMoveToEncodedMove(legalMove).IndexNeuralNet;
        if (!addedIdx.Add(idx)) continue;
        float p = visitedIdxToProb.TryGetValue(idx, out float teacherP)
                    ? teacherP
                    : CompressedPolicyVector.DEFAULT_MIN_PROBABILITY_LEGAL_MOVE;
        indices.Add(idx);
        probs.Add(p);
      }

      if (indices.Count == 0) return false;

      float[] probsArr = probs.ToArray();
      StatUtils.Normalize(probsArr);

      CompressedPolicyVector cpv = default;
      CompressedPolicyVector.Initialize(ref cpv, pos.SideToMove, indices.ToArray(), probsArr, false);

      float w = Clamp01(rec.TeacherW);
      float l = Clamp01(rec.TeacherL);
      float d = Math.Max(0f, 1f - w - l);
      float q = w - l;

      EncodedPositionWithHistory newPosHistory = default;
      // Rebuild real history from (StartFen + PriorUciMoves) so history/repetition planes
      // match what Ceres reconstructs at inference from EB's UCI `position ... moves ...`.
      // Without this, FILL_HISTORY=true would repeat the current position into the 7
      // history planes, producing a train/inference skew (and spurious repetitions).
      Span<Position> history = BuildHistorySpan(rec, in pos);
      const bool FILL_MISSING = true;
      newPosHistory.SetFromSequentialPositions(history, FILL_MISSING);

      EncodedPolicyVector epv = default;
      epv.InitilializeAllNegativeOne();
      unsafe
      {
        float* encodedProbs = epv.ProbabilitiesPtr;
        foreach ((EncodedMove move, float probability) in cpv.ProbabilitySummary(0))
          encodedProbs[move.IndexNeuralNet] = probability;
      }

      int bestMoveIndex = epv.BestMove.IndexNeuralNet;
      float m = 0;

      EncodedPositionEvalMiscInfoV6 trainingMiscInfo = new(
        invarianceInfo: LC0_INVARIANCE_INFO, depResult: default,
        rootQ: q, bestQ: q, rootD: d, bestD: d,
        rootM: m, bestM: m, pliesLeft: m,
        resultQ: q, resultD: d,
        playedQ: q, playedD: d, playedM: m,
        originalQ: q, originalD: d, originalM: m,
        numVisits: rec.TeacherNodes,
        playedIndex: (short)bestMoveIndex, bestIndex: (short)bestMoveIndex,
        kldPolicy: default, unused2: default);

      EncodedTrainingPositionMiscInfo miscInfoAll = new(newPosHistory.MiscInfo.InfoPosition, trainingMiscInfo);
      newPosHistory.SetMiscInfo(miscInfoAll);

      etp = new EncodedTrainingPosition(LC0_DATA_VERSION, LC0_INPUT_FORMAT, newPosHistory, epv);

      targetInfo.BestWDL = (w, d, l);
      targetInfo.ResultDeblunderedWDL = (w, d, l);
      targetInfo.ResultNonDeblunderedWDL = (w, d, l);
      targetInfo.Source = TPGTrainingTargetNonPolicyInfo.TargetSourceInfo.Training;
      targetInfo.NumSearchNodes = rec.TeacherNodes;
      targetInfo.MLH = 0;

      // Match Ceres inference defaults: NNEvaluatorOptionsCeres.DEFAULT_Q_BLUNDER = 0.03.
      // These values are written into every TPGSquareRecord's QNegativeBlunders /
      // QPositiveBlunders bytes (via TrainingPositionWriter -> ConvertPositionsToRawSquareBytes).
      // Leaving them at 0 created a train/inference skew: model trained seeing all zeros
      // for these byte positions while Ceres at inference populates 0.03, giving the
      // model inputs it never saw during training.
      const float DEFAULT_Q_BLUNDER = 0.03f;
      targetInfo.ForwardSumPositiveBlunders = DEFAULT_Q_BLUNDER;
      targetInfo.ForwardSumNegativeBlunders = DEFAULT_Q_BLUNDER;

      return true;
    }


    /// <summary>
    /// Emits a value-only TPG record: WDL target set, policy target all zeros so
    /// training policy loss evaluates to 0 on these records. Used for non-Standard
    /// record kinds (OppDefence, SolverAfterInferiorOpp, PreBlunder).
    /// </summary>
    static bool TryBuildValueOnly(LabeledPuzzleRecord rec, in Position pos, in MGPosition mgPos,
                                  out EncodedTrainingPosition etp,
                                  out TPGTrainingTargetNonPolicyInfo targetInfo)
    {
      etp = default;
      targetInfo = default;

      float w = Clamp01(rec.TeacherW);
      float l = Clamp01(rec.TeacherL);
      float d = Math.Max(0f, 1f - w - l);
      float q = w - l;

      // Build the encoded position from replayed real history, same as Standard path.
      EncodedPositionWithHistory newPosHistory = default;
      Span<Position> history = BuildHistorySpan(rec, in pos);
      const bool FILL_MISSING = true;
      newPosHistory.SetFromSequentialPositions(history, FILL_MISSING);

      // All-zero policy target: target.greater(0) is all False during training,
      // so cross-entropy loss on policy = 0 and the policy head gets no gradient
      // contribution from this record.
      EncodedPolicyVector epv = default;  // default struct is all zeros.

      float m = 0;
      // Since there's no "best move" here, set playedIndex/bestIndex to 0 (any
      // valid index works; loss/accuracy paths don't consume these for value-only
      // records since target has no peak).
      EncodedPositionEvalMiscInfoV6 trainingMiscInfo = new(
        invarianceInfo: LC0_INVARIANCE_INFO, depResult: default,
        rootQ: q, bestQ: q, rootD: d, bestD: d,
        rootM: m, bestM: m, pliesLeft: m,
        resultQ: q, resultD: d,
        playedQ: q, playedD: d, playedM: m,
        originalQ: q, originalD: d, originalM: m,
        numVisits: rec.TeacherNodes,
        playedIndex: 0, bestIndex: 0,
        kldPolicy: default, unused2: default);

      EncodedTrainingPositionMiscInfo miscInfoAll = new(newPosHistory.MiscInfo.InfoPosition, trainingMiscInfo);
      newPosHistory.SetMiscInfo(miscInfoAll);

      etp = new EncodedTrainingPosition(LC0_DATA_VERSION, LC0_INPUT_FORMAT, newPosHistory, epv);

      targetInfo.BestWDL = (w, d, l);
      targetInfo.ResultDeblunderedWDL = (w, d, l);
      targetInfo.ResultNonDeblunderedWDL = (w, d, l);
      // Source = ActionHeadDummyMove is the existing sentinel that tells
      // TrainingPositionWriter to skip policy-validity checks for this record.
      // TPGTrainingTargetNonPolicyInfo.Source is a C# writer-side marker; the
      // Python training side (tpg_dataset.py) does not read it, so value-head
      // learning from TeacherW/D/L proceeds normally while the policy loss on
      // these records is zero (target.greater(0) is all False).
      targetInfo.Source = TPGTrainingTargetNonPolicyInfo.TargetSourceInfo.ActionHeadDummyMove;
      targetInfo.NumSearchNodes = rec.TeacherNodes;
      targetInfo.MLH = 0;
      const float DEFAULT_Q_BLUNDER = 0.03f;
      targetInfo.ForwardSumPositiveBlunders = DEFAULT_Q_BLUNDER;
      targetInfo.ForwardSumNegativeBlunders = DEFAULT_Q_BLUNDER;

      return true;
    }


    /// <summary>
    /// Maps a puzzle rating to its bin weight. Thresholds define bin edges (ascending);
    /// weights has one more entry than thresholds. Rating &lt; thresholds[0] → weights[0];
    /// rating in [thresholds[i-1], thresholds[i]) → weights[i]; rating &gt;= thresholds[last] → weights[last].
    /// </summary>
    static int RatingWeight(int rating, int[] thresholds, int[] weights)
    {
      int bin = thresholds.Length; // default to last bin
      for (int i = 0; i < thresholds.Length; i++)
      {
        if (rating < thresholds[i]) { bin = i; break; }
      }
      return weights[bin];
    }


    /// <summary>
    /// Reconstructs up to NUM_MOVES_HISTORY Position entries ending at <paramref name="currentPos"/>
    /// by replaying (rec.StartFen + rec.PriorUciMoves). If either field is missing (older JSONL
    /// without history), falls back to a single-entry span — equivalent to the legacy behavior
    /// where SetFromSequentialPositions with fillInMissingPlanes=true repeats the position.
    /// </summary>
    static Position[] BuildHistorySpan(LabeledPuzzleRecord rec, in Position currentPos)
    {
      const int MAX_HIST = 8; // EncodedPositionBoards.NUM_MOVES_HISTORY

      if (string.IsNullOrWhiteSpace(rec.StartFen) || string.IsNullOrWhiteSpace(rec.PriorUciMoves))
      {
        return new[] { currentPos };
      }

      Position startPos;
      try { startPos = Position.FromFEN(rec.StartFen); }
      catch { return new[] { currentPos }; }

      string[] priorMoves = rec.PriorUciMoves.Split(' ', StringSplitOptions.RemoveEmptyEntries);

      // Build full sequence: [startPos, afterMove0, afterMove1, ..., afterMoveN-1 == currentPos].
      List<Position> all = new List<Position>(priorMoves.Length + 1);
      MGPosition mg = startPos.ToMGPosition;
      all.Add(mg.ToPosition);
      foreach (string uci in priorMoves)
      {
        MGMove mgMove;
        try { mgMove = MGMoveFromString.ParseMove(in mg, uci); }
        catch { return new[] { currentPos }; }
        if (mgMove == default) return new[] { currentPos };
        mg.MakeMove(mgMove);
        all.Add(mg.ToPosition);
      }

      // Keep only the last MAX_HIST entries, last = current (to-move) position.
      int startIdx = Math.Max(0, all.Count - MAX_HIST);
      int take = all.Count - startIdx;
      Position[] tail = new Position[take];
      for (int i = 0; i < take; i++) tail[i] = all[startIdx + i];
      return tail;
    }


    /// <summary>
    /// Flips a UCI move by vertical rank inversion (rank N → rank 9-N, file unchanged).
    /// Preserves promotion suffix. Matches Position.Reversed semantics.
    /// </summary>
    static string FlipUci(string uci)
    {
      if (string.IsNullOrEmpty(uci) || uci.Length < 4) return uci;
      char f1 = uci[0];
      char r1 = FlipRank(uci[1]);
      char f2 = uci[2];
      char r2 = FlipRank(uci[3]);
      string suffix = uci.Length > 4 ? uci.Substring(4) : "";
      return new string(new[] { f1, r1, f2, r2 }) + suffix;
    }


    static char FlipRank(char r) => r switch
    {
      '1' => '8', '2' => '7', '3' => '6', '4' => '5',
      '5' => '4', '6' => '3', '7' => '2', '8' => '1',
      _ => r,
    };


    static float Clamp01(float x) => x < 0 ? 0 : (x > 1 ? 1 : x);


    public readonly record struct EmitStats(long Emitted, long Skipped, double ElapsedSec);


    /// <summary>
    /// Find the largest divisor of <paramref name="total"/> not exceeding <paramref name="preferredMax"/>.
    /// Falls back to 1 (which divides anything) if nothing larger works, guaranteeing a valid batch size.
    /// </summary>
    static int ChooseBatchSize(long total, int preferredMax)
    {
      for (int bs = (int)Math.Min(total, preferredMax); bs >= 1; bs--)
      {
        if (total % bs == 0) return bs;
      }
      return 1;
    }
  }
}
