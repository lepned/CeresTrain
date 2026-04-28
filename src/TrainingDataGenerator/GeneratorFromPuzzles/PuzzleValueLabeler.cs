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

using Ceres.Chess;
using Ceres.Chess.MoveGen;
using Ceres.Chess.MoveGen.Converters;
using Ceres.Chess.NetEvaluation.Batch;
using Ceres.Chess.NNEvaluators;
using Ceres.Chess.Positions;

namespace CeresTrain.TrainingDataGenerator.GeneratorFromPuzzles
{
  /// <summary>
  /// Stage 3 teacher value labeler: reads labeled.jsonl, runs each position
  /// through a strong teacher ONNX (e.g. C1-640-34-I8) via NNEvaluator, and
  /// writes a new labeled_teacher.jsonl with TeacherW/D/L overwritten by
  /// calibrated per-position teacher outputs.
  ///
  /// Policy targets (SolutionUci + TeacherPolicy) remain Lichess-derived:
  /// Lichess's unique-winning guarantee is authoritative for policy; the
  /// teacher may be weaker than Lichess's Stockfish-verified solutions.
  /// Only the value targets benefit from teacher replacement — per-position
  /// W/D/L gives the value head the numeric discrimination it can't learn
  /// from theme-bucketed targets.
  ///
  /// Real history is reconstructed from StartFen + PriorUciMoves and fed to
  /// the evaluator as a full move sequence — same flow as the eval-labeled
  /// path, ensuring teacher's history planes match the training target's
  /// history planes.
  ///
  /// Records with Kind != Standard are passed through unchanged (they either
  /// don't exist in a baseline labeled.jsonl or were already specialized by
  /// prior enrichment and shouldn't be re-labeled).
  /// </summary>
  public static class PuzzleValueLabeler
  {
    public sealed class Stats
    {
      public long InputRecords;
      public long Labeled;
      public long SkippedBadFen;
      public long SkippedBadHistory;
      public double ElapsedSec;
    }


    public static Stats Run(PuzzleReplayOptions opts, string outputJsonlPath)
    {
      opts.Validate();
      if (!File.Exists(opts.LabeledJsonlPath))
        throw new FileNotFoundException("labeled.jsonl not found", opts.LabeledJsonlPath);

      Console.WriteLine($"[value-label] Loading teacher: {opts.NetSpec} on {opts.Device}");
      NNEvaluator evaluator = NNEvaluator.FromSpecification(opts.NetSpec, opts.Device);
      int batchSize = Math.Max(64, opts.MineBatchSize);
      Console.WriteLine($"[value-label] Batch size: {batchSize}");

      Stats s = new Stats();
      Stopwatch sw = Stopwatch.StartNew();

      using StreamWriter writer = new StreamWriter(outputJsonlPath, append: false);

      foreach (IReadOnlyList<LabeledPuzzleRecord> batch in Batched(
                 JsonlIO.Read<LabeledPuzzleRecord>(opts.LabeledJsonlPath), batchSize))
      {
        s.InputRecords += batch.Count;

        // Build PositionWithHistory for the batch with REAL history (replayed from StartFen + PriorUciMoves).
        List<PositionWithHistory> pwhBatch = new List<PositionWithHistory>(batch.Count);
        List<int> batchIndexToRecordIndex = new List<int>(batch.Count);

        for (int i = 0; i < batch.Count; i++)
        {
          LabeledPuzzleRecord rec = batch[i];
          // Only (re)label Standard records. Others (OppDefence etc.) should not
          // be present in a bare labeled.jsonl but we pass them through as-is.
          if (rec.Kind != PuzzlePositionKind.Standard)
          {
            JsonlIO.AppendLine(writer, rec);
            continue;
          }

          PositionWithHistory pwh = BuildRealHistoryPwh(rec);
          if (pwh == null)
          {
            s.SkippedBadHistory++;
            // Still write the record unchanged so downstream TPG gen doesn't see gaps.
            JsonlIO.AppendLine(writer, rec);
            continue;
          }
          pwhBatch.Add(pwh);
          batchIndexToRecordIndex.Add(i);
        }

        if (pwhBatch.Count == 0)
          continue;

        NNEvaluatorResult[] results = evaluator.Evaluate(pwhBatch, fillInMissingPlanes: true);

        // Write teacher-labeled records in the original batch order so output file
        // preserves labeled.jsonl ordering (important for downstream reproducibility).
        int evalIdx = 0;
        for (int i = 0; i < batch.Count; i++)
        {
          LabeledPuzzleRecord rec = batch[i];
          if (rec.Kind != PuzzlePositionKind.Standard) continue;

          // batchIndexToRecordIndex[evalIdx] tells us the batch-position of the next evaluated result
          if (evalIdx < batchIndexToRecordIndex.Count && batchIndexToRecordIndex[evalIdx] == i)
          {
            NNEvaluatorResult r = results[evalIdx];
            float w = Clamp01(r.W);
            float l = Clamp01(r.L);
            float d = Math.Max(0f, 1f - w - l);

            LabeledPuzzleRecord t = new LabeledPuzzleRecord
            {
              PuzzleId = rec.PuzzleId,
              FEN = rec.FEN,
              SolutionUci = rec.SolutionUci,
              Rating = rec.Rating,
              Themes = rec.Themes,
              StartFen = rec.StartFen,
              PriorUciMoves = rec.PriorUciMoves,
              Kind = PuzzlePositionKind.Standard,
              TeacherNodes = 1,  // nodes=1 teacher
              TeacherTopUci = rec.TeacherTopUci,
              TeacherV = w - l,
              TeacherW = w,
              TeacherD = d,
              TeacherL = l,
              TeacherPolicy = rec.TeacherPolicy,  // keep Lichess one-hot policy
            };
            JsonlIO.AppendLine(writer, t);
            s.Labeled++;
            evalIdx++;
          }
        }

        if (s.InputRecords % 100_000 == 0)
        {
          writer.Flush();
          double pps = s.InputRecords / Math.Max(1, sw.Elapsed.TotalSeconds);
          double eta = (s.InputRecords > 0) ? ((double)(11_720_827 - s.InputRecords) / Math.Max(1, pps)) : 0;
          Console.WriteLine($"[value-label] {s.InputRecords:N0} in, {s.Labeled:N0} labeled, {pps:N0} pos/s, ETA {eta/60:F1} min");
        }
      }

      writer.Flush();
      sw.Stop();
      s.ElapsedSec = sw.Elapsed.TotalSeconds;
      Console.WriteLine();
      Console.WriteLine($"[value-label] Done.  Input={s.InputRecords:N0}  Labeled={s.Labeled:N0}");
      Console.WriteLine($"  SkippedBadFen      : {s.SkippedBadFen:N0}");
      Console.WriteLine($"  SkippedBadHistory  : {s.SkippedBadHistory:N0}");
      Console.WriteLine($"  Elapsed            : {s.ElapsedSec:F1}s  ({s.ElapsedSec/60:F1} min)");
      return s;
    }


    /// <summary>
    /// Builds a PositionWithHistory for a labeled record by replaying
    /// (StartFen + PriorUciMoves). Returns null on any parse failure.
    /// Copied from PuzzleEvalOnLabeled's BuildRealHistoryPwh — keeps the history
    /// semantics identical between eval and teacher-labeling paths.
    /// </summary>
    static PositionWithHistory BuildRealHistoryPwh(LabeledPuzzleRecord rec)
    {
      Position pos;
      try { pos = Position.FromFEN(rec.FEN); }
      catch { return null; }

      if (string.IsNullOrWhiteSpace(rec.StartFen) || string.IsNullOrWhiteSpace(rec.PriorUciMoves))
      {
        return new PositionWithHistory(pos);
      }

      Position startPos;
      try { startPos = Position.FromFEN(rec.StartFen); }
      catch { return null; }

      string[] priorUci = rec.PriorUciMoves.Split(' ', StringSplitOptions.RemoveEmptyEntries);
      MGPosition mg = startPos.ToMGPosition;
      List<MGMove> mgMoves = new List<MGMove>(priorUci.Length);
      foreach (string uci in priorUci)
      {
        MGMove mv;
        try { mv = MGMoveFromString.ParseMove(in mg, uci); }
        catch { return null; }
        if (mv == default) return null;
        mgMoves.Add(mv);
        mg.MakeMove(mv);
      }
      return new PositionWithHistory(startPos, mgMoves);
    }


    static float Clamp01(float x)
    {
      if (float.IsNaN(x)) return 0.5f;
      return x < 0 ? 0 : (x > 1 ? 1 : x);
    }


    static IEnumerable<IReadOnlyList<T>> Batched<T>(IEnumerable<T> source, int size)
    {
      List<T> buf = new List<T>(size);
      foreach (T item in source)
      {
        buf.Add(item);
        if (buf.Count >= size) { yield return buf; buf = new List<T>(size); }
      }
      if (buf.Count > 0) yield return buf;
    }
  }
}
