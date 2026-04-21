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
  /// Evaluates a trained net on labeled.jsonl — i.e. on the exact positions used for
  /// training. If top-1 policy matches the stored SolutionUci at high rate, the
  /// training→inference path is intact and any remaining gap to in-distribution eval
  /// is pure generalization. If still low, something else in the pipeline is broken.
  /// </summary>
  public static class PuzzleEvalOnLabeled
  {
    public static void Run(PuzzleReplayOptions opts)
    {
      opts.Validate();
      if (!File.Exists(opts.LabeledJsonlPath))
        throw new FileNotFoundException("labeled.jsonl not found", opts.LabeledJsonlPath);

      NNEvaluator evaluator = NNEvaluator.FromSpecification(opts.NetSpec, opts.Device);

      long total = 0, topMatchSolution = 0, topMatchTeacherTop = 0;
      Stopwatch sw = Stopwatch.StartNew();

      IEnumerable<LabeledPuzzleRecord> source = JsonlIO.Read<LabeledPuzzleRecord>(opts.LabeledJsonlPath);
      if (opts.MinRating > 0 || opts.MaxRating < int.MaxValue)
        source = System.Linq.Enumerable.Where(source, r => r.Rating >= opts.MinRating && r.Rating <= opts.MaxRating);
      if (opts.EvalStartingPositionsOnly)
      {
        // Setup move only → exactly one token in PriorUciMoves.
        source = System.Linq.Enumerable.Where(source, r =>
          !string.IsNullOrWhiteSpace(r.PriorUciMoves) &&
          r.PriorUciMoves.Split(' ', StringSplitOptions.RemoveEmptyEntries).Length == 1);
      }
      if (opts.MaxEvalRecords > 0) source = System.Linq.Enumerable.Take(source, opts.MaxEvalRecords);
      foreach (IReadOnlyList<LabeledPuzzleRecord> batch in Batched(source, opts.MineBatchSize))
      {
        List<PositionWithHistory> pwhBatch = new List<PositionWithHistory>(batch.Count);
        List<Position> posList = new List<Position>(batch.Count);
        List<MGPosition> mgList = new List<MGPosition>(batch.Count);
        List<MGMove> solutionMoves = new List<MGMove>(batch.Count);
        List<MGMove> teacherTopMoves = new List<MGMove>(batch.Count);
        List<bool> includeRow = new List<bool>(batch.Count);

        foreach (LabeledPuzzleRecord rec in batch)
        {
          bool ok = true;
          Position pos;
          try { pos = Position.FromFEN(rec.FEN); }
          catch { ok = false; pos = default; }

          MGPosition mg = ok ? pos.ToMGPosition : default;
          MGMove solMove = default, teacherMove = default;
          if (ok)
          {
            try { solMove = MGMoveFromString.ParseMove(in mg, rec.SolutionUci); }
            catch { ok = false; }
          }
          if (ok && !string.IsNullOrEmpty(rec.TeacherTopUci))
          {
            try { teacherMove = MGMoveFromString.ParseMove(in mg, rec.TeacherTopUci); }
            catch { /* teacher top not parseable — OK, just skip teacher-match count */ }
          }

          // Build a real-history PositionWithHistory by replaying (StartFen + PriorUciMoves).
          // This matches what EB supplies via UCI `position fen X moves Y...` and what the
          // trained net expects. Falls back to single-position (fake fill history) only for
          // legacy JSONL records missing the StartFen/PriorUciMoves fields.
          PositionWithHistory pwh = null;
          if (ok)
          {
            pwh = BuildRealHistoryPwh(rec, in pos);
            if (pwh == null) ok = false;
          }

          if (ok)
          {
            pwhBatch.Add(pwh);
            posList.Add(pos);
            mgList.Add(mg);
            solutionMoves.Add(solMove);
            teacherTopMoves.Add(teacherMove);
            includeRow.Add(true);
          }
          else
          {
            includeRow.Add(false);
          }
        }

        if (pwhBatch.Count == 0) continue;

        NNEvaluatorResult[] results = evaluator.Evaluate(pwhBatch, fillInMissingPlanes: true);

        for (int i = 0; i < pwhBatch.Count; i++)
        {
          Position p = posList[i];
          MGMove netTop = results[i].Policy.TopMove(in p);
          total++;
          if (netTop != default && netTop == solutionMoves[i]) topMatchSolution++;
          if (teacherTopMoves[i] != default && netTop != default && netTop == teacherTopMoves[i]) topMatchTeacherTop++;
        }

        if (total % 5000 < opts.MineBatchSize)
        {
          double dur = Math.Max(1, sw.Elapsed.TotalSeconds);
          Console.WriteLine($"[eval-labeled] {total:N0} processed  " +
                            $"solution-match={(100.0 * topMatchSolution / total):F1}%  " +
                            $"teacher-top-match={(100.0 * topMatchTeacherTop / total):F1}%  " +
                            $"{total / dur:N0} pos/s");
        }
      }

      sw.Stop();
      Console.WriteLine();
      Console.WriteLine($"[eval-labeled] Done.  Total={total:N0}  Elapsed={sw.Elapsed.TotalSeconds:F1}s");
      Console.WriteLine($"  Net top-1 == Lichess SolutionUci :  {topMatchSolution:N0}  ({(total == 0 ? 0 : 100.0 * topMatchSolution / total):F2}%)");
      Console.WriteLine($"  Net top-1 == TeacherTopUci       :  {topMatchTeacherTop:N0}  ({(total == 0 ? 0 : 100.0 * topMatchTeacherTop / total):F2}%)");
    }


    /// <summary>
    /// Builds a PositionWithHistory by replaying rec.StartFen + rec.PriorUciMoves,
    /// mirroring what Ceres constructs at inference from EB's UCI `position fen ... moves ...`.
    /// Returns null on any parse/replay failure. If StartFen/PriorUciMoves are absent
    /// (legacy JSONL without the new fields), falls back to the single-position constructor.
    /// </summary>
    static PositionWithHistory BuildRealHistoryPwh(LabeledPuzzleRecord rec, in Position currentPos)
    {
      if (string.IsNullOrWhiteSpace(rec.StartFen) || string.IsNullOrWhiteSpace(rec.PriorUciMoves))
      {
        return new PositionWithHistory(currentPos);
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
