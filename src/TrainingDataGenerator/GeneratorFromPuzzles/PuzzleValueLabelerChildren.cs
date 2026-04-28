#region License notice

/*
  This file is part of the CeresTrain project at https://github.com/dje-dev/the-dream.
  Copyright (C) 2023- by David Elliott and the CeresTrain Authors.

  Ceres is free software under the terms of the GNU General Public License v3.0.
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
  /// For each Standard solver-to-move record, emits one additional opp-to-move record
  /// that represents the position AFTER the solver plays the Lichess solution move.
  /// WDL target is a fresh teacher (e.g. C1-640-34) forward pass on the child — not
  /// a reversed parent WDL — so the value head trains on calibrated per-child targets
  /// for the exact kind of positions `go value` queries at inference.
  ///
  /// Output: labeled_teacher_plus.jsonl = input records (passed through unchanged)
  /// + new OppAfterCorrectSolver records interleaved per parent.
  ///
  /// The new records carry Kind = OppDefence semantically (opp-to-move in puzzle line),
  /// but SolutionUci/TeacherPolicy are null — this is a value-only record, because
  /// we don't want to train the policy head on opp-to-move positions (that experiment
  /// hurt policy by 7 pts in v2 enrichment).
  /// </summary>
  public static class PuzzleValueLabelerChildren
  {
    public sealed class Stats
    {
      public long InputRecords;
      public long StandardPassthrough;
      public long ChildrenEmitted;
      public long SkippedParseOrHistory;
      public double ElapsedSec;
    }


    public static Stats Run(PuzzleReplayOptions opts, string inputJsonlPath, string outputJsonlPath)
    {
      if (!File.Exists(inputJsonlPath))
        throw new FileNotFoundException("input labeled jsonl not found", inputJsonlPath);

      Console.WriteLine($"[child-label] Loading teacher: {opts.NetSpec} on {opts.Device}");
      NNEvaluator evaluator = NNEvaluator.FromSpecification(opts.NetSpec, opts.Device);
      int batchSize = Math.Max(64, opts.MineBatchSize);
      Console.WriteLine($"[child-label] Batch size: {batchSize}");

      Stats s = new Stats();
      Stopwatch sw = Stopwatch.StartNew();

      using StreamWriter writer = new StreamWriter(outputJsonlPath, append: false);

      foreach (IReadOnlyList<LabeledPuzzleRecord> batch in Batched(
                 JsonlIO.Read<LabeledPuzzleRecord>(inputJsonlPath), batchSize))
      {
        s.InputRecords += batch.Count;

        // Build child positions + context, track which input records have a valid child.
        List<PositionWithHistory> childPwhBatch = new();
        List<(int batchIdx, Position childPos, string priorForChild)> context = new();

        for (int i = 0; i < batch.Count; i++)
        {
          LabeledPuzzleRecord rec = batch[i];
          if (rec.Kind != PuzzlePositionKind.Standard) continue;
          if (string.IsNullOrWhiteSpace(rec.SolutionUci)) continue;

          // Parse current position and the solver move
          Position pos;
          try { pos = Position.FromFEN(rec.FEN); }
          catch { s.SkippedParseOrHistory++; continue; }
          MGPosition mg = pos.ToMGPosition;
          MGMove solverMG;
          try { solverMG = MGMoveFromString.ParseMove(in mg, rec.SolutionUci); }
          catch { s.SkippedParseOrHistory++; continue; }
          if (solverMG == default) { s.SkippedParseOrHistory++; continue; }

          // Apply solver move → opp-to-move child
          MGPosition mgChild = mg;
          mgChild.MakeMove(solverMG);
          Position childPos = mgChild.ToPosition;

          // Skip terminal positions (checkmate / stalemate) — NNEvaluator's policy
          // extraction throws on positions with zero legal moves. Mate-in-1 puzzles
          // produce checkmate children after the solution move; we just skip those.
          MGMoveList childLegal = new MGMoveList();
          MGMoveGen.GenerateMoves(in mgChild, childLegal);
          if (childLegal.NumMovesUsed == 0) { s.SkippedParseOrHistory++; continue; }

          // Build real history for child: priorForChild = parent's prior + solver move
          string priorForChild = string.IsNullOrWhiteSpace(rec.PriorUciMoves)
            ? rec.SolutionUci
            : rec.PriorUciMoves + " " + rec.SolutionUci;

          // Construct PositionWithHistory for teacher evaluation (with real history)
          PositionWithHistory childPwh = BuildPwh(rec.StartFen, priorForChild, childPos);
          if (childPwh == null) { s.SkippedParseOrHistory++; continue; }
          childPwhBatch.Add(childPwh);
          context.Add((i, childPos, priorForChild));
        }

        NNEvaluatorResult[] childResults = childPwhBatch.Count > 0
          ? evaluator.Evaluate(childPwhBatch, fillInMissingPlanes: true)
          : Array.Empty<NNEvaluatorResult>();

        int evalIdx = 0;
        for (int i = 0; i < batch.Count; i++)
        {
          LabeledPuzzleRecord rec = batch[i];
          // Always pass through the input record first
          JsonlIO.AppendLine(writer, rec);
          s.StandardPassthrough++;

          // Check if this index has a child to emit
          if (evalIdx < context.Count && context[evalIdx].batchIdx == i)
          {
            var ctx = context[evalIdx];
            NNEvaluatorResult r = childResults[evalIdx];
            float w = Clamp01(r.W);
            float l = Clamp01(r.L);
            float d = Math.Max(0f, 1f - w - l);

            LabeledPuzzleRecord childRec = new LabeledPuzzleRecord
            {
              PuzzleId = rec.PuzzleId,
              FEN = ctx.childPos.FEN,
              SolutionUci = null,  // no policy target on child — value-only
              Rating = rec.Rating,
              Themes = rec.Themes,
              StartFen = rec.StartFen,
              PriorUciMoves = ctx.priorForChild,
              Kind = PuzzlePositionKind.OppDefence,
              TeacherNodes = 1,
              TeacherTopUci = null,
              TeacherV = w - l,
              TeacherW = w,
              TeacherD = d,
              TeacherL = l,
              TeacherPolicy = null,
            };
            JsonlIO.AppendLine(writer, childRec);
            s.ChildrenEmitted++;
            evalIdx++;
          }
        }

        if (s.InputRecords % 100_000 == 0)
        {
          writer.Flush();
          double pps = s.InputRecords / Math.Max(1, sw.Elapsed.TotalSeconds);
          double remaining = Math.Max(0, 11_720_827 - s.InputRecords);
          Console.WriteLine($"[child-label] {s.InputRecords:N0} in, {s.ChildrenEmitted:N0} children out, {pps:N0} pos/s, ETA {remaining/Math.Max(1,pps)/60:F1} min");
        }
      }

      writer.Flush();
      sw.Stop();
      s.ElapsedSec = sw.Elapsed.TotalSeconds;
      Console.WriteLine();
      Console.WriteLine($"[child-label] Done.  Input={s.InputRecords:N0}  Passed={s.StandardPassthrough:N0}  Children={s.ChildrenEmitted:N0}");
      Console.WriteLine($"  SkippedParseOrHistory : {s.SkippedParseOrHistory:N0}");
      Console.WriteLine($"  Elapsed                : {s.ElapsedSec:F1}s  ({s.ElapsedSec/60:F1} min)");
      return s;
    }


    static PositionWithHistory BuildPwh(string startFen, string priorMoves, Position currentPos)
    {
      if (string.IsNullOrWhiteSpace(startFen) || string.IsNullOrWhiteSpace(priorMoves))
        return new PositionWithHistory(currentPos);

      Position startPos;
      try { startPos = Position.FromFEN(startFen); }
      catch { return null; }

      string[] moves = priorMoves.Split(' ', StringSplitOptions.RemoveEmptyEntries);
      MGPosition mg = startPos.ToMGPosition;
      List<MGMove> mgMoves = new(moves.Length);
      foreach (string uci in moves)
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
