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
using System.Linq;

using Ceres.Chess;
using Ceres.Chess.MoveGen;
using Ceres.Chess.MoveGen.Converters;
using Ceres.Chess.NNEvaluators.Defs;
using Ceres.Chess.Positions;

using Ceres.MCGS.GameEngines;
using Ceres.MCGS.Graphs.GNodes;

namespace CeresTrain.TrainingDataGenerator.GeneratorFromPuzzles
{
  /// <summary>
  /// Smoke test for the hypothesis that 100-node MCGS over-reports D on simplified
  /// post-solver-move (OppDef) positions. Re-searches a sample of OppDef records
  /// where the original |TeacherV| was below a threshold (suspect-drawn) at a higher
  /// node budget, and reports how many of them resolve to a sharper Q.
  ///
  /// Inputs (from PuzzleReplayOptions):
  ///   - LabeledJsonlPath: existing labeled_with_oppdef.jsonl
  ///   - DeepenQThreshold: filter records where |old TeacherV| &lt; this.
  ///   - DeepenSampleN:    how many filtered records to process.
  ///   - DeepenNodes:      search-node budget for the deeper search.
  ///   - LichessCsvPath:   for puzzle-line lookup.
  ///   - NetSpec, Device:  same teacher net used in original enrichment.
  /// </summary>
  public static class PuzzleOppDefenceDeepenSmoke
  {
    public static void Run(PuzzleReplayOptions opts)
    {
      opts.Validate();
      if (!File.Exists(opts.LabeledJsonlPath))
        throw new FileNotFoundException("labeled jsonl not found", opts.LabeledJsonlPath);

      Console.WriteLine($"[deepen-smoke] Loading puzzle moves from CSV: {opts.LichessCsvPath}");
      Dictionary<string, (string startFen, string[] moves)> puzzleMoves = LoadPuzzleMoves(opts.LichessCsvPath);
      Console.WriteLine($"[deepen-smoke] Loaded {puzzleMoves.Count:N0} puzzles");

      Console.WriteLine($"[deepen-smoke] Filter: keep OppDef records with |TeacherV| < {opts.DeepenQThreshold:F2}");
      Console.WriteLine($"[deepen-smoke] Sample size: {opts.DeepenSampleN:N0}");
      Console.WriteLine($"[deepen-smoke] Deep search nodes: {opts.DeepenNodes}");

      // Collect filtered sample
      List<LabeledPuzzleRecord> sample = new();
      long oppdefSeen = 0;
      foreach (LabeledPuzzleRecord rec in JsonlIO.Read<LabeledPuzzleRecord>(opts.LabeledJsonlPath))
      {
        if (rec.Kind != PuzzlePositionKind.OppDefence) continue;
        oppdefSeen++;
        if (Math.Abs(rec.TeacherV) >= opts.DeepenQThreshold) continue;
        sample.Add(rec);
        if (sample.Count >= opts.DeepenSampleN) break;
      }
      Console.WriteLine($"[deepen-smoke] Collected {sample.Count:N0} sample records (scanned {oppdefSeen:N0} OppDef before stopping)");
      if (sample.Count == 0) { Console.WriteLine("[deepen-smoke] Empty sample — nothing to do."); return; }

      // Setup engine
      Console.WriteLine($"[deepen-smoke] Loading teacher: {opts.NetSpec} on {opts.Device}");
      string deviceSpec = opts.Device.Contains('=') ? opts.Device : opts.Device + "=DeepenSmoke";
      NNEvaluatorDef evalDef = NNEvaluatorDefFactory.FromSpecification(opts.NetSpec, deviceSpec);
      Ceres.Chess.NNEvaluators.NNEvaluator warmup = Ceres.Chess.NNEvaluators.NNEvaluatorFactory.BuildEvaluator(evalDef);
      Ceres.Chess.NNEvaluators.NNEvaluatorBenchmark.Warmup(warmup, warmup.MaxBatchSize);
      Console.WriteLine("[deepen-smoke] Warmup complete.");

      GameEngineCeresMCGSInProcess engine = new GameEngineCeresMCGSInProcess(
        id: "DeepenSmoke", evaluatorDef: evalDef, moveImmediateIfOnlyOneMove: false);

      // Aggregate stats
      int processed = 0;
      int failed = 0;
      double sumOldQ = 0, sumNewQ = 0, sumOldD = 0, sumNewD = 0;
      double sumDeltaAbsQ = 0;
      int flippedToWin = 0;     // |new Q| >= 0.3 from old |Q| < threshold
      int flippedToWinHard = 0; // |new Q| >= 0.5
      int stayedDrawish = 0;    // |new Q| stayed < 0.2

      Stopwatch sw = Stopwatch.StartNew();

      foreach (LabeledPuzzleRecord rec in sample)
      {
        if (!puzzleMoves.TryGetValue(rec.PuzzleId, out var puz)) { failed++; continue; }

        // Reconstruct the post-solver-move position via StartFen + PriorUciMoves
        // (these were populated by the enricher).
        Position startPos;
        try { startPos = Position.FromFEN(rec.StartFen); } catch { failed++; continue; }

        string[] priorMoves = string.IsNullOrEmpty(rec.PriorUciMoves)
                              ? Array.Empty<string>()
                              : rec.PriorUciMoves.Split(' ', StringSplitOptions.RemoveEmptyEntries);

        // Build PositionWithHistory with full real history (matches what the trainer/inference will see).
        PositionWithHistory pwh = new PositionWithHistory(startPos);
        try { foreach (string m in priorMoves) pwh.AppendMove(m); }
        catch { failed++; continue; }

        // Run deeper MCGS
        engine.ResetGame();
        GameEngineSearchResultCeresMCGS result;
        try { result = engine.SearchCeres(pwh, SearchLimit.NodesPerMove(opts.DeepenNodes)); }
        catch { failed++; continue; }

        if (result?.BestMoveInfo == null) { failed++; continue; }

        GNode root = result.Search.SearchRootNode;
        float w = Math.Max(0f, root.W);
        float l = Math.Max(0f, root.L);
        float d = Math.Max(0f, (float)root.D);
        float sum = w + l + d;
        if (sum > 0) { w /= sum; l /= sum; d /= sum; }
        float newV = w - l;

        float oldV = rec.TeacherV;
        float oldD = rec.TeacherD;

        sumOldQ += Math.Abs(oldV);
        sumNewQ += Math.Abs(newV);
        sumOldD += oldD;
        sumNewD += d;
        sumDeltaAbsQ += Math.Abs(newV) - Math.Abs(oldV);

        if (Math.Abs(newV) >= 0.3f) flippedToWin++;
        if (Math.Abs(newV) >= 0.5f) flippedToWinHard++;
        if (Math.Abs(newV) < 0.2f) stayedDrawish++;

        processed++;
        if (processed % 100 == 0)
        {
          double rate = processed / Math.Max(1, sw.Elapsed.TotalSeconds);
          Console.WriteLine($"[deepen-smoke] {processed}/{sample.Count} processed @ {rate:F1} rec/s, " +
                            $"|Q|: {sumOldQ/processed:F3} -> {sumNewQ/processed:F3}");
        }
      }
      sw.Stop();
      engine.Dispose();

      Console.WriteLine();
      Console.WriteLine("=== oppdef-deepen-smoke RESULTS ===");
      Console.WriteLine($"Sample:               {sample.Count}, processed: {processed}, failed: {failed}");
      Console.WriteLine($"Original node budget: {(processed > 0 ? sample[0].TeacherNodes : 0)} (per record TeacherNodes)");
      Console.WriteLine($"Deepened node budget: {opts.DeepenNodes}");
      Console.WriteLine($"Filter:               |old TeacherV| < {opts.DeepenQThreshold:F2}");
      Console.WriteLine($"Wall time:            {sw.Elapsed.TotalSeconds:F1} s ({(processed/Math.Max(1,sw.Elapsed.TotalSeconds)):F1} rec/s)");
      Console.WriteLine();
      if (processed > 0)
      {
        Console.WriteLine($"  Mean |Q|:   old={sumOldQ/processed:F3}  new={sumNewQ/processed:F3}  Δ={sumDeltaAbsQ/processed:+0.000;-0.000;0.000}");
        Console.WriteLine($"  Mean  D:    old={sumOldD/processed:F3}  new={sumNewD/processed:F3}");
        Console.WriteLine();
        Console.WriteLine($"  |new Q| >= 0.3:  {flippedToWin}/{processed} = {100.0*flippedToWin/processed:F1}%   (flipped from drawish to clearly winning/losing)");
        Console.WriteLine($"  |new Q| >= 0.5:  {flippedToWinHard}/{processed} = {100.0*flippedToWinHard/processed:F1}%   (flipped to strongly winning/losing)");
        Console.WriteLine($"  |new Q| <  0.2:  {stayedDrawish}/{processed} = {100.0*stayedDrawish/processed:F1}%   (stayed drawish)");
      }
    }


    static Dictionary<string, (string startFen, string[] moves)> LoadPuzzleMoves(string csvPath)
    {
      Dictionary<string, (string, string[])> map = new();
      using StreamReader r = new(csvPath);
      r.ReadLine();
      string line;
      while ((line = r.ReadLine()) != null)
      {
        string[] cols = line.Split(',');
        if (cols.Length < 3) continue;
        map[cols[0]] = (cols[1], cols[2].Split(' ', StringSplitOptions.RemoveEmptyEntries));
      }
      return map;
    }
  }
}
