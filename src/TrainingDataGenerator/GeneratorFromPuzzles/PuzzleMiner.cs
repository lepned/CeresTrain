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
using Ceres.Chess.NetEvaluation.Batch;
using Ceres.Chess.NNEvaluators;
using Ceres.Chess.Positions;

namespace CeresTrain.TrainingDataGenerator.GeneratorFromPuzzles
{
  /// <summary>
  /// Evaluates a stream of puzzle solve positions at nodes=1 (raw policy head) and
  /// emits one HardPuzzleRecord to JSONL for every position where the student's top
  /// policy move disagrees with the Lichess solution.
  /// </summary>
  public static class PuzzleMiner
  {
    /// <summary>
    /// Mines hard puzzles from the configured Lichess CSV using the given net.
    /// Writes to <see cref="PuzzleReplayOptions.HardJsonlPath"/>.
    /// </summary>
    public static MineStats Run(PuzzleReplayOptions opts)
    {
      opts.Validate();

      string deviceSpec = EnsureSharedName(opts.Device, "PuzzleMiner");
      NNEvaluator evaluator = NNEvaluator.FromSpecification(opts.NetSpec, deviceSpec);

      long total = 0, hard = 0;
      Stopwatch sw = Stopwatch.StartNew();

      using StreamWriter writer = new StreamWriter(opts.HardJsonlPath);

      IEnumerable<PuzzleSolvePosition> source = LichessPuzzleReader.Read(
        opts.LichessCsvPath,
        opts.MinRating, opts.MaxRating,
        opts.ThemeIncludeAny, opts.ThemeExcludeAny,
        opts.MaxPuzzlesToRead);

      foreach (IReadOnlyList<PuzzleSolvePosition> batch in Batched(source, opts.MineBatchSize))
      {
        List<PositionWithHistory> pwhBatch = new List<PositionWithHistory>(batch.Count);
        foreach (PuzzleSolvePosition p in batch)
          pwhBatch.Add(new PositionWithHistory(p.Position));

        NNEvaluatorResult[] results = evaluator.Evaluate(pwhBatch, fillInMissingPlanes: true);

        for (int i = 0; i < batch.Count; i++)
        {
          PuzzleSolvePosition p = batch[i];
          NNEvaluatorResult r = results[i];

          Position pos = p.Position;
          MGMove studentTop = r.Policy.TopMove(in pos);
          bool wrong = (studentTop == default) || !(studentTop == p.SolutionMG);

          total++;
          if (wrong)
          {
            hard++;
            JsonlIO.AppendLine(writer, new HardPuzzleRecord
            {
              PuzzleId = p.PuzzleId,
              FEN = p.FEN,
              SolutionUci = p.SolutionUci,
              SolveStepIndex = p.SolveStepIndex,
              Rating = p.Rating,
              Themes = p.Themes,
              StudentTopUci = studentTop == default ? "" : studentTop.MoveStr(MGMoveNotationStyle.Coordinates),
              StudentV = r.V,
            });
          }
        }

        if (total % 10_000 < opts.MineBatchSize)
        {
          Console.WriteLine($"[mine] {total:N0} positions, {hard:N0} hard " +
                            $"({(total == 0 ? 0 : 100.0 * hard / total):F1}%), " +
                            $"{total / Math.Max(1, sw.Elapsed.TotalSeconds):N0} pos/s");
        }
      }

      writer.Flush();
      sw.Stop();
      MineStats stats = new MineStats { Total = total, Hard = hard, ElapsedSec = sw.Elapsed.TotalSeconds };
      Console.WriteLine($"[mine] Done. Total={stats.Total:N0} Hard={stats.Hard:N0} " +
                        $"HardRate={(stats.Total == 0 ? 0 : 100.0 * stats.Hard / stats.Total):F1}% " +
                        $"Elapsed={stats.ElapsedSec:F1}s");
      return stats;
    }


    /// <summary>
    /// Append "=name" to the device spec so the NNEvaluator is registered as shared.
    /// Enables one underlying GPU evaluator even across repeated constructions.
    /// </summary>
    static string EnsureSharedName(string deviceSpec, string name) =>
      deviceSpec.Contains('=') ? deviceSpec : deviceSpec + "=" + name;


    /// <summary>Chunks an IEnumerable into fixed-size batches.</summary>
    static IEnumerable<IReadOnlyList<T>> Batched<T>(IEnumerable<T> source, int size)
    {
      List<T> buf = new List<T>(size);
      foreach (T item in source)
      {
        buf.Add(item);
        if (buf.Count >= size)
        {
          yield return buf;
          buf = new List<T>(size);
        }
      }
      if (buf.Count > 0) yield return buf;
    }


    public readonly record struct MineStats(long Total, long Hard, double ElapsedSec);
  }
}
