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

namespace CeresTrain.TrainingDataGenerator.GeneratorFromPuzzles
{
  /// <summary>
  /// Composite end-to-end pipeline: mine → label → to-tpg.
  /// Each stage writes its artifact independently, so failures mid-pipeline
  /// leave the upstream artifacts intact and the pipeline can be resumed.
  /// </summary>
  public static class PuzzleReplayPipeline
  {
    public static void Run(PuzzleReplayOptions opts)
    {
      opts.Validate();

      Console.WriteLine();
      Console.WriteLine("=== PuzzleReplay :: Stage 1/3  Mine  ========================================");
      PuzzleMiner.MineStats mine = PuzzleMiner.Run(opts);

      Console.WriteLine();
      Console.WriteLine("=== PuzzleReplay :: Stage 2/3  Label ========================================");
      PuzzleTeacherLabeler.LabelStats label = PuzzleTeacherLabeler.Run(opts);

      Console.WriteLine();
      Console.WriteLine("=== PuzzleReplay :: Stage 3/3  ToTPG ========================================");
      PuzzleToTPGGenerator.EmitStats emit = PuzzleToTPGGenerator.Run(opts);

      Console.WriteLine();
      Console.WriteLine("=== PuzzleReplay :: Summary =================================================");
      Console.WriteLine($"  Mined:    {mine.Total:N0} total, {mine.Hard:N0} hard ({PercentOrZero(mine.Hard, mine.Total):F1}%)");
      Console.WriteLine($"  Labeled:  {label.Accepted:N0} accepted, {label.Rejected:N0} rejected ({PercentOrZero(label.Rejected, label.Processed):F1}% reject rate)");
      Console.WriteLine($"  TPG out:  {emit.Emitted:N0} positions written to {opts.TpgOutDir}");
    }


    static double PercentOrZero(long num, long denom) => denom == 0 ? 0 : 100.0 * num / denom;
  }
}
