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

namespace CeresTrain.TrainingDataGenerator.GeneratorFromPuzzles
{
  /// <summary>
  /// Fast labeler that bypasses NN teacher search entirely.
  /// Uses the Lichess solution (Stockfish-verified ground truth) as the policy answer,
  /// and derives WDL value targets from the puzzle's theme tags.
  ///
  /// ~1000x faster than teacher search. Lichess labels are deeper/stronger than our
  /// 100-node teacher anyway, so teacher-validation filtering was discarding good
  /// training data rather than catching Lichess errors. Filter is dropped here.
  ///
  /// Theme → WDL mapping (user-specified):
  ///   equality                         → W=0.15, D=0.70, L=0.15  (draw-heavy)
  ///   mate / mateInN                   → W=0.95, D=0.05, L=0.00
  ///   crushing / advantage / default   → W=0.85, D=0.10, L=0.05
  /// </summary>
  public static class PuzzleFastLabeler
  {
    public static Stats Run(PuzzleReplayOptions opts)
    {
      opts.Validate();
      Directory.CreateDirectory(opts.OutDir);

      long total = 0, emitted = 0, skipped = 0;
      Stopwatch sw = Stopwatch.StartNew();

      // Deduplication: same PuzzleId+FEN can appear across multiple puzzles/runs. Track
      // keys to avoid double-emission on resume.
      HashSet<string> alreadySeen = new HashSet<string>();
      bool appendMode = false;
      if (opts.ResumeFromCheckpoint && File.Exists(opts.LabeledJsonlPath))
      {
        foreach (LabeledPuzzleRecord r in JsonlIO.Read<LabeledPuzzleRecord>(opts.LabeledJsonlPath))
          alreadySeen.Add((r.PuzzleId ?? "") + "|" + (r.FEN ?? ""));
        if (alreadySeen.Count > 0)
        {
          appendMode = true;
          Console.WriteLine($"[fast-label] Resume: {alreadySeen.Count:N0} records already labeled.");
        }
      }

      using StreamWriter writer = new StreamWriter(opts.LabeledJsonlPath, append: appendMode);

      foreach (PuzzleSolvePosition p in LichessPuzzleReader.Read(
                 opts.LichessCsvPath, opts.MinRating, opts.MaxRating,
                 opts.ThemeIncludeAny, opts.ThemeExcludeAny, opts.MaxPuzzlesToRead))
      {
        total++;
        string key = p.PuzzleId + "|" + p.FEN;
        if (alreadySeen.Count > 0 && alreadySeen.Contains(key))
        {
          skipped++;
          continue;
        }

        (float w, float d, float l) = DeriveWDL(p.Themes);

        LabeledPuzzleRecord rec = new LabeledPuzzleRecord
        {
          PuzzleId = p.PuzzleId,
          FEN = p.FEN,
          SolutionUci = p.SolutionUci,
          Rating = p.Rating,
          Themes = p.Themes,
          StartFen = p.StartFen,
          PriorUciMoves = p.PriorUciMoves,
          TeacherNodes = 0,               // 0 marks "no search, direct Lichess label"
          TeacherTopUci = p.SolutionUci,  // trust Lichess
          TeacherV = w - l,
          TeacherW = w,
          TeacherD = d,
          TeacherL = l,
          // Single-entry policy: 100% mass on Lichess solution. PuzzleToTPGGenerator will
          // enumerate all legal moves and fill the rest with DEFAULT_MIN_PROBABILITY_LEGAL_MOVE
          // to keep the legal-move mask correct downstream.
          TeacherPolicy = new List<PolicyEntry> { new PolicyEntry { Uci = p.SolutionUci, P = 1.0f } },
        };
        JsonlIO.AppendLine(writer, rec);
        emitted++;

        if (emitted % 50_000 == 0)
        {
          writer.Flush();
          double pps = total / Math.Max(1, sw.Elapsed.TotalSeconds);
          Console.WriteLine($"[fast-label] {total:N0} read, {emitted:N0} emitted, {skipped:N0} skipped (resume), {pps:N0} pos/s");
        }
      }

      writer.Flush();
      sw.Stop();
      Console.WriteLine();
      Console.WriteLine($"[fast-label] Done.  Total={total:N0}  Emitted={emitted:N0}  Skipped={skipped:N0}  Elapsed={sw.Elapsed.TotalSeconds:F1}s");
      return new Stats { Total = total, Emitted = emitted, Skipped = skipped, ElapsedSec = sw.Elapsed.TotalSeconds };
    }


    /// <summary>
    /// Maps the space-separated Lichess themes string to a WDL target (from solver's perspective).
    /// Priority: equality > mate > crushing/advantage > default.
    /// </summary>
    static (float w, float d, float l) DeriveWDL(string themes)
    {
      if (string.IsNullOrEmpty(themes)) return (0.85f, 0.10f, 0.05f);

      ReadOnlySpan<char> themesSpan = themes.AsSpan();

      // Build a hashset once per puzzle for fast lookup (tokens are space-separated).
      HashSet<string> themeSet = new HashSet<string>(
        themes.Split(' ', StringSplitOptions.RemoveEmptyEntries),
        StringComparer.OrdinalIgnoreCase);

      // Only pure "equality" theme is draw-heavy. "defensiveMove" is a style tag and can
      // coexist with advantage/winning positions; do NOT treat it as a value class.
      if (themeSet.Contains("equality"))
        return (0.15f, 0.70f, 0.15f);

      // Any mate* theme → near-certain win.
      foreach (string t in themeSet)
        if (t.StartsWith("mate", StringComparison.OrdinalIgnoreCase))
          return (0.95f, 0.05f, 0.00f);

      // Default: crushing/advantage/tactical — strong advantage.
      return (0.85f, 0.10f, 0.05f);
    }


    public readonly record struct Stats(long Total, long Emitted, long Skipped, double ElapsedSec);
  }
}
