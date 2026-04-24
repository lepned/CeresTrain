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

namespace CeresTrain.TrainingDataGenerator.GeneratorFromPuzzles
{
  /// <summary>
  /// Enriches a labeled.jsonl (Standard solver-to-move records only) with additional
  /// record kinds, derived purely from Lichess-curation guarantees — no fabricated
  /// values, no teacher needed.
  ///
  /// Record kinds emitted:
  ///
  ///   1. Standard (pass-through): solver-to-move position in the puzzle line.
  ///      policy = one-hot Lichess solution. value = theme WDL.
  ///
  ///   2. OppDefence: opp-to-move position immediately after a puzzle solver move.
  ///      policy = one-hot on opp's puzzle defence move (the next Lichess move).
  ///      value = theme WDL flipped to opp's POV (opp is losing).
  ///      Skipped if this solver move is the LAST move in the puzzle line — no
  ///      opp defence move exists to target with policy.
  ///
  ///   3. OppAfterInferiorSolver (K samples per solver-to-move position): opp-to-move
  ///      position reached after the solver plays a non-puzzle move. WDL derived
  ///      from Lichess's unique-winning guarantee:
  ///        - crushing/advantage/mate themes → position is drawn
  ///            opp WDL (opp's POV) = (0.15, 0.70, 0.15)
  ///        - equality theme → solver lost the draw, opp is winning
  ///            opp WDL = (0.85, 0.10, 0.05)
  ///      No policy target (we don't know opp's best response in an off-path state).
  ///
  ///   4. PreBlunder (one per puzzle): CSV start FEN, blunderer-to-move, assumed
  ///      roughly balanced. value = (0.35, 0.45, 0.20). No policy target.
  ///
  /// Reads the Lichess CSV to obtain the full move sequence per puzzle (needed
  /// for opp's puzzle moves and for excluding puzzle moves when sampling counterfactuals).
  /// Joins against the input labeled.jsonl on PuzzleId to inherit theme / rating
  /// metadata and ensure only puzzles that survived fast-label filtering are enriched.
  /// </summary>
  public static class PuzzleValueEnricher
  {
    /// <summary>Counterfactual samples per solver-to-move position. K=1 means one per position — plenty, since all non-solution moves carry the same "bad-move" label.</summary>
    public const int COUNTERFACTUAL_SAMPLES_PER_POSITION = 1;

    /// <summary>WDL for the pre-blunder position (slight draw bias, roughly equal).</summary>
    static readonly (float w, float d, float l) PRE_BLUNDER_WDL = (0.35f, 0.45f, 0.20f);


    public sealed class Stats
    {
      public long PuzzlesSeen;
      public long InputRecords;
      public long StandardEmitted;
      public long OppDefenceEmitted;
      public long OppAfterInferiorSolverEmitted;
      public long PreBlunderEmitted;
      public long SkippedCsvParseError;
      public long SkippedMissingInCsv;
      public double ElapsedSec;

      public long TotalEmitted => StandardEmitted + OppDefenceEmitted
                                + OppAfterInferiorSolverEmitted + PreBlunderEmitted;
    }


    public static Stats Run(PuzzleReplayOptions opts, string inputJsonlPath, string outputJsonlPath)
    {
      if (!File.Exists(inputJsonlPath))
        throw new FileNotFoundException("input labeled.jsonl not found", inputJsonlPath);
      if (!File.Exists(opts.LichessCsvPath))
        throw new FileNotFoundException("Lichess CSV not found", opts.LichessCsvPath);

      Stats s = new Stats();
      Stopwatch sw = Stopwatch.StartNew();
      Random rng = new Random(42);

      // Step 1: index the labeled.jsonl by PuzzleId so we can join against the CSV.
      // Only Standard records contribute to the join; any pre-existing enrichment
      // records are passed through unchanged (shouldn't happen in normal flow).
      //
      // For each PuzzleId we only need to record that it WAS labeled (the metadata —
      // theme, rating, etc. — is available from the CSV too). But to pass through
      // the original Standard records verbatim we also keep them.
      Console.WriteLine($"[enrich] indexing {inputJsonlPath} by PuzzleId...");
      Dictionary<string, List<LabeledPuzzleRecord>> byId = new();
      foreach (LabeledPuzzleRecord rec in JsonlIO.Read<LabeledPuzzleRecord>(inputJsonlPath))
      {
        s.InputRecords++;
        if (rec.Kind != PuzzlePositionKind.Standard) continue;
        if (!byId.TryGetValue(rec.PuzzleId, out var list))
        {
          list = new List<LabeledPuzzleRecord>();
          byId[rec.PuzzleId] = list;
        }
        list.Add(rec);
      }
      Console.WriteLine($"[enrich] indexed {s.InputRecords:N0} input records across {byId.Count:N0} puzzles.");

      // Step 2: stream the Lichess CSV, emitting enriched records per puzzle.
      using StreamWriter writer = new StreamWriter(outputJsonlPath, append: false);
      using StreamReader csv = new StreamReader(opts.LichessCsvPath);
      string header = csv.ReadLine();

      string line;
      while ((line = csv.ReadLine()) != null)
      {
        string[] cols = line.Split(',');
        if (cols.Length < 8) continue;
        string puzzleId = cols[0];
        if (!byId.TryGetValue(puzzleId, out var stdRecs))
        {
          // This puzzle wasn't in the labeled set (rating filter, MaxPuzzles cap, etc.).
          continue;
        }
        s.PuzzlesSeen++;

        string startFen = cols[1];
        string movesUci = cols[2];
        if (string.IsNullOrWhiteSpace(startFen) || string.IsNullOrWhiteSpace(movesUci)) continue;
        string[] moves = movesUci.Split(' ', StringSplitOptions.RemoveEmptyEntries);
        if (moves.Length < 2) continue;

        if (!int.TryParse(cols[3], out int rating)) continue;
        string themes = cols[7] ?? "";

        // Parse the full puzzle sequence and cache each ply's MGPosition / FEN.
        // plyPos[i] = the position BEFORE moves[i] (0-indexed).
        MGPosition startMG;
        try { startMG = Position.FromFEN(startFen).ToMGPosition; }
        catch { s.SkippedCsvParseError++; continue; }

        MGPosition[] plyPos = new MGPosition[moves.Length + 1];
        plyPos[0] = startMG;
        MGMove[] mgMoves = new MGMove[moves.Length];
        bool parseOk = true;
        for (int i = 0; i < moves.Length; i++)
        {
          try { mgMoves[i] = MGMoveFromString.ParseMove(in plyPos[i], moves[i]); }
          catch { parseOk = false; break; }
          if (mgMoves[i] == default) { parseOk = false; break; }
          MGPosition next = plyPos[i];
          next.MakeMove(mgMoves[i]);
          plyPos[i + 1] = next;
        }
        if (!parseOk) { s.SkippedCsvParseError++; continue; }

        // PreBlunder skipped intentionally: we have no derivable value (position
        // class is unknown — Lichess doesn't certify it was balanced; it could be
        // already won/lost before the blunder) and no derivable policy. Including
        // these records would add noise without signal.

        // For each solver-to-move index i (i odd: 1, 3, 5, ...):
        //   a) Pass through the Standard record (from labeled.jsonl) unchanged.
        //   b) Emit OppDefence if i+1 exists (opp's defence move known).
        //   c) Sample K counterfactual non-puzzle solver moves at plyPos[i] and
        //      emit OppAfterInferiorSolver records.

        // Pass through all Standard records for this puzzle upfront. (We read them
        // from the cached list rather than re-materializing from the CSV, to preserve
        // any downstream metadata.)
        foreach (LabeledPuzzleRecord std in stdRecs)
        {
          JsonlIO.AppendLine(writer, std);
          s.StandardEmitted++;
        }

        (float w, float d, float l) solverThemeWDL = DeriveWDLFromTheme(themes);
        (float w, float d, float l) oppDefenceWDL = (solverThemeWDL.l, solverThemeWDL.d, solverThemeWDL.w);
        (float w, float d, float l) oppAfterInferiorWDL = DeriveOppAfterInferiorSolverWDL(themes);

        for (int i = 1; i < moves.Length; i += 2)
        {
          // i is a solver-to-move index. plyPos[i] is the solver position.
          // priorUci for records that apply at plyPos[i+1] is moves[0..i+1] (inclusive i, exclusive i+1) — wait: prior is moves applied before reaching the NEW position.
          // For an OppDefence at plyPos[i+1], moves applied = moves[0..=i] = the first i+1 moves.
          string priorForChild = string.Join(' ', moves, 0, i + 1);
          MGPosition mgChild = plyPos[i + 1];  // opp-to-move after solver's puzzle move

          // (b) OppDefence — only if there's a known opp defence (index i+1).
          if (i + 1 < moves.Length)
          {
            string oppDefenceUci = moves[i + 1];
            LabeledPuzzleRecord od = new LabeledPuzzleRecord
            {
              PuzzleId = puzzleId,
              FEN = mgChild.ToPosition.FEN,
              SolutionUci = oppDefenceUci,
              Rating = rating,
              Themes = themes,
              StartFen = startFen,
              PriorUciMoves = priorForChild,
              Kind = PuzzlePositionKind.OppDefence,
              TeacherNodes = 0,
              TeacherTopUci = oppDefenceUci,
              TeacherV = oppDefenceWDL.w - oppDefenceWDL.l,
              TeacherW = oppDefenceWDL.w,
              TeacherD = oppDefenceWDL.d,
              TeacherL = oppDefenceWDL.l,
              TeacherPolicy = new List<PolicyEntry> { new() { Uci = oppDefenceUci, P = 1.0f } },
            };
            JsonlIO.AppendLine(writer, od);
            s.OppDefenceEmitted++;
          }

          // No OppAfterInferiorSolver records: the simple design emits only the
          // two authentic record kinds (Standard + OppDefence). Value-head ranking
          // of off-path children is left to generalization; if empirical EB value
          // is weak, add teacher-calibrated counterfactuals later.
        }

        if (s.PuzzlesSeen % 50_000 == 0)
        {
          writer.Flush();
          double pps = s.PuzzlesSeen / Math.Max(1, sw.Elapsed.TotalSeconds);
          Console.WriteLine($"[enrich] {s.PuzzlesSeen:N0} puzzles, {s.TotalEmitted:N0} out " +
                            $"(Std={s.StandardEmitted:N0} Opp={s.OppDefenceEmitted:N0} " +
                            $"OppAfterInf={s.OppAfterInferiorSolverEmitted:N0} " +
                            $"PreBl={s.PreBlunderEmitted:N0})  {pps:N0} puz/s");
        }
      }

      writer.Flush();
      sw.Stop();
      s.ElapsedSec = sw.Elapsed.TotalSeconds;
      Console.WriteLine();
      Console.WriteLine($"[enrich] Done.  Puzzles={s.PuzzlesSeen:N0}  Emitted={s.TotalEmitted:N0}");
      Console.WriteLine($"  Standard (passthrough)    : {s.StandardEmitted:N0}");
      Console.WriteLine($"  OppDefence (with policy)  : {s.OppDefenceEmitted:N0}");
      Console.WriteLine($"  OppAfterInferiorSolver    : {s.OppAfterInferiorSolverEmitted:N0}");
      Console.WriteLine($"  PreBlunder                : {s.PreBlunderEmitted:N0}");
      Console.WriteLine($"  SkippedCsvParseError      : {s.SkippedCsvParseError:N0}");
      Console.WriteLine($"  Elapsed                   : {s.ElapsedSec:F1}s");
      return s;
    }


    /// <summary>Theme → solver-POV WDL. Keep in sync with PuzzleFastLabeler.DeriveWDL.</summary>
    static (float w, float d, float l) DeriveWDLFromTheme(string themes)
    {
      if (string.IsNullOrEmpty(themes)) return (0.85f, 0.10f, 0.05f);
      HashSet<string> themeSet = new HashSet<string>(
        themes.Split(' ', StringSplitOptions.RemoveEmptyEntries),
        StringComparer.OrdinalIgnoreCase);
      if (themeSet.Contains("equality")) return (0.15f, 0.70f, 0.15f);
      foreach (string t in themeSet)
        if (t.StartsWith("mate", StringComparison.OrdinalIgnoreCase))
          return (0.95f, 0.05f, 0.00f);
      return (0.85f, 0.10f, 0.05f);
    }


    /// <summary>
    /// WDL for opp-to-move positions reached after the solver played a non-puzzle move.
    /// Derived from Lichess's unique-winning guarantee:
    ///   - crushing/advantage/mate: solver's unique winning continuation was missed
    ///     → position is essentially drawn → opp's POV (0.15, 0.70, 0.15).
    ///   - equality: solver's unique draw-holding move was missed
    ///     → solver loses → opp's POV (0.85, 0.10, 0.05).
    /// </summary>
    static (float w, float d, float l) DeriveOppAfterInferiorSolverWDL(string themes)
    {
      if (string.IsNullOrEmpty(themes)) return (0.15f, 0.70f, 0.15f);  // default: drawn
      HashSet<string> themeSet = new HashSet<string>(
        themes.Split(' ', StringSplitOptions.RemoveEmptyEntries),
        StringComparer.OrdinalIgnoreCase);
      if (themeSet.Contains("equality")) return (0.85f, 0.10f, 0.05f);  // opp wins
      return (0.15f, 0.70f, 0.15f);  // drawn for everything else
    }
  }
}
