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
using System.IO;
using System.Linq;

using Ceres.Chess;
using Ceres.Chess.MoveGen;
using Ceres.Chess.MoveGen.Converters;
using Ceres.Chess.Positions;
using Ceres.Chess.Textual;

namespace CeresTrain.TrainingDataGenerator.GeneratorFromPuzzles
{
  /// <summary>
  /// One solver-to-move position extracted from a Lichess puzzle, with the Lichess-declared solution move.
  /// A single puzzle produces one record per solver-to-move position in the line (Q1(c)).
  /// </summary>
  public readonly record struct PuzzleSolvePosition
  {
    /// <summary>Lichess puzzle identifier.</summary>
    public string PuzzleId { get; init; }

    /// <summary>FEN of the position where the solver is to move.</summary>
    public string FEN { get; init; }

    /// <summary>Lichess solution move at this position (UCI, e.g. e2e4).</summary>
    public string SolutionUci { get; init; }

    /// <summary>Zero-based index of this solve step within the puzzle's solver sequence.</summary>
    public int SolveStepIndex { get; init; }

    /// <summary>Puzzle rating (Glicko).</summary>
    public int Rating { get; init; }

    /// <summary>Lichess-tagged themes (space-joined from the CSV).</summary>
    public string Themes { get; init; }

    /// <summary>Populated Position (derived from FEN) for downstream inference. Not serialized.</summary>
    public Position Position { get; init; }

    /// <summary>MGMove form of the solution, for policy-head comparisons and move encoding. Not serialized.</summary>
    public MGMove SolutionMG { get; init; }

    /// <summary>CSV start FEN (pre-setup-move).</summary>
    public string StartFen { get; init; }

    /// <summary>Space-joined UCI prefix (setup + prior moves) applied to StartFen to reach FEN.</summary>
    public string PriorUciMoves { get; init; }
  }


  /// <summary>
  /// Streams PuzzleSolvePosition records from the Lichess puzzle CSV.
  /// Handles the first-move-is-setup convention and expands each puzzle
  /// to every solver-to-move position in the line.
  /// </summary>
  public static class LichessPuzzleReader
  {
    /// <summary>
    /// Expected CSV header:
    /// PuzzleId,FEN,Moves,Rating,RatingDeviation,Popularity,NbPlays,Themes,GameUrl,OpeningTags
    /// </summary>
    const int COL_PUZZLE_ID = 0;
    const int COL_FEN = 1;
    const int COL_MOVES = 2;
    const int COL_RATING = 3;
    const int COL_THEMES = 7;


    /// <summary>
    /// Enumerates solve positions from the CSV.
    /// </summary>
    public static IEnumerable<PuzzleSolvePosition> Read(string csvPath,
                                                        int minRating = 0,
                                                        int maxRating = int.MaxValue,
                                                        string themeIncludeAny = null,
                                                        string themeExcludeAny = null,
                                                        int maxPuzzles = int.MaxValue)
    {
      if (!File.Exists(csvPath))
        throw new FileNotFoundException("Lichess puzzle CSV not found", csvPath);

      string[] includeThemes = SplitThemes(themeIncludeAny);
      string[] excludeThemes = SplitThemes(themeExcludeAny);

      int puzzlesYielded = 0;
      using StreamReader sr = new StreamReader(csvPath);
      string header = sr.ReadLine();
      if (header == null) yield break;

      string line;
      while ((line = sr.ReadLine()) != null && puzzlesYielded < maxPuzzles)
      {
        string[] cols = SplitCsvLine(line);
        if (cols.Length < 8) continue;

        if (!int.TryParse(cols[COL_RATING], out int rating)) continue;
        if (rating < minRating || rating > maxRating) continue;

        string themes = cols[COL_THEMES] ?? "";
        if (includeThemes != null && !AnyThemeMatches(themes, includeThemes)) continue;
        if (excludeThemes != null && AnyThemeMatches(themes, excludeThemes)) continue;

        string puzzleId = cols[COL_PUZZLE_ID];
        string startFen = cols[COL_FEN];
        string movesUci = cols[COL_MOVES];
        if (string.IsNullOrWhiteSpace(startFen) || string.IsNullOrWhiteSpace(movesUci)) continue;

        bool anyYielded = false;
        foreach (PuzzleSolvePosition p in ExpandPuzzle(puzzleId, startFen, movesUci, rating, themes))
        {
          anyYielded = true;
          yield return p;
        }
        if (anyYielded) puzzlesYielded++;
      }
    }


    /// <summary>
    /// Applies the setup move, then yields one record per solver-to-move position.
    /// Q1(c) expansion: every solver-to-move position in the line is emitted.
    /// Downstream mining decides which of them count as "hard".
    /// </summary>
    static IEnumerable<PuzzleSolvePosition> ExpandPuzzle(string puzzleId, string startFen,
                                                         string movesUci, int rating, string themes)
    {
      string[] moves = movesUci.Split(' ', StringSplitOptions.RemoveEmptyEntries);
      if (moves.Length < 2) yield break;

      Position pos;
      try { pos = Position.FromFEN(startFen); }
      catch { yield break; }

      MGPosition mg = pos.ToMGPosition;

      // Index 0 in Lichess CSV is the opponent setup move; solver moves start at index 1.
      for (int i = 0; i < moves.Length; i++)
      {
        MGMove mgMove;
        try { mgMove = MGMoveFromString.ParseMove(in mg, moves[i]); }
        catch { yield break; }
        if (mgMove == default) yield break;

        bool isSolverMove = (i % 2 == 1);
        if (isSolverMove)
        {
          Position solverPos = mg.ToPosition;
          // Prefix = moves[0..i] (setup + all opponent/solver moves already applied).
          string priorUciMoves = string.Join(' ', moves, 0, i);
          yield return new PuzzleSolvePosition
          {
            PuzzleId = puzzleId,
            FEN = solverPos.FEN,
            SolutionUci = moves[i],
            SolveStepIndex = (i - 1) / 2,
            Rating = rating,
            Themes = themes,
            Position = solverPos,
            SolutionMG = mgMove,
            StartFen = startFen,
            PriorUciMoves = priorUciMoves,
          };
        }

        mg.MakeMove(mgMove);
      }
    }


    /// <summary>
    /// Minimal CSV split. Lichess puzzle CSV fields do not contain commas or quoting,
    /// so a plain split is safe for this specific schema.
    /// </summary>
    static string[] SplitCsvLine(string line) => line.Split(',');


    static string[] SplitThemes(string themes)
    {
      if (string.IsNullOrWhiteSpace(themes)) return null;
      return themes.Split(new[] { ',', ' ', ';' }, StringSplitOptions.RemoveEmptyEntries);
    }


    static bool AnyThemeMatches(string themesField, string[] candidates)
    {
      if (string.IsNullOrEmpty(themesField)) return false;
      HashSet<string> present = new HashSet<string>(
        themesField.Split(' ', StringSplitOptions.RemoveEmptyEntries),
        StringComparer.OrdinalIgnoreCase);
      foreach (string c in candidates)
        if (present.Contains(c)) return true;
      return false;
    }
  }
}
