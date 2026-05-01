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
using System.Text.Json;

using Ceres.Chess;
using Ceres.Chess.MoveGen;
using Ceres.Chess.MoveGen.Converters;
using Ceres.Chess.NNEvaluators.Defs;
using Ceres.Chess.Positions;

using Ceres.MCGS.GameEngines;
using Ceres.MCGS.Graphs.GNodes;
using Ceres.MCGS.Graphs.GEdges;

namespace CeresTrain.TrainingDataGenerator.GeneratorFromPuzzles
{
  /// <summary>
  /// Adds OppDefence (opp-to-move) records to an existing labeled.jsonl by:
  ///   1. For each Standard record (solver-to-move), reconstruct the post-solver-move
  ///      position by applying SolutionUci.
  ///   2. Look up the puzzle's prescribed opponent reply (next move in the Lichess
  ///      puzzle line) from the CSV.
  ///   3. Run MCGS at TeacherNodes visits on the post-solver position, capture
  ///      search-backed W/D/L (from opp's perspective = side-to-move at that node).
  ///   4. Apply the same fixes as PuzzleTeacherLabeler: search-backed root.W/L/D
  ///      (not WinP/LossP/DrawP), rank-1 nudge to ensure opp's prescribed reply
  ///      tops the policy, clamp+renormalize negative WDL.
  ///   5. Emit a Kind=OppDefence record alongside the original Standard record.
  ///
  /// Hypothesis: the value head's collapse on full-body LoRA training (v201) is
  /// caused by missing supervision on opp-to-move positions. MCTS at inference
  /// reaches those positions and the head outputs miscalibrated values. Adding
  /// search-backed OppDefence records gives the head training signal on both
  /// position types, potentially curing the collapse.
  ///
  /// CRITICAL difference from prior OAIS records: those used heuristically-derived
  /// (theme-based) WDL targets and caused the draw-bias issue. THIS enricher uses
  /// search-backed Q from C3-MCGS, so the targets are properly calibrated.
  /// </summary>
  public static class PuzzleOppDefenceEnricher
  {
    public sealed class Stats
    {
      public long InputRecords;
      public long OppDefenceAdded;
      public long SkippedNoPuzzle;
      public long SkippedNoNextMove;
      public long SkippedSearchFailed;
      public double ElapsedSec;
    }


    public static Stats Run(PuzzleReplayOptions opts, string outputJsonlPath)
    {
      opts.Validate();
      if (!File.Exists(opts.LabeledJsonlPath))
        throw new FileNotFoundException("labeled.jsonl not found", opts.LabeledJsonlPath);

      Console.WriteLine($"[opp-def] Loading puzzle moves from CSV: {opts.LichessCsvPath}");
      Dictionary<string, (string startFen, string[] moves)> puzzleMoves = LoadPuzzleMoves(opts.LichessCsvPath);
      Console.WriteLine($"[opp-def] Loaded {puzzleMoves.Count:N0} puzzles");

      Console.WriteLine($"[opp-def] Loading teacher: {opts.NetSpec} on {opts.Device}");
      string deviceSpec = opts.Device.Contains('=') ? opts.Device : opts.Device + "=PuzzleOppDefence";
      NNEvaluatorDef evalDef = NNEvaluatorDefFactory.FromSpecification(opts.NetSpec, deviceSpec);

      Ceres.Chess.NNEvaluators.NNEvaluator warmupEval = Ceres.Chess.NNEvaluators.NNEvaluatorFactory.BuildEvaluator(evalDef);
      Console.WriteLine($"[opp-def] Warming up shared evaluator (MaxBatchSize={warmupEval.MaxBatchSize})...");
      Ceres.Chess.NNEvaluators.NNEvaluatorBenchmark.Warmup(warmupEval, warmupEval.MaxBatchSize);
      Console.WriteLine("[opp-def] Warmup complete.");

      GameEngineCeresMCGSInProcess engine = new GameEngineCeresMCGSInProcess(
        id: "PuzzleOppDefence",
        evaluatorDef: evalDef,
        moveImmediateIfOnlyOneMove: false);

      Stats s = new Stats();
      Stopwatch sw = Stopwatch.StartNew();

      JsonSerializerOptions jsonOpts = new() { PropertyNameCaseInsensitive = true };

      using StreamWriter writer = new StreamWriter(outputJsonlPath, append: false);

      foreach (LabeledPuzzleRecord rec in JsonlIO.Read<LabeledPuzzleRecord>(opts.LabeledJsonlPath))
      {
        s.InputRecords++;

        // Pass through the original record unchanged.
        writer.WriteLine(JsonSerializer.Serialize(rec, jsonOpts));

        // Only enrich Standard records (skip records that are themselves OppDefence/etc.).
        if (rec.Kind != PuzzlePositionKind.Standard) continue;

        LabeledPuzzleRecord oppDef = TryBuildOppDefence(engine, rec, puzzleMoves, opts.TeacherNodes, s);
        if (oppDef != null)
        {
          writer.WriteLine(JsonSerializer.Serialize(oppDef, jsonOpts));
          s.OppDefenceAdded++;
        }

        if (s.InputRecords % 500 == 0)
        {
          double elapsed = sw.Elapsed.TotalSeconds;
          double rate = s.OppDefenceAdded / Math.Max(1, elapsed);
          Console.WriteLine($"[opp-def] {s.InputRecords:N0} read, {s.OppDefenceAdded:N0} opp-def added, " +
                            $"{s.SkippedNoPuzzle + s.SkippedNoNextMove + s.SkippedSearchFailed:N0} skipped, " +
                            $"{rate:F1} adds/sec");
        }
      }

      sw.Stop();
      s.ElapsedSec = sw.Elapsed.TotalSeconds;
      engine.Dispose();

      Console.WriteLine($"[opp-def] Done. Input={s.InputRecords:N0} OppDefenceAdded={s.OppDefenceAdded:N0} " +
                        $"SkipNoPuzzle={s.SkippedNoPuzzle:N0} SkipNoNext={s.SkippedNoNextMove:N0} " +
                        $"SkipSearch={s.SkippedSearchFailed:N0} Elapsed={s.ElapsedSec:F1}s");
      return s;
    }


    static Dictionary<string, (string startFen, string[] moves)> LoadPuzzleMoves(string csvPath)
    {
      Dictionary<string, (string, string[])> map = new();
      using StreamReader r = new(csvPath);
      string header = r.ReadLine();  // skip header
      string line;
      while ((line = r.ReadLine()) != null)
      {
        // CSV columns: PuzzleId, FEN, Moves, Rating, RatingDeviation, Popularity, NbPlays, Themes, GameUrl, OpeningTags
        // Quick split by comma — FEN and Moves contain no commas.
        string[] cols = line.Split(',');
        if (cols.Length < 3) continue;
        string id = cols[0];
        string fen = cols[1];
        string moves = cols[2];
        map[id] = (fen, moves.Split(' ', StringSplitOptions.RemoveEmptyEntries));
      }
      return map;
    }


    /// <summary>FEN minus the halfmove and fullmove counters (last two fields). Used for position equality.</summary>
    static string FenKey(string fen)
    {
      if (string.IsNullOrEmpty(fen)) return fen;
      // FEN has 6 fields: pieces side castle ep half full. Keep first 4.
      int spaceCount = 0;
      for (int i = 0; i < fen.Length; i++)
      {
        if (fen[i] == ' ')
        {
          spaceCount++;
          if (spaceCount == 4) return fen.Substring(0, i);
        }
      }
      return fen;
    }


    static LabeledPuzzleRecord TryBuildOppDefence(GameEngineCeresMCGSInProcess engine,
                                                   LabeledPuzzleRecord rec,
                                                   Dictionary<string, (string startFen, string[] moves)> puzzleMoves,
                                                   int nodes,
                                                   Stats s)
    {
      // 1. Look up the puzzle's CSV row (start FEN + full move list).
      if (!puzzleMoves.TryGetValue(rec.PuzzleId, out (string startFen, string[] moves) puz))
      {
        s.SkippedNoPuzzle++;
        return null;
      }
      string[] moves = puz.moves;

      // 2. Resolve StartFen + priorMoves.
      //    Forward-compat: if the record already has them populated, trust them.
      //    Legacy fallback: derive them by walking the puzzle line and matching
      //    rec.FEN (counter-stripped) AND moves[i] == rec.SolutionUci.
      string startFenStr;
      string[] priorMoves;
      int posInMoves;

      if (!string.IsNullOrEmpty(rec.StartFen))
      {
        startFenStr = rec.StartFen;
        priorMoves = string.IsNullOrEmpty(rec.PriorUciMoves)
                     ? Array.Empty<string>()
                     : rec.PriorUciMoves.Split(' ', StringSplitOptions.RemoveEmptyEntries);
        posInMoves = priorMoves.Length;
        if (posInMoves >= moves.Length || moves[posInMoves] != rec.SolutionUci)
        {
          s.SkippedNoPuzzle++;
          return null;
        }
      }
      else
      {
        // Reconstruct: walk Lichess line, find solver-to-move position whose FEN matches rec.FEN.
        startFenStr = puz.startFen;
        Position startProbe;
        try { startProbe = Position.FromFEN(startFenStr); }
        catch { s.SkippedNoPuzzle++; return null; }

        string targetKey = FenKey(rec.FEN);
        MGPosition probe = startProbe.ToMGPosition;
        int matchIdx = -1;
        // Solver-to-move positions are at odd indices (1,3,5,...) — but be permissive
        // and check every position. Match requires both FEN equality and next-move match.
        for (int i = 0; i < moves.Length; i++)
        {
          string curKey = FenKey(probe.ToPosition.FEN);
          if (curKey == targetKey && moves[i] == rec.SolutionUci)
          {
            matchIdx = i;
            break;
          }
          MGMove pm;
          try { pm = MGMoveFromString.ParseMove(in probe, moves[i]); }
          catch { matchIdx = -2; break; }
          if (pm == default) { matchIdx = -2; break; }
          probe.MakeMove(pm);
        }
        if (matchIdx < 0)
        {
          s.SkippedNoPuzzle++;
          return null;
        }
        posInMoves = matchIdx;
        priorMoves = new string[matchIdx];
        Array.Copy(moves, 0, priorMoves, 0, matchIdx);
      }

      // 3. Opp's prescribed reply is the next move after solver's.
      int oppIdx = posInMoves + 1;
      if (oppIdx >= moves.Length)
      {
        // Puzzle ends here (solver's move was the final mate, no opp reply).
        s.SkippedNoNextMove++;
        return null;
      }
      string oppMove = moves[oppIdx];

      // 4. Build the post-solver-move position by replaying the line.
      Position startPos;
      try { startPos = Position.FromFEN(startFenStr); }
      catch { s.SkippedNoPuzzle++; return null; }

      MGPosition mgPos = startPos.ToMGPosition;
      try
      {
        foreach (string m in priorMoves)
        {
          MGMove pm = MGMoveFromString.ParseMove(in mgPos, m);
          if (pm == default) { s.SkippedNoPuzzle++; return null; }
          mgPos.MakeMove(pm);
        }
        // Apply solver's move.
        MGMove solverMG = MGMoveFromString.ParseMove(in mgPos, rec.SolutionUci);
        if (solverMG == default) { s.SkippedNoPuzzle++; return null; }
        mgPos.MakeMove(solverMG);
      }
      catch { s.SkippedNoPuzzle++; return null; }

      Position oppPos = mgPos.ToPosition;

      // 5. Build PositionWithHistory with full real history (matches what the trainer/inference will see).
      PositionWithHistory pwh = new PositionWithHistory(startPos);
      try
      {
        foreach (string m in priorMoves) pwh.AppendMove(m);
        pwh.AppendMove(rec.SolutionUci);
      }
      catch { s.SkippedNoPuzzle++; return null; }

      // 6. Run MCGS search.
      engine.ResetGame();
      GameEngineSearchResultCeresMCGS result;
      try
      {
        result = engine.SearchCeres(pwh, SearchLimit.NodesPerMove(nodes));
      }
      catch
      {
        s.SkippedSearchFailed++;
        return null;
      }

      if (result?.BestMoveInfo == null)
      {
        s.SkippedSearchFailed++;
        return null;
      }

      // 7. Capture search-backed W/L/D (NOT static WinP/LossP/DrawP — same fix as in PuzzleTeacherLabeler).
      GNode root = result.Search.SearchRootNode;
      float rootW = root.W;
      float rootL = root.L;
      float rootD = (float)root.D;

      // Clamp + renormalize (MCGS aggregation rounding can produce small negatives).
      if (rootW < 0f) rootW = 0f;
      if (rootL < 0f) rootL = 0f;
      if (rootD < 0f) rootD = 0f;
      float wdlSum = rootW + rootL + rootD;
      if (wdlSum > 0f) { rootW /= wdlSum; rootL /= wdlSum; rootD /= wdlSum; }
      float rootV = rootW - rootL;

      string teacherTopUci = "";
      MGMove teacherTopMG = result.BestMoveMG;
      if (teacherTopMG != default)
      {
        teacherTopUci = teacherTopMG.MoveStr(MGMoveNotationStyle.Coordinates);
      }

      // 8. Build policy distribution from search root's child edges.
      List<PolicyEntry> policy = new List<PolicyEntry>();
      long totalN = 0;
      GEdge[] sortedEdges = root.NumEdgesExpanded == 0 ? null : root.EdgesSorted(node => node.N);
      if (sortedEdges != null)
      {
        foreach (GEdge edge in sortedEdges)
        {
          if (!edge.IsExpanded) continue;
          totalN += edge.ChildNode.N;
        }
        if (totalN > 0)
        {
          foreach (GEdge edge in sortedEdges)
          {
            if (!edge.IsExpanded) continue;
            long childN = edge.ChildNode.N;
            if (childN <= 0) continue;
            MGMove moveMG = edge.MoveMG;
            policy.Add(new PolicyEntry
            {
              Uci = moveMG == default ? "" : moveMG.MoveStr(MGMoveNotationStyle.Coordinates),
              P = (float)childN / totalN,
            });
          }
        }
      }

      // 9. Rank-1 nudge: ensure opp's prescribed reply is the unique top of the policy by epsilon.
      const float RANK_ONE_EPSILON = 0.03f;
      int oppIdxInPolicy = -1;
      float maxNonOppP = 0f;
      for (int i = 0; i < policy.Count; i++)
      {
        if (policy[i].Uci == oppMove) oppIdxInPolicy = i;
        else if (policy[i].P > maxNonOppP) maxNonOppP = policy[i].P;
      }
      bool needsRenormalize = false;
      if (oppIdxInPolicy < 0)
      {
        policy.Add(new PolicyEntry { Uci = oppMove, P = maxNonOppP + RANK_ONE_EPSILON });
        needsRenormalize = true;
      }
      else if (policy[oppIdxInPolicy].P < maxNonOppP + RANK_ONE_EPSILON)
      {
        PolicyEntry e = policy[oppIdxInPolicy];
        policy[oppIdxInPolicy] = new PolicyEntry { Uci = e.Uci, P = maxNonOppP + RANK_ONE_EPSILON };
        needsRenormalize = true;
      }
      if (needsRenormalize)
      {
        float total = 0f;
        foreach (PolicyEntry pe in policy) total += pe.P;
        if (total > 0f)
        {
          for (int i = 0; i < policy.Count; i++)
          {
            PolicyEntry pe = policy[i];
            policy[i] = new PolicyEntry { Uci = pe.Uci, P = pe.P / total };
          }
        }
      }

      // 10. Build the OppDefence LabeledPuzzleRecord.
      string newPriorUci = (priorMoves.Length == 0)
                          ? rec.SolutionUci
                          : string.Join(' ', priorMoves) + " " + rec.SolutionUci;

      return new LabeledPuzzleRecord
      {
        PuzzleId = rec.PuzzleId,
        FEN = oppPos.FEN,
        SolutionUci = oppMove,
        Rating = rec.Rating,
        Themes = rec.Themes,
        Kind = PuzzlePositionKind.OppDefence,
        StartFen = startFenStr,
        PriorUciMoves = newPriorUci,
        TeacherNodes = nodes,
        TeacherTopUci = teacherTopUci,
        TeacherV = rootV,
        TeacherW = rootW,
        TeacherD = rootD,
        TeacherL = rootL,
        TeacherPolicy = policy,
      };
    }
  }
}
