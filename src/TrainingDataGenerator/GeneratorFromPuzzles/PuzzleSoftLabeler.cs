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
using Ceres.Chess.EncodedPositions;
using Ceres.Chess.MoveGen;
using Ceres.Chess.MoveGen.Converters;
using Ceres.Chess.NetEvaluation.Batch;
using Ceres.Chess.NNEvaluators;
using Ceres.Chess.Positions;

namespace CeresTrain.TrainingDataGenerator.GeneratorFromPuzzles
{
  /// <summary>
  /// Rank-1 soft-label puzzle dataset generator.
  ///
  /// For every solver-to-move ply of every puzzle, and every opp-to-move ply that
  /// has a known puzzle reply, runs the teacher NN forward once and builds:
  ///
  ///   TeacherPolicy: orig's full policy distribution, minimally nudged so the
  ///                  Lichess-verified move (solution for Standard, opp's reply
  ///                  for OppDefence) is the argmax. Additive epsilon margin
  ///                  before renormalisation.
  ///
  ///   TeacherW/D/L:  orig's raw WDL output, minimally nudged so the
  ///                  theme-implied class is the argmax:
  ///                    solver-POV crushing/mate/advantage → W dominant
  ///                    solver-POV equality                 → D dominant
  ///                    opp-POV    crushing/mate/advantage → L dominant
  ///                    opp-POV    equality                 → D dominant
  ///                  Same additive-epsilon-then-renormalise rule as policy.
  ///
  /// Records where orig already ranked the theme/move on top pass through
  /// essentially unchanged (pure distillation). Records where orig disagreed
  /// get the smallest correction needed to flip the argmax — adaptive to how
  /// wrong orig was.
  ///
  /// OppAfterInferiorSolver records are deliberately NOT emitted — we have no
  /// principled per-position ground truth for off-path blunder children, and
  /// v10 showed that coarse bucket-labels there collapse the value head.
  /// </summary>
  public static class PuzzleSoftLabeler
  {
    /// <summary>
    /// Pre-normalise additive margin used by both policy and WDL rank-1 rules.
    /// Too small and orig-severe-disagree cases produce near-ties in the target;
    /// too large and we distort orig's distribution unnecessarily.
    /// </summary>
    public const float EPSILON = 0.03f;


    public sealed class Stats
    {
      public long PuzzlesSeen;
      public long StandardEmitted;
      public long OppDefenceEmitted;
      public long SkippedCsvParseError;
      public long SkippedTerminalPosition;
      public long SkippedKnownMoveNotInPolicy;
      public long PolicyNudged;
      public long WDLNudged;
      public double ElapsedSec;

      public long TotalEmitted => StandardEmitted + OppDefenceEmitted;
    }


    sealed class PendingRecord
    {
      public PuzzlePositionKind Kind;
      public string PuzzleId;
      public string FEN;
      public int Rating;
      public string Themes;
      public string StartFen;
      public string PriorUciMoves;
      public string KnownMoveUci;
      public Position Position;
      public bool OppPov;  // false = solver-POV (Standard), true = opp-POV (OppDefence)
    }


    public static Stats Run(PuzzleReplayOptions opts, string outputJsonlPath)
    {
      if (!File.Exists(opts.LichessCsvPath))
        throw new FileNotFoundException("Lichess CSV not found", opts.LichessCsvPath);
      if (string.IsNullOrWhiteSpace(opts.NetSpec))
        throw new InvalidOperationException("NetSpec must be set in puzzle-config (teacher network)");

      Console.WriteLine($"[soft-label] Loading teacher: {opts.NetSpec} on {opts.Device}");
      NNEvaluator evaluator = NNEvaluator.FromSpecification(opts.NetSpec, opts.Device);
      int batchSize = Math.Max(64, opts.MineBatchSize);
      Console.WriteLine($"[soft-label] Batch size: {batchSize}, EPSILON={EPSILON}");

      Stats s = new Stats();
      Stopwatch sw = Stopwatch.StartNew();

      using StreamWriter writer = new StreamWriter(outputJsonlPath, append: false);
      using StreamReader csv = new StreamReader(opts.LichessCsvPath);
      _ = csv.ReadLine();  // skip CSV header

      List<PositionWithHistory> pwhBatch = new List<PositionWithHistory>(batchSize);
      List<PendingRecord> pendingBatch = new List<PendingRecord>(batchSize);

      int maxPuzzles = opts.MaxPuzzlesToRead <= 0 ? int.MaxValue : opts.MaxPuzzlesToRead;

      string line;
      while ((line = csv.ReadLine()) != null && s.PuzzlesSeen < maxPuzzles)
      {
        string[] cols = line.Split(',');
        if (cols.Length < 8) continue;

        string puzzleId = cols[0];
        string startFen = cols[1];
        string movesUci = cols[2];
        if (string.IsNullOrWhiteSpace(startFen) || string.IsNullOrWhiteSpace(movesUci)) continue;
        if (!int.TryParse(cols[3], out int rating)) continue;
        if (rating < opts.MinRating || rating > opts.MaxRating) continue;
        string themes = cols[7] ?? "";

        string[] moves = movesUci.Split(' ', StringSplitOptions.RemoveEmptyEntries);
        if (moves.Length < 2) continue;

        // Replay moves to build plyPos[].
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
          plyPos[i + 1] = plyPos[i];
          plyPos[i + 1].MakeMove(mgMoves[i]);
        }
        if (!parseOk) { s.SkippedCsvParseError++; continue; }

        s.PuzzlesSeen++;

        // For each solver ply i (odd), queue a Standard record at plyPos[i]
        // and an OppDefence record at plyPos[i+1] if opp's reply exists.
        for (int i = 1; i < moves.Length; i += 2)
        {
          // Standard: solver-to-move at plyPos[i], known move = moves[i].
          QueuePending(pwhBatch, pendingBatch, s,
                       puzzleId, rating, themes, startFen,
                       priorForRecord: string.Join(' ', moves, 0, i),
                       recordMG: plyPos[i],
                       knownMoveUci: moves[i],
                       kind: PuzzlePositionKind.Standard,
                       oppPov: false);

          // OppDefence: opp-to-move at plyPos[i+1], known move = moves[i+1].
          if (i + 1 < moves.Length)
          {
            QueuePending(pwhBatch, pendingBatch, s,
                         puzzleId, rating, themes, startFen,
                         priorForRecord: string.Join(' ', moves, 0, i + 1),
                         recordMG: plyPos[i + 1],
                         knownMoveUci: moves[i + 1],
                         kind: PuzzlePositionKind.OppDefence,
                         oppPov: true);
          }

          if (pwhBatch.Count >= batchSize)
          {
            FlushBatch(evaluator, pwhBatch, pendingBatch, writer, s);
            pwhBatch.Clear();
            pendingBatch.Clear();
          }
        }

        if (s.PuzzlesSeen % 50_000 == 0)
        {
          writer.Flush();
          double pps = s.PuzzlesSeen / Math.Max(1, sw.Elapsed.TotalSeconds);
          Console.WriteLine($"[soft-label] {s.PuzzlesSeen:N0} puzzles, {s.TotalEmitted:N0} records " +
                            $"(Std={s.StandardEmitted:N0} Opp={s.OppDefenceEmitted:N0}) " +
                            $"PolicyNudged={s.PolicyNudged:N0} WDLNudged={s.WDLNudged:N0}, {pps:N0} puz/s");
        }
      }

      // Final flush.
      if (pwhBatch.Count > 0)
      {
        FlushBatch(evaluator, pwhBatch, pendingBatch, writer, s);
        pwhBatch.Clear();
        pendingBatch.Clear();
      }

      writer.Flush();
      sw.Stop();
      s.ElapsedSec = sw.Elapsed.TotalSeconds;
      Console.WriteLine();
      Console.WriteLine($"[soft-label] Done. Puzzles={s.PuzzlesSeen:N0} Emitted={s.TotalEmitted:N0}");
      Console.WriteLine($"  Standard                    : {s.StandardEmitted:N0}");
      Console.WriteLine($"  OppDefence                  : {s.OppDefenceEmitted:N0}");
      Console.WriteLine($"  SkippedCsvParseError        : {s.SkippedCsvParseError:N0}");
      Console.WriteLine($"  SkippedTerminalPosition     : {s.SkippedTerminalPosition:N0}");
      Console.WriteLine($"  SkippedKnownMoveNotInPolicy : {s.SkippedKnownMoveNotInPolicy:N0}");
      Console.WriteLine($"  PolicyNudged                : {s.PolicyNudged:N0} (records where orig's top wasn't the known move)");
      Console.WriteLine($"  WDLNudged                   : {s.WDLNudged:N0} (records where orig's argmax disagreed with theme)");
      Console.WriteLine($"  Elapsed                     : {s.ElapsedSec:F1}s ({s.ElapsedSec / 60:F1} min)");
      return s;
    }


    static void QueuePending(List<PositionWithHistory> pwhBatch, List<PendingRecord> pendingBatch, Stats s,
                             string puzzleId, int rating, string themes, string startFen,
                             string priorForRecord, MGPosition recordMG, string knownMoveUci,
                             PuzzlePositionKind kind, bool oppPov)
    {
      // Skip terminal positions — NN policy extraction is undefined with zero legal moves.
      MGMoveList legal = new MGMoveList();
      MGMoveGen.GenerateMoves(in recordMG, legal);
      if (legal.NumMovesUsed == 0) { s.SkippedTerminalPosition++; return; }

      Position recordPos = recordMG.ToPosition;
      PositionWithHistory pwh = BuildPwh(startFen, priorForRecord, recordPos);
      if (pwh == null) { s.SkippedCsvParseError++; return; }

      pwhBatch.Add(pwh);
      pendingBatch.Add(new PendingRecord
      {
        Kind = kind,
        PuzzleId = puzzleId,
        FEN = recordPos.FEN,
        Rating = rating,
        Themes = themes,
        StartFen = startFen,
        PriorUciMoves = priorForRecord,
        KnownMoveUci = knownMoveUci,
        Position = recordPos,
        OppPov = oppPov,
      });
    }


    static void FlushBatch(NNEvaluator evaluator,
                           List<PositionWithHistory> pwhBatch, List<PendingRecord> pendingBatch,
                           StreamWriter writer, Stats s)
    {
      if (pwhBatch.Count == 0) return;

      NNEvaluatorResult[] results = evaluator.Evaluate(pwhBatch, fillInMissingPlanes: true);

      for (int i = 0; i < pendingBatch.Count; i++)
      {
        PendingRecord p = pendingBatch[i];
        NNEvaluatorResult r = results[i];

        // Build policy entries (one per legal move) with rank-1 correction on knownMoveUci.
        (List<PolicyEntry> policyEntries, bool policyNudged, bool knownFound)
          = BuildRank1Policy(r.Policy, p.Position, p.KnownMoveUci);
        if (!knownFound) { s.SkippedKnownMoveNotInPolicy++; continue; }
        if (policyNudged) s.PolicyNudged++;

        // Build WDL with rank-1 correction on theme-dominant class.
        (float w, float d, float l, bool wdlNudged) = BuildRank1WDL(r.W, r.L, p.Themes, p.OppPov);
        if (wdlNudged) s.WDLNudged++;

        LabeledPuzzleRecord rec = new LabeledPuzzleRecord
        {
          PuzzleId = p.PuzzleId,
          FEN = p.FEN,
          SolutionUci = p.KnownMoveUci,
          Rating = p.Rating,
          Themes = p.Themes,
          Kind = p.Kind,
          StartFen = p.StartFen,
          PriorUciMoves = p.PriorUciMoves,
          TeacherNodes = 1,
          TeacherTopUci = p.KnownMoveUci,
          TeacherV = w - l,
          TeacherW = w,
          TeacherD = d,
          TeacherL = l,
          TeacherPolicy = policyEntries,
        };
        JsonlIO.AppendLine(writer, rec);

        if (p.Kind == PuzzlePositionKind.Standard) s.StandardEmitted++;
        else if (p.Kind == PuzzlePositionKind.OppDefence) s.OppDefenceEmitted++;
      }
    }


    /// <summary>
    /// Applies the rank-1-on-move rule to a policy distribution: the knownMove's
    /// P is lifted to (maxOtherP + EPSILON) if it's not already the top, then the
    /// whole distribution is renormalised to sum to 1.
    /// Returns (entries, nudged, knownMoveFound).
    /// </summary>
    static (List<PolicyEntry>, bool, bool) BuildRank1Policy(
      CompressedPolicyVector policy,
      Position pos, string knownMoveUci)
    {
      List<(MGMove move, float p)> all =
        policy.MGMovesAndProbabilities(pos).Where(x => x.Probability > 0).ToList();
      if (all.Count == 0) return (null, false, false);

      int knownIdx = -1;
      float topOtherP = 0f;
      for (int k = 0; k < all.Count; k++)
      {
        string uci = all[k].move.MoveStr(MGMoveNotationStyle.Coordinates);
        if (uci == knownMoveUci)
          knownIdx = k;
      }
      if (knownIdx < 0) return (null, false, false);

      for (int k = 0; k < all.Count; k++)
      {
        if (k == knownIdx) continue;
        if (all[k].p > topOtherP) topOtherP = all[k].p;
      }

      float knownP = all[knownIdx].p;
      float target = Math.Max(knownP, topOtherP + EPSILON);
      float delta = target - knownP;

      bool nudged = delta > 0;
      float scale = nudged ? (1f / (1f + delta)) : 1f;
      List<PolicyEntry> entries = new List<PolicyEntry>(all.Count);
      for (int k = 0; k < all.Count; k++)
      {
        float p = all[k].p;
        if (k == knownIdx) p = target;
        entries.Add(new PolicyEntry
        {
          Uci = all[k].move.MoveStr(MGMoveNotationStyle.Coordinates),
          P = p * scale,
        });
      }
      return (entries, nudged, true);
    }


    /// <summary>
    /// Applies the rank-1-on-class rule to a WDL 3-vector: the theme-dominant
    /// class is lifted to (maxOtherClass + EPSILON) if not already the top, then
    /// the 3-vector is renormalised to sum to 1. Returns (w, d, l, nudged).
    /// </summary>
    static (float, float, float, bool) BuildRank1WDL(float origW, float origL, string themes, bool oppPov)
    {
      float origD = Math.Max(0f, 1f - origW - origL);
      int dominant = DetermineDominantClass(themes, oppPov);  // 0=W, 1=D, 2=L

      float[] wdl = new[] { origW, origD, origL };

      float topOther = 0f;
      for (int k = 0; k < 3; k++)
        if (k != dominant && wdl[k] > topOther) topOther = wdl[k];

      float target = Math.Max(wdl[dominant], topOther + EPSILON);
      float delta = target - wdl[dominant];
      bool nudged = delta > 0;

      if (nudged)
      {
        wdl[dominant] = target;
        float sum = 1f + delta;
        for (int k = 0; k < 3; k++) wdl[k] /= sum;
      }

      return (wdl[0], wdl[1], wdl[2], nudged);
    }


    /// <summary>
    /// Maps (themes, perspective) → dominant WDL class index (0=W, 1=D, 2=L).
    /// Mirrors the theme → WDL convention used in PuzzleFastLabeler.DeriveWDL.
    /// </summary>
    static int DetermineDominantClass(string themes, bool oppPov)
    {
      if (!string.IsNullOrEmpty(themes))
      {
        HashSet<string> themeSet = new HashSet<string>(
          themes.Split(' ', StringSplitOptions.RemoveEmptyEntries),
          StringComparer.OrdinalIgnoreCase);
        if (themeSet.Contains("equality")) return 1;  // D regardless of perspective
      }
      // crushing / mate / advantage / default
      return oppPov ? 2 : 0;  // L from opp POV, W from solver POV
    }


    /// <summary>
    /// Reconstructs a PositionWithHistory by replaying priorMoves from startFen.
    /// Mirrors PuzzleValueLabelerChildren.BuildPwh.
    /// </summary>
    static PositionWithHistory BuildPwh(string startFen, string priorMoves, Position currentPos)
    {
      if (string.IsNullOrWhiteSpace(startFen) || string.IsNullOrWhiteSpace(priorMoves))
        return new PositionWithHistory(currentPos);

      Position startPos;
      try { startPos = Position.FromFEN(startFen); }
      catch { return null; }

      string[] moves = priorMoves.Split(' ', StringSplitOptions.RemoveEmptyEntries);
      MGPosition mg = startPos.ToMGPosition;
      List<MGMove> mgMoves = new List<MGMove>(moves.Length);
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
  }
}
