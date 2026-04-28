#region License notice

/*
  This file is part of the CeresTrain project at https://github.com/dje-dev/cerestrain.
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
  // ===================================================================================
  // WDL CONVENTION — READ THIS BEFORE EDITING
  // ===================================================================================
  // The action head of C3-768 (and equivalents) emits action[m] = (W, D, L) from the
  // CHILD-STM's POV — i.e., from the perspective of whoever is to move in the position
  // resulting from move m. For our use case the parent is solver-to-move, so child-STM
  // is OPP. Therefore, in every (W, D, L) value below:
  //
  //     W = opp's win probability in the child position
  //     D = opp's draw probability
  //     L = opp's loss probability  ← which equals solver's win probability
  //
  // This is verified empirically in test_c3_action_head.py (2026-04-27): on a position
  // labeled (W=0.903, D=0.087, L=0.009) from solver's POV, action[best_solution_move]
  // came out to (W=0.005, D=0.585, L=0.410) — i.e., LOW W and HIGH L from opp's POV.
  // The action head is NOT in solver's POV; intuitions like "the winning solver move
  // should have the highest W" are correct only for a Q-of-move-from-solver convention,
  // which this net does NOT use.
  //
  // Cross-record convention (Ceres training expectation):
  //   - OppDefence records describe opp-to-move positions; their (W, D, L) target is
  //     stored from OPP's POV (matches the record's own STM, matches what the value
  //     head outputs at inference for that position).
  //   - OAIS records likewise (opp-to-move after a blunder).
  // So action[m] slots directly into the record's WDL field with NO flip.
  //
  // Theme-aware "best for solver" axis (which axis solver should DOMINATE):
  //   - Winning theme (mate, crushing, advantage): solver's HIGHEST L  (opp loses most).
  //   - Equality theme: solver's HIGHEST D  (opp drawn most; solver holds the draw).
  //   - In both themes: solver's LOWEST W  (opp doesn't win — invariant of Lichess).
  //
  // The non-solver clamps mirror this: cap L (winning) or D (equality) from above,
  // floor W from below — preserving Lichess uniqueness regardless of action-head noise.
  // ===================================================================================

  /// <summary>
  /// Action-head-driven enrichment. For each Standard solver-to-move record:
  ///
  ///   - Runs an action-head-bearing teacher (e.g. C3-768-30-pre3-I8) forward on the parent
  ///     position via NNEvaluator, retrieving per-move WDL via <see cref="NNEvaluatorResult.ActionWDLForMove(MGMove)"/>.
  ///   - Applies a rank-1 nudge to the solver move's action vector so its L (from child STM POV)
  ///     dominates the cross-move L array by <c>opts.RankOneEpsilon</c>; renormalizes the solver
  ///     move's own (W,D,L) to remain a valid distribution.
  ///   - Emits an <see cref="PuzzlePositionKind.OppDefence"/> record for the solver-played child
  ///     with WDL = post-nudge solver action.
  ///   - Samples <c>opts.OAISSamplesPerParent</c> random non-solver legal moves and emits one
  ///     <see cref="PuzzlePositionKind.OppAfterInferiorSolver"/> record per sample, with
  ///     WDL = action[move_idx] taken AS-IS (no flip — V-of-child convention puts opp-to-move
  ///     POV directly into the WDL slot the OAIS record needs).
  ///
  /// Skips terminal child positions (no legal moves at the child) — matches the existing pattern in
  /// <see cref="PuzzleValueEnricher"/> and <see cref="PuzzleValueLabelerChildren"/>.
  ///
  /// Pass-through: Standard records are emitted unchanged. Other input record kinds are passed through
  /// unchanged too (this command does not re-label or re-emit OppDefence/OAIS records produced by an
  /// upstream enrichment stage; if you want a clean action-head dataset, point this at the raw
  /// labeled_2600plus.jsonl which contains only Standard records).
  ///
  /// Disagreement policy with Lichess: the rank-1 solver nudge guarantees the Lichess solution
  /// has the highest cross-move L by construction, even when the teacher disagrees on the parent's
  /// best move. All non-solver moves keep their raw action-head WDL — C3-768's per-move evaluation
  /// is strictly higher quality than the hardcoded theme-WDL it replaces, even on positions where
  /// the teacher misranks the solution.
  /// </summary>
  public static class PuzzleValueLabelerActionChildren
  {
    public sealed class Stats
    {
      public long InputRecords;
      public long StandardPassthrough;
      public long NonStandardPassthrough;
      public long OppDefenceEmitted;
      public long OAISEmitted;
      public long SkippedParseOrHistory;
      public long SkippedTerminalSolverChild;
      public long SkippedTerminalCounterfactualChild;
      public long OAISLClipped;
      public long OAISWFloored;
      public long OAISDClipped;
      public long EqualityParents;
      public long WinningParents;
      public double ElapsedSec;
    }


    public static Stats Run(PuzzleReplayOptions opts, string inputJsonlPath, string outputJsonlPath)
    {
      if (!File.Exists(inputJsonlPath))
        throw new FileNotFoundException("input labeled jsonl not found", inputJsonlPath);

      string netSpec = string.IsNullOrWhiteSpace(opts.ActionNetSpec) ? opts.NetSpec : opts.ActionNetSpec;
      string device = string.IsNullOrWhiteSpace(opts.ActionDevice) ? opts.Device : opts.ActionDevice;
      if (string.IsNullOrWhiteSpace(netSpec))
        throw new ArgumentException("ActionNetSpec (or NetSpec fallback) is required for enrich-action-head.");

      Console.WriteLine($"[action-enrich] Loading action-head teacher: {netSpec} on {device}");
      NNEvaluator evaluator = NNEvaluator.FromSpecification(netSpec, device);
      if (!evaluator.HasAction)
      {
        throw new InvalidOperationException(
          $"Configured net '{netSpec}' does not expose an action head (HasAction=false). " +
          $"enrich-action-head requires a net like C3-768-30-pre3 with the action output.");
      }

      int batchSize = Math.Max(64, opts.MineBatchSize);
      int kSamples = Math.Max(1, opts.OAISSamplesPerParent);
      float epsilon = opts.RankOneEpsilon;
      Random rng = new Random(42);   // fixed seed for reproducible OAIS sampling

      Console.WriteLine($"[action-enrich] Batch={batchSize}  K={kSamples}  ε={epsilon:F3}");

      Stats s = new Stats();
      Stopwatch sw = Stopwatch.StartNew();

      using StreamWriter writer = new StreamWriter(outputJsonlPath, append: false);

      foreach (IReadOnlyList<LabeledPuzzleRecord> batch in Batched(
                 JsonlIO.Read<LabeledPuzzleRecord>(inputJsonlPath), batchSize))
      {
        s.InputRecords += batch.Count;

        // For each Standard record we need the parent position (the FEN itself) evaluated.
        List<PositionWithHistory> parentPwhBatch = new();
        // Per-batch context: index into batch + materialized state we'll need post-eval.
        List<ParentContext> contexts = new();

        for (int i = 0; i < batch.Count; i++)
        {
          LabeledPuzzleRecord rec = batch[i];
          if (rec.Kind != PuzzlePositionKind.Standard) continue;
          if (string.IsNullOrWhiteSpace(rec.SolutionUci)) continue;

          Position pos;
          try { pos = Position.FromFEN(rec.FEN); }
          catch { s.SkippedParseOrHistory++; continue; }

          MGPosition mgParent = pos.ToMGPosition;
          MGMove solverMG;
          try { solverMG = MGMoveFromString.ParseMove(in mgParent, rec.SolutionUci); }
          catch { s.SkippedParseOrHistory++; continue; }
          if (solverMG == default) { s.SkippedParseOrHistory++; continue; }

          // Enumerate legal moves at the parent.
          MGMoveList legal = new MGMoveList();
          MGMoveGen.GenerateMoves(in mgParent, legal);
          if (legal.NumMovesUsed == 0) { s.SkippedParseOrHistory++; continue; }

          // Build PositionWithHistory for the PARENT (so the teacher sees real history).
          PositionWithHistory parentPwh = BuildPwh(rec.StartFen, rec.PriorUciMoves, pos);
          if (parentPwh == null) { s.SkippedParseOrHistory++; continue; }

          // Snapshot legal moves into a managed array (don't hold the MGMoveList struct beyond the loop).
          MGMove[] legalSnap = new MGMove[legal.NumMovesUsed];
          legal.MovesArray.AsSpan(0, legal.NumMovesUsed).CopyTo(legalSnap);

          parentPwhBatch.Add(parentPwh);
          contexts.Add(new ParentContext
          {
            BatchIndex = i,
            ParentMG = mgParent,
            SolverMG = solverMG,
            SolverUci = rec.SolutionUci,
            LegalMoves = legalSnap,
          });
        }

        NNEvaluatorResult[] parentResults = parentPwhBatch.Count > 0
          ? evaluator.Evaluate(parentPwhBatch, fillInMissingPlanes: true)
          : Array.Empty<NNEvaluatorResult>();

        // Emit pass-through + new records, preserving input order.
        int ctxCursor = 0;
        for (int i = 0; i < batch.Count; i++)
        {
          LabeledPuzzleRecord rec = batch[i];
          // Always pass through the input record first.
          JsonlIO.AppendLine(writer, rec);
          if (rec.Kind == PuzzlePositionKind.Standard) s.StandardPassthrough++;
          else s.NonStandardPassthrough++;

          if (ctxCursor < contexts.Count && contexts[ctxCursor].BatchIndex == i)
          {
            ParentContext ctx = contexts[ctxCursor];
            NNEvaluatorResult result = parentResults[ctxCursor];
            ctxCursor++;

            EmitForParent(rec, ctx, in result, kSamples, epsilon, rng, writer, s);
          }
        }

        if (s.InputRecords % 50_000 == 0)
        {
          writer.Flush();
          double pps = s.InputRecords / Math.Max(1, sw.Elapsed.TotalSeconds);
          Console.WriteLine($"[action-enrich] {s.InputRecords:N0} in, {s.OppDefenceEmitted:N0} OppDef out, " +
                            $"{s.OAISEmitted:N0} OAIS out, {pps:N0} pos/s");
        }
      }

      writer.Flush();
      sw.Stop();
      s.ElapsedSec = sw.Elapsed.TotalSeconds;
      Console.WriteLine();
      Console.WriteLine($"[action-enrich] Done.  Input={s.InputRecords:N0}  StdPass={s.StandardPassthrough:N0}  NonStdPass={s.NonStandardPassthrough:N0}");
      Console.WriteLine($"  OppDefence emitted          : {s.OppDefenceEmitted:N0}");
      Console.WriteLine($"  OAIS emitted                : {s.OAISEmitted:N0}");
      Console.WriteLine($"  SkippedParseOrHistory       : {s.SkippedParseOrHistory:N0}");
      Console.WriteLine($"  SkippedTerminalSolverChild  : {s.SkippedTerminalSolverChild:N0}");
      Console.WriteLine($"  SkippedTerminalCounterfact. : {s.SkippedTerminalCounterfactualChild:N0}");
      Console.WriteLine($"  Parents (winning themes)    : {s.WinningParents:N0}");
      Console.WriteLine($"  Parents (equality theme)    : {s.EqualityParents:N0}");
      double pct(long num) => s.OAISEmitted > 0 ? 100.0 * num / s.OAISEmitted : 0.0;
      Console.WriteLine($"  OAIS W-floored (disagree)   : {s.OAISWFloored:N0}  ({pct(s.OAISWFloored):F1}%)");
      Console.WriteLine($"  OAIS L-clipped (winning)    : {s.OAISLClipped:N0}  ({pct(s.OAISLClipped):F1}%)");
      Console.WriteLine($"  OAIS D-clipped (equality)   : {s.OAISDClipped:N0}  ({pct(s.OAISDClipped):F1}%)");
      Console.WriteLine($"  Elapsed                     : {s.ElapsedSec:F1}s ({s.ElapsedSec / 60:F1} min)");
      return s;
    }


    private struct ParentContext
    {
      public int BatchIndex;
      public MGPosition ParentMG;
      public MGMove SolverMG;
      public string SolverUci;
      public MGMove[] LegalMoves;
    }


    /// <summary>
    /// True if puzzle is "equality" theme (solver holds a draw, doesn't win).
    /// Matches the same classification used by PuzzleValueEnricher.DeriveOppAfterInferiorSolverWDL.
    /// </summary>
    static bool IsEqualityTheme(string themes)
    {
      if (string.IsNullOrEmpty(themes)) return false;
      foreach (string t in themes.Split(' ', StringSplitOptions.RemoveEmptyEntries))
        if (t.Equals("equality", StringComparison.OrdinalIgnoreCase)) return true;
      return false;
    }

    /// <summary>
    /// True if any theme token starts with "mate" (mate, mateIn1..N, etc.).
    /// </summary>
    static bool IsMateTheme(string themes)
    {
      if (string.IsNullOrEmpty(themes)) return false;
      foreach (string t in themes.Split(' ', StringSplitOptions.RemoveEmptyEntries))
        if (t.StartsWith("mate", StringComparison.OrdinalIgnoreCase)) return true;
      return false;
    }


    static void EmitForParent(LabeledPuzzleRecord rec, ParentContext ctx,
                              in NNEvaluatorResult result,
                              int kSamples, float epsilon, Random rng,
                              StreamWriter writer, Stats s)
    {
      bool isEquality = IsEqualityTheme(rec.Themes);
      if (isEquality) s.EqualityParents++; else s.WinningParents++;
      // Pull per-move action WDLs for ALL legal moves (including the solver).
      // ActionWDLForMove returns (w, d, l) already-softmaxed within (W,D,L).
      int n = ctx.LegalMoves.Length;
      (float w, float d, float l)[] actions = new (float, float, float)[n];
      int solverIdx = -1;
      for (int j = 0; j < n; j++)
      {
        MGMove m = ctx.LegalMoves[j];
        actions[j] = result.ActionWDLForMove(m);
        if (m == ctx.SolverMG) solverIdx = j;
      }
      if (solverIdx < 0)
      {
        // Solver move not found among legal moves — treat as parse failure.
        s.SkippedParseOrHistory++;
        return;
      }

      // ----- Solver target = theme floor (overrides action head) -----
      //
      // VALIDATED 2026-04-28: C3-768's action head is calibrated DIFFERENTLY than
      // its own value head. action[solver] systematically undersells L for
      // solver-played-winning moves (mean target L=0.34 vs orig val head L=0.72,
      // C3-768 val head L=0.75 across 30 mate samples — 83% anti-distill rate).
      //
      // Fix: discard action[solver]. Use theme-derived hardcoded floor as the
      // OppDefence target instead. These represent "the move you should have
      // found, would have been crushing" — boost confidence to align with
      // Lichess's theme guarantee, not action-head's miscalibration.
      //
      // Theme floors (opp's POV — solver loses → opp wins, etc.):
      //   mate              : (W=0.00, D=0.02, L=0.98)   forced mate; truth → L=1
      //   crushing/default  : (W=0.05, D=0.10, L=0.85)   above orig val head L=0.71
      //   advantage         : (W=0.05, D=0.10, L=0.85)   above orig val head L=0.76
      //   equality          : (W=0.20, D=0.60, L=0.20)   above orig val head D=0.46
      //
      // OAIS records continue to use action[non_solver_blunder] which IS
      // accurate (validated: target W=0.945 ≈ orig val W=0.965 on mate OAIS).
      // Action head is only inaccurate for SOLVER moves; non-solvers are fine.
      bool isMate = !isEquality && IsMateTheme(rec.Themes);
      float wNew, dNew, lNew;
      if (isEquality)        { wNew = 0.20f; dNew = 0.60f; lNew = 0.20f; }
      else if (isMate)       { wNew = 0.00f; dNew = 0.02f; lNew = 0.98f; }
      else /* winning */     { wNew = 0.05f; dNew = 0.10f; lNew = 0.85f; }
      actions[solverIdx] = (wNew, dNew, lNew);

      // The original rank-1 nudge logic is no longer used; OppDefence target
      // is now theme-floor regardless of action[solver]. The local vars
      // wNew/dNew/lNew (solver target) are still used below to set OAIS clamp
      // boundaries and to populate the OppDefence record's WDL fields.

      // (Original rank-1 logic preserved below as DEAD CODE for reference; the
      // unused branch is gated and will be removed once approach is confirmed.)
      #if false
      // ----- Solver rank-1 nudge (theme-aware) [DEPRECATED 2026-04-28] -----
      //
      // Lichess puzzle semantics by theme (matches PuzzleValueEnricher.DeriveOppAfterInferiorSolverWDL):
      //   - Winning themes (mate, crushing, advantage, others): solver's puzzle move is
      //     the unique winning continuation. Solver-played → opp losing. Non-solver → drawn.
      //     Cross-move ordering: solver's L is highest, solver's W is lowest.
      //   - Equality theme: solver's puzzle move is the unique drawing move. Solver-played
      //     → opp drawn. Non-solver → opp winning.
      //     Cross-move ordering: solver's W is lowest. L is weakly ordered (solver L
      //     moderate, non-solver L even lower) — magnitude gap is small, near-noise.
      //
      // Strategy: enforce two axes per theme. Solver should have:
      //
      //   Winning theme:  HIGHEST L (opp losing)  +  LOWEST W (opp winning)
      //   Equality theme: HIGHEST D (opp drawn)   +  LOWEST W (opp winning)
      //
      // The "best-for-solver" axis differs (L for winning, D for equality); the W
      // axis is invariant across themes (solver always minimizes opp's winning prob).
      // The third axis takes whatever residual mass keeps W+D+L=1.
      //
      // "Don't worsen" guards always apply: if solver's original value on an axis is
      // already on the correct side of the target, keep it. So well-behaved positions
      // get no nudge.
      float maxOtherL = 0f;
      float maxOtherD = 0f;
      float minOtherW = 1f;
      for (int j = 0; j < n; j++)
      {
        if (j == solverIdx) continue;
        if (actions[j].l > maxOtherL) maxOtherL = actions[j].l;
        if (actions[j].d > maxOtherD) maxOtherD = actions[j].d;
        if (actions[j].w < minOtherW) minOtherW = actions[j].w;
      }

      float origSolverW = actions[solverIdx].w;
      float origSolverD = actions[solverIdx].d;
      float origSolverL = actions[solverIdx].l;

      // W axis target (both themes): solver W ≤ minOtherW - ε.
      float wTarget = Math.Max(0.001f, minOtherW - epsilon);
      if (wTarget > origSolverW) wTarget = origSolverW;

      float wNew, dNew, lNew;
      if (isEquality)
      {
        // Equality theme: solver D ≥ maxOtherD + ε. L absorbs residual.
        float dTarget = Math.Min(0.999f, maxOtherD + epsilon);
        if (dTarget < origSolverD) dTarget = origSolverD;
        float sumWD = wTarget + dTarget;
        if (sumWD <= 1f)
        {
          wNew = wTarget;
          dNew = dTarget;
          lNew = 1f - wNew - dNew;
        }
        else
        {
          // Degenerate (rare): scale both down proportionally, L=0.
          float scale = 1f / sumWD;
          wNew = wTarget * scale;
          dNew = dTarget * scale;
          lNew = 0f;
        }
      }
      else
      {
        // Winning theme: solver L ≥ maxOtherL + ε. D absorbs residual.
        float lTarget = Math.Min(0.999f, maxOtherL + epsilon);
        if (lTarget < origSolverL) lTarget = origSolverL;
        float sumWL = wTarget + lTarget;
        if (sumWL <= 1f)
        {
          wNew = wTarget;
          lNew = lTarget;
          dNew = 1f - wNew - lNew;
        }
        else
        {
          // Degenerate: scale both down proportionally, D=0.
          float scale = 1f / sumWL;
          wNew = wTarget * scale;
          lNew = lTarget * scale;
          dNew = 0f;
        }
      }
      actions[solverIdx] = (wNew, dNew, lNew);
      #endif

      // ----- Emit OppDefence record for the solver-played child -----
      MGPosition mgSolverChild = ctx.ParentMG;
      mgSolverChild.MakeMove(ctx.SolverMG);
      MGMoveList solverChildLegal = new MGMoveList();
      MGMoveGen.GenerateMoves(in mgSolverChild, solverChildLegal);
      if (solverChildLegal.NumMovesUsed == 0)
      {
        // Mate-in-1 / stalemate child — TPG can't emit a record without legal moves.
        s.SkippedTerminalSolverChild++;
      }
      else
      {
        string priorForSolverChild = string.IsNullOrWhiteSpace(rec.PriorUciMoves)
          ? ctx.SolverUci
          : rec.PriorUciMoves + " " + ctx.SolverUci;
        LabeledPuzzleRecord oppDef = new LabeledPuzzleRecord
        {
          PuzzleId = rec.PuzzleId,
          FEN = mgSolverChild.ToPosition.FEN,
          SolutionUci = null,
          Rating = rec.Rating,
          Themes = rec.Themes,
          StartFen = rec.StartFen,
          PriorUciMoves = priorForSolverChild,
          Kind = PuzzlePositionKind.OppDefence,
          TeacherNodes = 1,
          TeacherTopUci = null,
          TeacherV = wNew - lNew,
          TeacherW = wNew,
          TeacherD = dNew,
          TeacherL = lNew,
          TeacherPolicy = null,
        };
        JsonlIO.AppendLine(writer, oppDef);
        s.OppDefenceEmitted++;
      }

      // ----- Emit K random OAIS records for non-solver legal moves -----
      // Build candidate list of non-solver indices.
      List<int> candidates = new List<int>(n - 1);
      for (int j = 0; j < n; j++)
      {
        if (j != solverIdx) candidates.Add(j);
      }
      // Sample without replacement: shuffle prefix up to min(K, candidates.Count).
      int take = Math.Min(kSamples, candidates.Count);
      for (int p = 0; p < take; p++)
      {
        int swapWith = p + rng.Next(candidates.Count - p);
        (candidates[p], candidates[swapWith]) = (candidates[swapWith], candidates[p]);
      }

      // ----- Non-solver clamps (theme-aware) -----
      // Three ordering invariants per OAIS record. The active axes depend on theme:
      //
      //   ALWAYS:    W-floor: action[non_solver].W ≥ solver.W + ε
      //              (After solver plays the puzzle move, opp does NOT win. After a
      //              blunder, opp must have at least slightly higher win chance.)
      //
      //   WINNING:   L-cap: action[non_solver].L ≤ solver.L − ε
      //              (Solver's puzzle move makes opp most-losing; blunders less so.)
      //
      //   EQUALITY:  D-cap: action[non_solver].D ≤ solver.D − ε
      //              (Solver's puzzle move keeps the draw — highest D from opp POV.
      //              Blunders give opp a winning position — lower D.)
      //
      // Each clamp is applied; non-active axes use sentinel values (1.0 → no-op).
      // After clamping, (W,D,L) is rebuilt: target W and (L or D) take their clamped
      // values; the third axis = 1 − sum, with proportional scale-down if W+other > 1.
      float wFloor = Math.Min(1.0f, wNew + epsilon);
      float lCap = isEquality ? 1.0f : Math.Max(0f, lNew - epsilon);
      float dCap = isEquality ? Math.Max(0f, dNew - epsilon) : 1.0f;

      for (int p = 0; p < take; p++)
      {
        int j = candidates[p];
        MGMove cf = ctx.LegalMoves[j];
        MGPosition mgBlunderChild = ctx.ParentMG;
        mgBlunderChild.MakeMove(cf);
        MGMoveList bcLegal = new MGMoveList();
        MGMoveGen.GenerateMoves(in mgBlunderChild, bcLegal);
        if (bcLegal.NumMovesUsed == 0)
        {
          s.SkippedTerminalCounterfactualChild++;
          continue;
        }

        string cfUci = cf.MoveStr(MGMoveNotationStyle.Coordinates);
        string priorForBlunderChild = string.IsNullOrWhiteSpace(rec.PriorUciMoves)
          ? cfUci
          : rec.PriorUciMoves + " " + cfUci;

        (float w, float d, float l) act = actions[j];

        // Apply theme-aware clamps. Each axis independently: W from below, plus
        // either L from above (winning) or D from above (equality). After clamping,
        // resolve the third (free) axis from the constraint W+D+L=1.
        float wAfterFloor = (act.w < wFloor) ? wFloor : act.w;
        float lAfterCap = (act.l > lCap) ? lCap : act.l;
        float dAfterCap = (act.d > dCap) ? dCap : act.d;
        bool wFlooredNow = (wAfterFloor != act.w);
        bool lClippedNow = (lAfterCap != act.l);
        bool dClippedNow = (dAfterCap != act.d);

        if (wFlooredNow || lClippedNow || dClippedNow)
        {
          float wResult, dResult, lResult;
          if (isEquality)
          {
            // Equality: W and D are pinned; L = 1 - W - D (or scale if degenerate).
            float sumWD = wAfterFloor + dAfterCap;
            if (sumWD <= 1f)
            {
              wResult = wAfterFloor;
              dResult = dAfterCap;
              lResult = 1f - wResult - dResult;
            }
            else
            {
              float scale = 1f / sumWD;
              wResult = wAfterFloor * scale;
              dResult = dAfterCap * scale;
              lResult = 0f;
            }
          }
          else
          {
            // Winning: W and L are pinned; D = 1 - W - L.
            float sumWL = wAfterFloor + lAfterCap;
            if (sumWL <= 1f)
            {
              wResult = wAfterFloor;
              lResult = lAfterCap;
              dResult = 1f - wResult - lResult;
            }
            else
            {
              float scale = 1f / sumWL;
              wResult = wAfterFloor * scale;
              lResult = lAfterCap * scale;
              dResult = 0f;
            }
          }
          act = (wResult, dResult, lResult);
          if (wFlooredNow) s.OAISWFloored++;
          if (lClippedNow) s.OAISLClipped++;
          if (dClippedNow) s.OAISDClipped++;
        }

        // Action-head WDL is from child STM's POV (V-of-child convention,
        // verified 2026-04-27); slots directly into OAIS record's WDL target.
        // No flip needed.
        LabeledPuzzleRecord oais = new LabeledPuzzleRecord
        {
          PuzzleId = rec.PuzzleId,
          FEN = mgBlunderChild.ToPosition.FEN,
          SolutionUci = null,
          Rating = rec.Rating,
          Themes = rec.Themes,
          StartFen = rec.StartFen,
          PriorUciMoves = priorForBlunderChild,
          Kind = PuzzlePositionKind.OppAfterInferiorSolver,
          TeacherNodes = 1,
          TeacherTopUci = null,
          TeacherV = act.w - act.l,
          TeacherW = act.w,
          TeacherD = act.d,
          TeacherL = act.l,
          TeacherPolicy = null,
        };
        JsonlIO.AppendLine(writer, oais);
        s.OAISEmitted++;
      }
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
