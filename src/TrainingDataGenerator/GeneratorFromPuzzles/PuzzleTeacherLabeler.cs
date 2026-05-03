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
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Threading;
using System.Threading.Tasks;

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
  /// For each HardPuzzleRecord, runs an MCTS search at nodes=N (using the older
  /// GameEngineCeresInProcess from Ceres.MCTS, same engine class used by Ceres's own
  /// BatchAnalyzer for bulk analysis workloads), and emits a LabeledPuzzleRecord when
  /// the teacher's top move agrees with the Lichess solution — or when the teacher finds
  /// a near-certain win (Q >= 0.98, covers proven mates and crushing positions).
  /// Disagreements go to rejected.jsonl alongside, with the reason.
  ///
  /// Parallelism: N worker threads, each with its own GameEngineCeresInProcess bound
  /// to the same NNEvaluatorDef. The underlying NN evaluator is shared across workers
  /// via the shared-name mechanism in NNEvaluatorDef.
  /// </summary>
  public static class PuzzleTeacherLabeler
  {
    /// <summary>
    /// Stream-label every record in <see cref="PuzzleReplayOptions.HardJsonlPath"/>.
    /// Writes LabeledJsonlPath and RejectedJsonlPath.
    /// </summary>
    public static LabelStats Run(PuzzleReplayOptions opts)
    {
      opts.Validate();
      if (!opts.SkipMining && !File.Exists(opts.HardJsonlPath))
        throw new FileNotFoundException("hard.jsonl not found — run mine stage first, or set SkipMining=true to label all puzzles directly from the Lichess CSV", opts.HardJsonlPath);

      // Resume-from-checkpoint: if existing labeled/rejected files exist, build a set of
      // already-processed (PuzzleId, FEN) keys so we skip them when streaming input.
      // Output files open in append mode when resuming.
      HashSet<string> alreadyDone = new HashSet<string>();
      bool appendMode = false;
      if (opts.ResumeFromCheckpoint)
      {
        if (File.Exists(opts.LabeledJsonlPath))
        {
          foreach (LabeledPuzzleRecord r in JsonlIO.Read<LabeledPuzzleRecord>(opts.LabeledJsonlPath))
            alreadyDone.Add((r.PuzzleId ?? "") + "|" + (r.FEN ?? ""));
        }
        if (File.Exists(opts.RejectedJsonlPath))
        {
          foreach (RejectedPuzzleRecord r in JsonlIO.Read<RejectedPuzzleRecord>(opts.RejectedJsonlPath))
            alreadyDone.Add((r.PuzzleId ?? "") + "|" + (r.FEN ?? ""));
        }
        if (alreadyDone.Count > 0)
        {
          appendMode = true;
          Console.WriteLine($"[label] Resume: skipping {alreadyDone.Count:N0} already-processed records.");
        }
      }

      // Share one evaluator across workers via SharedName (safe for ORT path; unsafe for TRT).
      string deviceSpec = opts.Device.Contains('=') ? opts.Device : opts.Device + "=PuzzleTeacher";
      NNEvaluatorDef evalDef = NNEvaluatorDefFactory.FromSpecification(opts.NetSpec, deviceSpec);

      Ceres.Chess.NNEvaluators.NNEvaluator warmupEval = Ceres.Chess.NNEvaluators.NNEvaluatorFactory.BuildEvaluator(evalDef);
      Console.WriteLine($"[label] Warming up shared evaluator (MaxBatchSize={warmupEval.MaxBatchSize}) under name 'PuzzleTeacher'...");
      Ceres.Chess.NNEvaluators.NNEvaluatorBenchmark.Warmup(warmupEval, warmupEval.MaxBatchSize);
      Console.WriteLine("[label] Warmup complete.");

      // Nothing MCTS-specific needed; search limit is passed per-call.

      long processed = 0, accepted = 0, rejected = 0;
      Stopwatch sw = Stopwatch.StartNew();

      object acceptLock = new();
      object rejectLock = new();
      using StreamWriter acceptWriter = new StreamWriter(opts.LabeledJsonlPath, append: appendMode);
      using StreamWriter rejectWriter = new StreamWriter(opts.RejectedJsonlPath, append: appendMode);

      BlockingCollection<HardPuzzleRecord> queue = new BlockingCollection<HardPuzzleRecord>(
        boundedCapacity: opts.TeacherWorkerThreads * 4);

      // Determine input stream. Three modes:
      //   (a) SkipMining=true:  build records on-the-fly from Lichess CSV (label-all).
      //   (b) MaxRecordsToLabel > 0 & !SkipMining: reservoir-sample from hard.jsonl.
      //   (c) default: stream hard.jsonl as-is.
      IEnumerable<HardPuzzleRecord> InputSource()
      {
        if (opts.SkipMining)
        {
          foreach (PuzzleSolvePosition p in LichessPuzzleReader.Read(
                     opts.LichessCsvPath, opts.MinRating, opts.MaxRating,
                     opts.ThemeIncludeAny, opts.ThemeExcludeAny, opts.MaxPuzzlesToRead))
          {
            yield return new HardPuzzleRecord
            {
              PuzzleId = p.PuzzleId,
              FEN = p.FEN,
              SolutionUci = p.SolutionUci,
              SolveStepIndex = p.SolveStepIndex,
              Rating = p.Rating,
              Themes = p.Themes,
              // Propagate real history so the teacher's MCTS search runs with the
              // same history planes EB's UCI puzzle harness uses (`position fen ... moves ...`).
              // Without this, the search saw bare-FEN-with-replicated-history-planes
              // and produced different decisions on continuation positions (ply 2+),
              // causing massive false-rejection rates.
              StartFen = p.StartFen,
              PriorUciMoves = p.PriorUciMoves,
              StudentTopUci = "",
              StudentV = 0f,
            };
          }
          yield break;
        }
        foreach (HardPuzzleRecord r in JsonlIO.Read<HardPuzzleRecord>(opts.HardJsonlPath))
          yield return r;
      }

      List<HardPuzzleRecord> preloaded = null;
      if (opts.MaxRecordsToLabel > 0)
      {
        List<HardPuzzleRecord> all = new List<HardPuzzleRecord>();
        foreach (HardPuzzleRecord rec in InputSource()) all.Add(rec);
        if (all.Count <= opts.MaxRecordsToLabel)
        {
          preloaded = all;
        }
        else
        {
          Random rng = new Random(opts.LabelSubsampleSeed);
          for (int i = all.Count - 1; i > 0; i--)
          {
            int j = rng.Next(i + 1);
            (all[i], all[j]) = (all[j], all[i]);
          }
          preloaded = all.GetRange(0, opts.MaxRecordsToLabel);
        }
        Console.WriteLine($"[label] Sub-sampled {preloaded.Count:N0} records from {all.Count:N0} total (seed={opts.LabelSubsampleSeed}).");
      }

      long skippedAlreadyDone = 0;
      Task producer = Task.Run(() =>
      {
        try
        {
          IEnumerable<HardPuzzleRecord> source = preloaded ?? InputSource();
          foreach (HardPuzzleRecord rec in source)
          {
            if (alreadyDone.Count > 0)
            {
              string key = (rec.PuzzleId ?? "") + "|" + (rec.FEN ?? "");
              if (alreadyDone.Contains(key))
              {
                Interlocked.Increment(ref skippedAlreadyDone);
                continue;
              }
            }
            queue.Add(rec);
          }
        }
        finally { queue.CompleteAdding(); }
      });

      Task[] workers = new Task[opts.TeacherWorkerThreads];
      for (int w = 0; w < workers.Length; w++)
      {
        int workerID = w;
        workers[w] = Task.Run(() =>
        {
          // Use MCGS engine (not legacy MCTS GameEngineCeresInProcess) to match the
          // search algorithm + PathMode PositionEquivalence that Ceres.exe uses by
          // default (verified via direct UCI test on 5 rejected positions: in-process
          // MCTS labeler produced the same wrong move on all 5 that Ceres.exe MCGS
          // got right). Without this, the labeler accepts ~22% of positions where
          // Ceres.exe accepts ~87% — a >4× false-rejection rate on the same positions.
          GameEngineCeresMCGSInProcess engine = new GameEngineCeresMCGSInProcess(
            id: "PuzzleTeacher_" + workerID,
            evaluatorDef: evalDef,
            moveImmediateIfOnlyOneMove: false);

          foreach (HardPuzzleRecord rec in queue.GetConsumingEnumerable())
          {
            try
            {
              LabelOne(engine, rec, opts.TeacherNodes,
                       out LabeledPuzzleRecord labeled, out string rejectReason, out string teacherTopUci);

              Interlocked.Increment(ref processed);

              if (labeled != null)
              {
                Interlocked.Increment(ref accepted);
                lock (acceptLock) JsonlIO.AppendLine(acceptWriter, labeled);
              }
              else
              {
                Interlocked.Increment(ref rejected);
                lock (rejectLock)
                {
                  JsonlIO.AppendLine(rejectWriter, new RejectedPuzzleRecord
                  {
                    PuzzleId = rec.PuzzleId,
                    FEN = rec.FEN,
                    SolutionUci = rec.SolutionUci,
                    Reason = rejectReason,
                    TeacherTopUci = teacherTopUci,
                  });
                }
              }
            }
            catch (Exception e)
            {
              Interlocked.Increment(ref rejected);
              lock (rejectLock)
              {
                JsonlIO.AppendLine(rejectWriter, new RejectedPuzzleRecord
                {
                  PuzzleId = rec.PuzzleId,
                  FEN = rec.FEN,
                  SolutionUci = rec.SolutionUci,
                  Reason = "exception: " + e.GetType().Name + ": " + e.Message,
                });
              }
            }

            long p = Interlocked.Read(ref processed);
            if (p % 500 == 0)
            {
              lock (acceptLock) acceptWriter.Flush();
              lock (rejectLock) rejectWriter.Flush();
              double posPerSec = p / Math.Max(1, sw.Elapsed.TotalSeconds);
              string skipSuffix = skippedAlreadyDone > 0 ? $" (+{skippedAlreadyDone:N0} skipped from resume)" : "";
              Console.WriteLine($"[label] {p:N0} processed{skipSuffix}, {accepted:N0} accepted, {rejected:N0} rejected, " +
                                $"{posPerSec:N1} pos/s");
            }
          }

          engine.Dispose();
        });
      }

      Task.WaitAll(workers);
      producer.Wait();
      acceptWriter.Flush();
      rejectWriter.Flush();
      sw.Stop();

      LabelStats stats = new LabelStats
      {
        Processed = processed,
        Accepted = accepted,
        Rejected = rejected,
        ElapsedSec = sw.Elapsed.TotalSeconds,
      };
      Console.WriteLine($"[label] Done. Processed={stats.Processed:N0} Accepted={stats.Accepted:N0} " +
                        $"Rejected={stats.Rejected:N0} RejectRate={(stats.Processed == 0 ? 0 : 100.0 * stats.Rejected / stats.Processed):F1}% " +
                        $"Elapsed={stats.ElapsedSec:F1}s");
      return stats;
    }


    static void LabelOne(GameEngineCeresMCGSInProcess engine, HardPuzzleRecord rec, int nodes,
                         out LabeledPuzzleRecord labeled, out string rejectReason, out string teacherTopUci)
    {
      labeled = null;
      rejectReason = null;
      teacherTopUci = null;

      Position pos;
      try { pos = Position.FromFEN(rec.FEN); }
      catch { rejectReason = "invalid_fen"; return; }

      if (pos.CalcTerminalStatus() != GameResult.Unknown)
      {
        rejectReason = "terminal_position";
        return;
      }

      engine.ResetGame();

      // Build PositionWithHistory using the real move sequence (StartFen + PriorUciMoves)
      // when available. Without this, the teacher's NN sees history planes filled with
      // current-position-replicas (or zeros), producing different policy/value evaluations
      // than EB's UCI harness (`position fen ... moves ...`) which uses real history.
      // The TPG generator already does this reconstruction (PuzzleToTPGGenerator.cs:248);
      // the labeler must too for consistency. Falls back to bare-FEN if PriorUciMoves missing
      // (legacy hard.jsonl files from before this field was populated).
      PositionWithHistory pwh;
      if (!string.IsNullOrEmpty(rec.StartFen) && !string.IsNullOrEmpty(rec.PriorUciMoves))
      {
        try
        {
          Position startPos = Position.FromFEN(rec.StartFen);
          pwh = new PositionWithHistory(startPos);
          foreach (string moveUci in rec.PriorUciMoves.Split(' ', StringSplitOptions.RemoveEmptyEntries))
          {
            pwh.AppendMove(moveUci);
          }
        }
        catch
        {
          // Fallback if history reconstruction fails for any reason.
          pwh = new PositionWithHistory(pos);
        }
      }
      else
      {
        pwh = new PositionWithHistory(pos);
      }

      GameEngineSearchResultCeresMCGS result;
      try
      {
        result = engine.SearchCeres(pwh, SearchLimit.NodesPerMove(nodes));
      }
      catch (Exception e)
      {
        rejectReason = "search_exception: " + e.GetType().Name;
        return;
      }

      if (result?.BestMoveInfo == null)
      {
        rejectReason = "no_best_move";
        return;
      }

      MGPosition mgPos = pos.ToMGPosition;
      MGMove teacherTopMG = result.BestMoveMG;
      teacherTopUci = teacherTopMG == default ? "" : teacherTopMG.MoveStr(MGMoveNotationStyle.Coordinates);

      MGMove solutionMG;
      try { solutionMG = MGMoveFromString.ParseMove(in mgPos, rec.SolutionUci); }
      catch { rejectReason = "solution_parse_fail"; return; }

      if (teacherTopMG == default)
      {
        rejectReason = "no_teacher_top_move";
        return;
      }

      // Forced-win exception: if the teacher's best line scores clearly winning, accept
      // even when the top move differs from Lichess. Lichess names ONE winning move per
      // puzzle but multiple winning moves often exist (especially mate-in-N where any of
      // several pieces deliver mate). Threshold lowered from 0.98 to 0.70 (2026-04-30):
      // 0.98 was rejecting many valid alternative-winning-move cases because shallow
      // search (e.g. 400 visits) doesn't drive Q to near-1.0 even on objectively winning
      // positions. 0.70 still requires the teacher to be confidently winning.
      const float MATE_ACCEPT_Q_THRESHOLD = 0.70f;
      bool teacherFoundWinningLine = result.BestMoveInfo.QOfBest >= MATE_ACCEPT_Q_THRESHOLD;

      if (!(teacherTopMG == solutionMG) && !teacherFoundWinningLine)
      {
        rejectReason = "teacher_disagreement";
        return;
      }

      // MCGS root node accessed via Search.SearchRootNode (a GNode).
      // CRITICAL: use the SEARCH-BACKED W/L/D properties (derived from Q averaged
      // over all visits), NOT the static NN-eval outputs (WinP/LossP/DrawP).
      // Pre-2026-04-30 bug: we used root.WinP / root.LossP, which threw away the
      // 200 nodes of search and saved C3's raw value-head guess. Verified by
      // user's hand-test: search-backed Q=0.85 vs static V=0.07 on a winning
      // tactical position — we were saving 0.07.
      // root.W = (Q + 1 - D) / 2  with search-backed Q and D.
      // root.L = (1 - D - Q) / 2.
      GNode root = result.Search.SearchRootNode;

      float rootW = root.W;
      float rootL = root.L;
      float rootD = (float)root.D;

      // Clamp + renormalize: GNode.W / GNode.L are derived as (Q+1-D)/2 and (1-D-Q)/2
      // from search-backed Q and D. Mathematically these are >=0, but in practice
      // floating-point rounding from MCGS aggregation across multiple parents in graph
      // mode can produce small negative values on confident-winning positions
      // (verified empirically: 52.7% of records had L<0 in 2600-2700 phase-1). Clamp
      // to [0,1] and renormalize the WDL distribution to a valid probability triple
      // before saving.
      if (rootW < 0f) rootW = 0f;
      if (rootL < 0f) rootL = 0f;
      if (rootD < 0f) rootD = 0f;
      float wdlSum = rootW + rootL + rootD;
      if (wdlSum > 0f)
      {
        rootW /= wdlSum;
        rootL /= wdlSum;
        rootD /= wdlSum;
      }
      float rootV = rootW - rootL;

      // Enumerate root's expanded child edges to build visit-count policy distribution.
      // MCGS uses graph edges (GEdge) instead of the MCTS ChildrenExpanded enumerable.
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
      }
      if (totalN <= 0)
      {
        rejectReason = "no_visits";
        return;
      }
      foreach (GEdge edge in sortedEdges)
      {
        if (!edge.IsExpanded) continue;
        long childN = edge.ChildNode.N;
        if (childN <= 0) continue;
        // The block below remains in the loop body (was structured around `child` MCTSNode);
        // we extract move-uci from the edge and visit fraction from childN.
        MGMove moveMG = edge.MoveMG;
        policy.Add(new PolicyEntry
        {
          Uci = moveMG == default ? "" : moveMG.MoveStr(MGMoveNotationStyle.Coordinates),
          P = (float)childN / totalN,
        });
      }

      // Rank-1 nudge: ensure the Lichess solution is the unique top of the policy
      // distribution by a small epsilon margin. Required because MCGS visit-count
      // distributions can produce near-ties (e.g., 0.503 vs 0.497) where the wrong
      // move accidentally gets more visits even when MCGS's chosen best-move (via
      // Q+N criterion) matches Lichess. Also handles immediate-no-search-mate
      // short-circuit cases where the solution may not appear in the visit-count
      // distribution at all (TeacherTopUci=solution but visits went to other moves
      // during root expansion). Without this nudge, the training target would
      // teach the network to play the wrong move on those positions.
      // Convention matches PuzzleSoftLabeler.RankOneEpsilon default (0.03).
      const float RANK_ONE_EPSILON = 0.03f;
      int solutionIdx = -1;
      float maxNonSolutionP = 0f;
      for (int i = 0; i < policy.Count; i++)
      {
        if (policy[i].Uci == rec.SolutionUci)
        {
          solutionIdx = i;
        }
        else if (policy[i].P > maxNonSolutionP)
        {
          maxNonSolutionP = policy[i].P;
        }
      }
      bool needsRenormalize = false;
      if (solutionIdx < 0)
      {
        // Solution missing from distribution (immediate-mate short-circuit). Add it
        // with dominating probability so the training target points at the right move.
        policy.Add(new PolicyEntry { Uci = rec.SolutionUci, P = maxNonSolutionP + RANK_ONE_EPSILON });
        needsRenormalize = true;
      }
      else if (policy[solutionIdx].P < maxNonSolutionP + RANK_ONE_EPSILON)
      {
        PolicyEntry e = policy[solutionIdx];
        policy[solutionIdx] = new PolicyEntry { Uci = e.Uci, P = maxNonSolutionP + RANK_ONE_EPSILON };
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

      labeled = new LabeledPuzzleRecord
      {
        PuzzleId = rec.PuzzleId,
        FEN = rec.FEN,
        SolutionUci = rec.SolutionUci,
        Rating = rec.Rating,
        Themes = rec.Themes,
        StartFen = rec.StartFen,
        PriorUciMoves = rec.PriorUciMoves,
        TeacherNodes = nodes,
        TeacherTopUci = teacherTopUci,
        TeacherV = rootV,
        TeacherW = rootW,
        TeacherD = rootD,
        TeacherL = rootL,
        TeacherPolicy = policy,
      };
    }


    public readonly record struct LabelStats(long Processed, long Accepted, long Rejected, double ElapsedSec);
  }
}
