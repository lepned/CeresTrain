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

using Ceres.MCTS.GameEngines;
using Ceres.MCTS.MTCSNodes;

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
          GameEngineCeresInProcess engine = new GameEngineCeresInProcess(
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


    static void LabelOne(GameEngineCeresInProcess engine, HardPuzzleRecord rec, int nodes,
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
      PositionWithHistory pwh = new PositionWithHistory(pos);

      GameEngineSearchResultCeres result;
      try
      {
        result = engine.SearchCeres(pwh, SearchLimit.NodesPerMove(nodes));
      }
      catch (Exception e)
      {
        rejectReason = "search_exception: " + e.GetType().Name;
        return;
      }

      if (result?.BestMove == null)
      {
        rejectReason = "no_best_move";
        return;
      }

      MGPosition mgPos = pos.ToMGPosition;
      MGMove teacherTopMG = result.BestMove.BestMove;
      teacherTopUci = teacherTopMG == default ? "" : teacherTopMG.MoveStr(MGMoveNotationStyle.Coordinates);

      MGMove solutionMG;
      try { solutionMG = MGMoveFromString.ParseMove(in mgPos, rec.SolutionUci); }
      catch { rejectReason = "solution_parse_fail"; return; }

      if (teacherTopMG == default)
      {
        rejectReason = "no_teacher_top_move";
        return;
      }

      // Mate/forced-win exception: if the teacher's best line scores near +1 (proven mate
      // or crushing winning position), accept even when the top move differs from Lichess.
      // Mate-in-N positions commonly have multiple winning moves; Lichess names one.
      const float MATE_ACCEPT_Q_THRESHOLD = 0.98f;
      bool teacherFoundWinningLine = result.BestMove.QOfBest >= MATE_ACCEPT_Q_THRESHOLD;

      if (!(teacherTopMG == solutionMG) && !teacherFoundWinningLine)
      {
        rejectReason = "teacher_disagreement";
        return;
      }

      MCTSNode root = result.Search.Manager.Root;

      float rootW = root.WinP;
      float rootL = root.LossP;
      float rootD = Math.Max(0f, 1f - rootW - rootL);
      float rootV = rootW - rootL;

      List<PolicyEntry> policy = new List<PolicyEntry>();
      long totalN = 0;
      foreach (MCTSNode child in root.ChildrenExpanded) totalN += child.N;
      if (totalN <= 0)
      {
        rejectReason = "no_visits";
        return;
      }
      foreach (MCTSNode child in root.ChildrenExpanded)
      {
        if (child.N <= 0) continue;
        MGMove moveMG = child.Annotation.PriorMoveMG;
        policy.Add(new PolicyEntry
        {
          Uci = moveMG.MoveStr(MGMoveNotationStyle.Coordinates),
          P = (float)child.N / totalN,
        });
      }

      labeled = new LabeledPuzzleRecord
      {
        PuzzleId = rec.PuzzleId,
        FEN = rec.FEN,
        SolutionUci = rec.SolutionUci,
        Rating = rec.Rating,
        Themes = rec.Themes,
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
