#region License notice

/*
  This file is part of the CeresTrain project at https://github.com/dje-dev/cerestrain.
  Copyright (C) 2023- by David Elliott and the CeresTrain Authors.

  Ceres is free software under the terms of the GNU General Public License v3.0.
  You should have received a copy of the GNU General Public License
  along with CeresTrain. If not, see <http://www.gnu.org/licenses/>.
*/

#endregion

#region Using directives

using System;
using System.Collections.Generic;
using System.IO;

using Ceres.Chess.MoveGen;
using Ceres.Chess.NetEvaluation.Batch;
using Ceres.Chess.NNEvaluators;
using Ceres.Chess.NNEvaluators.Ceres;
using Ceres.Chess.NNEvaluators.Ceres.TPG;

using CeresTrain.TPG;

#endregion

namespace CeresTrain.Tasks
{
  /// <summary>
  /// Diagnostic: evaluates raw TPG-shard positions through a real Ceres inference
  /// backend (e.g. TensorRT native, the tournament path) and reports value1/value2
  /// quality against the value targets stored in the same TPG records
  /// (WDLResultNonDeblundered = game result z, WDLQ = search Q).
  ///
  /// Purpose: bisect value-head quality issues between the ONNX file itself
  /// (already validated via onnxruntime in Python, see F:/cout/value_probe.py)
  /// and the Ceres serving stack. Uses DoEvaluateNativeIntoBuffers with the
  /// records' own square bytes, so inputs are bit-identical to the Python probe.
  /// Temperatures forced to 1.0 and blending disabled so raw heads come through.
  /// </summary>
  public static class ValueProbeFromTPG
  {
    public static unsafe void Run(string tpgFileName, string onnxFileName, string deviceSpec, long numPositions)
    {
      const int BATCH_SIZE = 256;

      Console.WriteLine($"ValueProbeFromTPG: {tpgFileName} -> {onnxFileName} on {deviceSpec}, {numPositions} positions");

      NNEvaluator evaluator = NNEvaluator.FromSpecification("Ceres:" + onnxFileName, deviceSpec);
      evaluator.Options = new NNEvaluatorOptionsCeres()
      {
        FractionValueHead2 = 0f,       // no blending: W/L = raw value1, W2/L2 = raw value2
        ValueHead1Temperature = 1f,    // no temperature scaling on either head
        ValueHead2Temperature = 1f,
      };

      TPGFileReader reader = new TPGFileReader(tpgFileName, BATCH_SIZE);

      // Batch-consistency diagnostic (CERES_PROBE_BATCH_CHECK=1): evaluate the same
      // records singly (batch=1) and as one batch, compare per-position value outputs.
      // A nonzero divergence implicates batch-size-dependent numerics in the engine
      // (e.g. profile padding contamination), which would corrupt BestValueMove's
      // batched child rankings while leaving single-position (policy) evals intact.
      if (Environment.GetEnvironmentVariable("CERES_PROBE_BATCH_CHECK") == "1")
      {
        const int CHECK_N = 32;
        // N.B. converter sizes by the ARRAY length, so slice to exactly CHECK_N records.
        TPGRecord[] checkRecs = reader.NextBatch()[..CHECK_N];
        MGMoveList movesScratchBC = new MGMoveList();
        short[] lmiScratch = new short[TPGRecordMovesExtractor.NUM_MOVE_SLOTS_PER_REQUEST * BATCH_SIZE];
        HashSet<int>[] legalBC = new HashSet<int>[CHECK_N];
        for (int i = 0; i < CHECK_N; i++)
        {
          TPGRecordMovesExtractor.ExtractLegalMoveIndicesForIndex(checkRecs, movesScratchBC, lmiScratch, i);
          HashSet<int> set = new HashSet<int>();
          for (int m = 0; m < TPGRecordMovesExtractor.NUM_MOVE_SLOTS_PER_REQUEST; m++)
          {
            short idx = lmiScratch[i * TPGRecordMovesExtractor.NUM_MOVE_SLOTS_PER_REQUEST + m];
            if (idx > 0 || m == 0) set.Add(idx);
          }
          legalBC[i] = set;
        }

        // Single-position evaluations.
        float[] v1Single = new float[CHECK_N];
        float[] v2Single = new float[CHECK_N];
        for (int i = 0; i < CHECK_N; i++)
        {
          TPGRecord[] one = new TPGRecord[] { checkRecs[i] };
          int iCopy = i;
          IPositionEvaluationBatch b1 = evaluator.DoEvaluateNativeIntoBuffers(one, false, 1,
                                          (p, nn) => legalBC[iCopy].Contains(nn));
          v1Single[i] = b1.GetWin1P(0) - b1.GetLoss1P(0);
          v2Single[i] = b1.GetWin2P(0) - b1.GetLoss2P(0);
        }

        // Batched evaluation of the same records.
        IPositionEvaluationBatch bAll = evaluator.DoEvaluateNativeIntoBuffers(checkRecs, false, CHECK_N,
                                          (p, nn) => legalBC[p].Contains(nn));
        double maxD1 = 0, maxD2 = 0, sumD1 = 0;
        for (int i = 0; i < CHECK_N; i++)
        {
          double d1 = Math.Abs((bAll.GetWin1P(i) - bAll.GetLoss1P(i)) - v1Single[i]);
          double d2 = Math.Abs((bAll.GetWin2P(i) - bAll.GetLoss2P(i)) - v2Single[i]);
          maxD1 = Math.Max(maxD1, d1); maxD2 = Math.Max(maxD2, d2); sumD1 += d1;
        }
        Console.WriteLine($"BATCH-CONSISTENCY (n={CHECK_N}): value1 maxDelta {maxD1:F5} meanDelta {sumD1 / CHECK_N:F5}   value2 maxDelta {maxD2:F5}");
        Console.WriteLine("  (fp16 noise ~0.002; anything >0.02 indicates batch-dependent corruption)");
        return;
      }

      // Accumulators for metrics (double precision).
      long n = 0;
      double ce1z = 0, ce1q = 0, ce2z = 0, ce2q = 0, ent1 = 0, ent2 = 0;
      long acc1z = 0, acc2z = 0;
      List<double> ev1List = new(), ev2List = new(), evzList = new(), evqList = new();
      long nonFinite1 = 0, nonFinite2 = 0;

      using StreamWriter csv = new StreamWriter(Path.Combine(Path.GetTempPath(), "value_probe_csharp.csv"));
      csv.WriteLine("w1,d1,l1,w2,d2,l2,zw,zd,zl,qw,qd,ql");

      MGMoveList movesScratch = new MGMoveList();
      short[] legalMoveIndicesScratch = new short[TPGRecordMovesExtractor.NUM_MOVE_SLOTS_PER_REQUEST * BATCH_SIZE];

      while (n < numPositions)
      {
        TPGRecord[] recs = reader.NextBatch();
        if (recs == null)
        {
          break;
        }

        int thisCount = Math.Min(recs.Length, (int)(numPositions - n));

        // Build per-position legal-move NN index sets (required by the extraction callback).
        HashSet<int>[] legalSets = new HashSet<int>[thisCount];
        for (int i = 0; i < thisCount; i++)
        {
          TPGRecordMovesExtractor.ExtractLegalMoveIndicesForIndex(recs, movesScratch, legalMoveIndicesScratch, i);
          HashSet<int> set = new HashSet<int>();
          for (int m = 0; m < TPGRecordMovesExtractor.NUM_MOVE_SLOTS_PER_REQUEST; m++)
          {
            short idx = legalMoveIndicesScratch[i * TPGRecordMovesExtractor.NUM_MOVE_SLOTS_PER_REQUEST + m];
            if (idx > 0 || (m == 0)) // slot 0 may legitimately hold NN index 0
            {
              set.Add(idx);
            }
          }
          legalSets[i] = set;
        }

        Func<int, int, bool> posMoveIsLegal = (posIndex, nnIndex) => legalSets[posIndex].Contains(nnIndex);

        IPositionEvaluationBatch batch = evaluator.DoEvaluateNativeIntoBuffers(recs, false, thisCount, posMoveIsLegal);

        for (int i = 0; i < thisCount; i++)
        {
          float w1 = batch.GetWin1P(i);
          float l1 = batch.GetLoss1P(i);
          float d1 = 1f - w1 - l1;
          float w2 = batch.GetWin2P(i);
          float l2 = batch.GetLoss2P(i);
          float d2 = 1f - w2 - l2;

          TPGRecord rec = recs[i];
          float zw = rec.WDLResultNonDeblundered[0], zd = rec.WDLResultNonDeblundered[1], zl = rec.WDLResultNonDeblundered[2];
          float qw = rec.WDLQ[0], qd = rec.WDLQ[1], ql = rec.WDLQ[2];

          if (!float.IsFinite(w1) || !float.IsFinite(l1)) { nonFinite1++; continue; }
          if (!float.IsFinite(w2) || !float.IsFinite(l2)) { nonFinite2++; }

          csv.WriteLine($"{w1:F6},{d1:F6},{l1:F6},{w2:F6},{d2:F6},{l2:F6},{zw:F4},{zd:F4},{zl:F4},{qw:F6},{qd:F6},{ql:F6}");

          ce1z += CE(w1, d1, l1, zw, zd, zl);
          ce1q += CE(w1, d1, l1, qw, qd, ql);
          ce2z += CE(w2, d2, l2, zw, zd, zl);
          ce2q += CE(w2, d2, l2, qw, qd, ql);
          ent1 += Entropy(w1, d1, l1);
          ent2 += Entropy(w2, d2, l2);
          if (ArgMax(w1, d1, l1) == ArgMax(zw, zd, zl)) acc1z++;
          if (ArgMax(w2, d2, l2) == ArgMax(zw, zd, zl)) acc2z++;
          ev1List.Add(w1 - l1);
          ev2List.Add(w2 - l2);
          evzList.Add(zw - zl);
          evqList.Add(qw - ql);
        }

        n += thisCount;
        if (n % 2048 == 0)
        {
          Console.WriteLine($"  {n} positions...");
        }
      }

      double nn = ev1List.Count;
      Console.WriteLine();
      Console.WriteLine($"positions evaluated: {(long)nn}   non-finite v1: {nonFinite1}  v2: {nonFinite2}");
      Console.WriteLine($"  value1:  CEz {ce1z / nn:F4}  CEq {ce1q / nn:F4}  accz {100 * acc1z / nn:F2}%  " +
                        $"corrEVz {Corr(ev1List, evzList):F4}  corrEVq {Corr(ev1List, evqList):F4}  ent {ent1 / nn:F4}");
      Console.WriteLine($"  value2:  CEz {ce2z / nn:F4}  CEq {ce2q / nn:F4}  accz {100 * acc2z / nn:F2}%  " +
                        $"corrEVz {Corr(ev2List, evzList):F4}  corrEVq {Corr(ev2List, evqList):F4}  ent {ent2 / nn:F4}");
      Console.WriteLine($"  per-position CSV: {Path.Combine(Path.GetTempPath(), "value_probe_csharp.csv")}");
    }


    static double CE(float pw, float pd, float pl, float tw, float td, float tl)
      => -(tw * Math.Log(Math.Clamp(pw, 1e-9f, 1f))
         + td * Math.Log(Math.Clamp(pd, 1e-9f, 1f))
         + tl * Math.Log(Math.Clamp(pl, 1e-9f, 1f)));

    static double Entropy(float pw, float pd, float pl)
      => CE(pw, pd, pl, pw, pd, pl);

    static int ArgMax(float a, float b, float c) => a >= b ? (a >= c ? 0 : 2) : (b >= c ? 1 : 2);

    static double Corr(List<double> x, List<double> y)
    {
      double mx = 0, my = 0;
      for (int i = 0; i < x.Count; i++) { mx += x[i]; my += y[i]; }
      mx /= x.Count; my /= y.Count;
      double sxy = 0, sxx = 0, syy = 0;
      for (int i = 0; i < x.Count; i++)
      {
        double dx = x[i] - mx, dy = y[i] - my;
        sxy += dx * dy; sxx += dx * dx; syy += dy * dy;
      }
      return sxy / Math.Sqrt(sxx * syy);
    }
  }
}
