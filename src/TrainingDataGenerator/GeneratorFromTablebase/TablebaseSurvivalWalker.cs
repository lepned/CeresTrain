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

using System.Collections.Generic;

using Ceres.Chess;
using Ceres.Chess.MoveGen;
using Ceres.Chess.NNEvaluators.LC0DLL;

#endregion

namespace CeresTrain.TrainingDataGenerator
{
  /// <summary>
  /// Synthesizes the continuation that tablebase-sampled training positions lack:
  /// walks K plies of tablebase-OPTIMAL play (DTZ-fastest win for the stronger side,
  /// DTZ-longest resistance for the weaker) so K-ply piece-survival labels can be
  /// computed exactly as for game corpora — but as PERFECT-PLAY ground truth rather
  /// than a description of one (possibly blundering) game continuation.
  /// See SURVIVAL_TARGET_SPEC.md section 8a.
  /// </summary>
  public static class TablebaseSurvivalWalker
  {
    /// <summary>
    /// Returns the optimal-play line [pos, pos after 1 ply, ...] up to horizonPlies
    /// plies for a DECISIVE position, or null when the position should be left
    /// unsupervised (caller emits an all-zero sidecar row):
    ///   - root probes as a draw, cursed win or blessed loss (under "any WDL-preserving
    ///     move is optimal" the piece fates are line-dependent, so no single ground truth),
    ///   - any probe fails mid-line (never risk a wrong fate label).
    /// The line is shorter than horizonPlies when the game ends first (mate/stalemate);
    /// pieces alive at the final position are labeled "survives" by the labeler,
    /// the same rule as game-end truncation in game corpora.
    /// </summary>
    public static List<Position> TryWalkOptimalLine(in Position pos,
                                                    ISyzygyEvaluatorEngine tbEvaluator,
                                                    int horizonPlies,
                                                    bool succeedIfIncompleteDTZInfo)
    {
      tbEvaluator.ProbeWDL(in pos, out SyzygyWDLScore score, out SyzygyProbeState state);
      if (state == SyzygyProbeState.Fail
       || (score != SyzygyWDLScore.WDLWin && score != SyzygyWDLScore.WDLLoss))
      {
        return null;
      }

      List<Position> line = new(horizonPlies + 1) { pos };
      Position cur = pos;
      for (int ply = 0; ply < horizonPlies; ply++)
      {
        if (cur.CalcTerminalStatus() != GameResult.Unknown)
        {
          break; // game over: remaining pieces survive (labeler's game-end rule).
        }

        // DTZ-optimal move for the side to move (fastest win / longest resistance).
        // returnOnlyWinningMoves: false so the losing side also receives its best move.
        MGMove best = tbEvaluator.CheckTablebaseBestNextMoveViaDTZ(in cur, out _, out _, out _,
                                                                   returnOnlyWinningMoves: false,
                                                                   succeedIfIncompleteDTZInfo: succeedIfIncompleteDTZInfo);
        if (best == default)
        {
          return null; // probe failure mid-line: unsupervise rather than mislabel.
        }

        MGPosition mg = cur.ToMGPosition;
        mg.MakeMove(best);
        cur = mg.ToPosition;
        line.Add(cur);
      }

      return line;
    }
  }
}
