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

using Ceres.Chess;
using Ceres.Chess.EncodedPositions;

#endregion

namespace CeresTrain.TPG.TPGGenerator
{
  /// <summary>
  /// Computes K-ply piece-survival target labels for every position of a game
  /// (see SURVIVAL_TARGET_SPEC.md at repo root).
  ///
  /// For each position and each REAL-board square (A1=0 .. H8=63), the label byte is:
  ///   0            square is empty at this position (masked from the training loss)
  ///   d in 1..K    the piece standing here is CAPTURED exactly d plies later
  ///   K+1          the piece survives the horizon (or survives to game end)
  ///
  /// Piece identity is tracked by diffing consecutive positions (no move decoding),
  /// which is robust across promotions, castling (including Chess960/KTR overlaps
  /// where a piece lands on another mover's origin square), and en passant:
  ///   - a square whose occupant at t is an ENEMY of the side to move and changes
  ///     by t+1 is a capture (arrival capture, or en-passant vacation);
  ///   - mover pieces that left their square are matched to newly-occupied/changed
  ///     squares by piece type (promotion falls back to the vacated pawn).
  /// </summary>
  public static class SurvivalLabeler
  {
    /// <summary>
    /// Returns [numPositions][64] label bytes in REAL-board square indexing
    /// (A1=0 .. H8=63, BottomToTopLeftToRight). Callers writing TPG records must
    /// remap to record slots (slot = sq if White to move, else 63 - sq).
    /// </summary>
    public static byte[][] ComputeGameSurvival(in EncodedTrainingPositionGame game, int horizonPlies)
    {
      int numPos = game.NumPositions;
      Position[] positions = new Position[numPos];
      for (int t = 0; t < numPos; t++)
      {
        positions[t] = game.PositionAtIndex(t).FinalPosition;
      }
      return ComputeSurvivalForLine(positions, horizonPlies);
    }


    /// <summary>
    /// Same computation for an arbitrary sequence of consecutive positions (e.g. a puzzle
    /// solution line: current position followed by the forced continuation). A sequence of
    /// length 1 yields a row where every piece "survives" (no continuation to observe).
    /// </summary>
    public static byte[][] ComputeSurvivalForLine(IReadOnlyList<Position> positionsList, int horizonPlies)
    {
      if (horizonPlies <= 0 || horizonPlies > 254)
      {
        throw new ArgumentException("horizonPlies must be in 1..254", nameof(horizonPlies));
      }

      int numPos = positionsList.Count;
      Position[] positions = positionsList as Position[] ?? new Position[numPos];
      if (!ReferenceEquals(positions, positionsList))
      {
        for (int t = 0; t < numPos; t++)
        {
          positions[t] = positionsList[t];
        }
      }

      // ids[t][sq] = piece id occupying real-board square sq at position t (-1 = empty).
      int[][] ids = new int[numPos][];
      List<int> deathPly = new(48); // deathPly[id] = first position index at which the piece is absent (captured); int.MaxValue = never

      ids[0] = new int[64];
      for (int s = 0; s < 64; s++)
      {
        if (PieceAt(in positions[0], s).Type == PieceType.None)
        {
          ids[0][s] = -1;
        }
        else
        {
          ids[0][s] = deathPly.Count;
          deathPly.Add(int.MaxValue);
        }
      }

      List<int> vacatedMover = new(4);
      List<int> arrived = new(4);

      for (int t = 0; t < numPos - 1; t++)
      {
        ref readonly Position before = ref positions[t];
        ref readonly Position after = ref positions[t + 1];
        SideType mover = before.SideToMove;

        int[] cur = ids[t];
        int[] nxt = new int[64];
        Array.Copy(cur, nxt, 64);

        vacatedMover.Clear();
        arrived.Clear();
        int numDeaths = 0;

        for (int s = 0; s < 64; s++)
        {
          Piece pb = PieceAt(in before, s);
          Piece pa = PieceAt(in after, s);
          bool bOcc = pb.Type != PieceType.None;
          bool aOcc = pa.Type != PieceType.None;
          if (bOcc == aOcc && (!bOcc || (pb.Type == pa.Type && pb.Side == pa.Side)))
          {
            continue; // unchanged
          }

          if (bOcc && pb.Side != mover)
          {
            // Enemy piece disappeared/replaced: captured during this transition
            // (arrival capture if aOcc, en-passant vacation if !aOcc).
            if (cur[s] < 0)
            {
              throw new Exception($"SurvivalLabeler: capture on square {s} with no tracked piece id (ply {t})");
            }
            deathPly[cur[s]] = t + 1;
            numDeaths++;
          }

          if (bOcc && pb.Side == mover)
          {
            // Mover piece left this square (possibly another mover piece arrived here — Chess960 castling overlap).
            vacatedMover.Add(s);
          }

          if (aOcc)
          {
            if (pa.Side != mover)
            {
              throw new Exception($"SurvivalLabeler: non-mover piece arrived on square {s} (ply {t})");
            }
            arrived.Add(s);
          }

          nxt[s] = -1; // cleared; arrivals assigned below
        }

        if (numDeaths > 1)
        {
          throw new Exception($"SurvivalLabeler: multiple captures in one transition (ply {t})");
        }

        // Match each arrival square to its vacated source (piece identity relocation).
        foreach (int a in arrived)
        {
          Piece pa = PieceAt(in after, a);
          int srcIndexInList = -1;

          // Exact piece-type match (normal moves; castling pairs king->king, rook->rook).
          for (int v = 0; v < vacatedMover.Count; v++)
          {
            if (PieceAt(in before, vacatedMover[v]).Type == pa.Type)
            {
              srcIndexInList = v;
              break;
            }
          }
          // Promotion: the arriving piece type has no vacated match; source is the vacated pawn.
          if (srcIndexInList < 0)
          {
            for (int v = 0; v < vacatedMover.Count; v++)
            {
              if (PieceAt(in before, vacatedMover[v]).Type == PieceType.Pawn)
              {
                srcIndexInList = v;
                break;
              }
            }
          }
          if (srcIndexInList < 0)
          {
            throw new Exception($"SurvivalLabeler: no vacated source found for arrival on square {a} (ply {t})");
          }

          int src = vacatedMover[srcIndexInList];
          nxt[a] = cur[src];
          vacatedMover.RemoveAt(srcIndexInList);
        }

        ids[t + 1] = nxt;
      }

      // Emit label bytes.
      byte[][] result = new byte[numPos][];
      for (int t = 0; t < numPos; t++)
      {
        byte[] row = new byte[64];
        int[] cur = ids[t];
        for (int s = 0; s < 64; s++)
        {
          int id = cur[s];
          if (id < 0)
          {
            row[s] = 0;
          }
          else
          {
            int dp = deathPly[id];
            long d = dp == int.MaxValue ? long.MaxValue : (long)dp - t;
            row[s] = (byte)(d >= 1 && d <= horizonPlies ? d : horizonPlies + 1);
          }
        }
        result[t] = row;
      }

      return result;
    }


    static Piece PieceAt(in Position pos, int squareNum)
      => pos.PieceOnSquare(new Square(squareNum, Square.SquareIndexType.BottomToTopLeftToRight));
  }
}
