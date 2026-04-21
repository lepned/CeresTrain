#region License notice

/*
  This file is part of the CeresTrain project at https://github.com/dje-dev/cerestrain.
  Copyright (C) 2023- by David Elliott and the CeresTrain Authors.

  Ceres is free software under the terms of the GNU General Public License v3.0.
  You should have received a copy of the GNU General Public License
  along with CeresTrain. If not, see <http://www.gnu.org/licenses/>.
*/

#endregion

using System.Collections.Generic;
using System.IO;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace CeresTrain.TrainingDataGenerator.GeneratorFromPuzzles
{
  /// <summary>
  /// One position the student got wrong at nodes=1. Written by the miner,
  /// read by the teacher labeler.
  /// </summary>
  public sealed class HardPuzzleRecord
  {
    public string PuzzleId { get; set; }
    public string FEN { get; set; }
    public string SolutionUci { get; set; }
    public int SolveStepIndex { get; set; }
    public int Rating { get; set; }
    public string Themes { get; set; }

    /// <summary>CSV start FEN (before the setup move). Needed to rebuild real history for training.</summary>
    public string StartFen { get; set; }

    /// <summary>UCI move prefix applied to StartFen to reach FEN (setup move + any prior solver/opponent moves).</summary>
    public string PriorUciMoves { get; set; }

    /// <summary>Top policy move the student net produced at nodes=1.</summary>
    public string StudentTopUci { get; set; }

    /// <summary>Student's value head output (W-L) at the position.</summary>
    public float StudentV { get; set; }
  }


  /// <summary>
  /// One teacher-labeled training position. Emitted only when the teacher's
  /// top move agrees with the Lichess solution (filter policy).
  /// </summary>
  public sealed class LabeledPuzzleRecord
  {
    public string PuzzleId { get; set; }
    public string FEN { get; set; }
    public string SolutionUci { get; set; }
    public int Rating { get; set; }
    public string Themes { get; set; }

    /// <summary>CSV start FEN (before the setup move). Needed to rebuild real history for training.</summary>
    public string StartFen { get; set; }

    /// <summary>UCI move prefix applied to StartFen to reach FEN (setup move + any prior solver/opponent moves).</summary>
    public string PriorUciMoves { get; set; }

    public int TeacherNodes { get; set; }
    public string TeacherTopUci { get; set; }
    public float TeacherV { get; set; }
    public float TeacherW { get; set; }
    public float TeacherD { get; set; }
    public float TeacherL { get; set; }

    /// <summary>
    /// Teacher policy as list of (move UCI, visit-weighted probability).
    /// Only moves with non-zero visits are included.
    /// </summary>
    public List<PolicyEntry> TeacherPolicy { get; set; }
  }


  public sealed class PolicyEntry
  {
    public string Uci { get; set; }
    public float P { get; set; }
  }


  /// <summary>
  /// A puzzle that the labeler discarded, with the reason. Useful for health checks.
  /// </summary>
  public sealed class RejectedPuzzleRecord
  {
    public string PuzzleId { get; set; }
    public string FEN { get; set; }
    public string SolutionUci { get; set; }
    public string Reason { get; set; }
    public string TeacherTopUci { get; set; }
  }


  /// <summary>
  /// Minimal JSONL reader/writer helpers. We write one record per line, unindented,
  /// so files stream cleanly and grep as plain text.
  /// </summary>
  public static class JsonlIO
  {
    static readonly JsonSerializerOptions OPTS = new JsonSerializerOptions
    {
      WriteIndented = false,
      DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
    };


    public static void AppendLine<T>(StreamWriter sw, T record)
    {
      sw.WriteLine(JsonSerializer.Serialize(record, OPTS));
    }


    public static IEnumerable<T> Read<T>(string path)
    {
      using StreamReader sr = new StreamReader(path);
      string line;
      while ((line = sr.ReadLine()) != null)
      {
        if (string.IsNullOrWhiteSpace(line)) continue;
        yield return JsonSerializer.Deserialize<T>(line, OPTS);
      }
    }
  }
}
