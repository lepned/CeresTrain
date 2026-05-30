#region License notice

/*
  This file is part of the CeresTrain project at https://github.com/dje-dev/CeresTrain.
  Copyright (C) 2023- by David Elliott and the CeresTrain Authors.
*/

#endregion

#region Using directives

using System;
using System.Buffers;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Threading.Tasks;

using Zstandard.Net;

using Ceres.Base.Misc;
using Ceres.Chess.NNEvaluators.Ceres.TPG;
using Ceres.Chess.PositionDataInfo;

#endregion

namespace CeresTrain.Tasks
{
  /// <summary>
  /// In-place upgrade of V2 TPG shards (137 bytes/square, 9378 bytes/record) to V3
  /// (140 bytes/square, 9570 bytes/record). Writes the 3 augmented-input-feature bytes
  /// per square (our_attackers / opp_attackers / shifted_net) derived from the piece
  /// placement already present in the V2 record.
  ///
  /// Crucially: this is a BYTE-LEVEL UPGRADE — no re-labeling, no re-scoring. The
  /// existing labels, policy targets, value targets, history, etc. are preserved
  /// untouched. Only the 3 new aug bytes per square (totaling 192 extra bytes per
  /// record) are computed and inserted into the per-square slots.
  ///
  /// This is the recommended path when you have an expensive-to-regenerate V2 corpus
  /// (e.g. teacher-labeled puzzles) and want to use it with V3-trained networks.
  /// Vs full TAR→TPG regeneration, the saving is the entire MCTS-search-based
  /// labeling pipeline cost.
  ///
  /// Output throughput: ~tens of millions of positions/min on a modern multi-core
  /// machine (the per-position work is ~5µs of bitboard ops via PerSquareAttacks).
  /// </summary>
  public static class TPGConvertV2ToV3
  {
    public const int V2_BYTES_PER_POS = 9378;
    public const int V3_BYTES_PER_POS = 9570;
    public const int HEADER_BYTES = 610;          // BYTES_PER_POS - 64 * BYTES_PER_SQUARE_RECORD (same in both V2/V3)
    public const int SQ_BYTES_V2 = 137;
    public const int SQ_BYTES_V3 = 140;
    public const int NUM_AUG_BYTES = 3;

    /// <summary>
    /// Upgrade a single V2 .zst shard to V3 .zst. Streams through the file in chunks so
    /// memory usage stays bounded regardless of shard size.
    ///
    /// `parallelChunkSize` positions are decoded, aug-computed (in parallel), and
    /// re-encoded together. Larger values yield better throughput; default 8192 is
    /// a reasonable balance.
    ///
    /// Default zstd compression level is 5 (gen-tpg's "Fastest") — produces files
    /// ~10-15% larger than level 11 but compresses 3-5× faster. Use level 11 only
    /// if storage matters more than upgrade wall time.
    /// </summary>
    /// <param name="v2Path">Source V2 .zst path</param>
    /// <param name="v3Path">Destination V3 .zst path (overwritten)</param>
    /// <param name="parallelChunkSize">Positions per parallel batch</param>
    /// <param name="zstdCompressionLevel">5 = fast (default), 11 = optimal-but-slow</param>
    /// <returns>Number of positions upgraded</returns>
    public static long UpgradeFile(string v2Path, string v3Path,
                                    int parallelChunkSize = 8192,
                                    int zstdCompressionLevel = 5)
    {
      const int IO_BUF = 4 * 1024 * 1024;   // 4 MB read/write buffers — keeps disk in sequential mode
      using var v2In = new BufferedStream(File.OpenRead(v2Path), IO_BUF);
      using var v2Zstd = new ZstandardStream(v2In, CompressionMode.Decompress);
      using var v3Out = new BufferedStream(File.Create(v3Path), IO_BUF);
      using var v3Zstd = new ZstandardStream(v3Out, zstdCompressionLevel);

      // Sanity-check the file: ALL bytes must be a multiple of V2_BYTES_PER_POS, OR
      // we error early (could be V3 already, or truncated).

      byte[] v2Chunk = new byte[(long)parallelChunkSize * V2_BYTES_PER_POS];
      byte[] v3Chunk = new byte[(long)parallelChunkSize * V3_BYTES_PER_POS];
      long totalPositions = 0;

      while (true)
      {
        int bytesRead = ReadFully(v2Zstd, v2Chunk, v2Chunk.Length);
        if (bytesRead == 0) break;
        if (bytesRead % V2_BYTES_PER_POS != 0)
        {
          throw new InvalidDataException(
            $"V2 shard {v2Path}: chunk size {bytesRead} not a multiple of {V2_BYTES_PER_POS} " +
            $"(file may be corrupted, or already in V3 format).");
        }
        int positionsInChunk = bytesRead / V2_BYTES_PER_POS;

        // Process positions in parallel — each writes into its own slot of v3Chunk.
        // No locking needed since output slots are disjoint.
        Parallel.For(0, positionsInChunk, posIdx =>
        {
          UpgradeOnePosition(v2Chunk, posIdx * V2_BYTES_PER_POS,
                             v3Chunk, posIdx * V3_BYTES_PER_POS);
        });

        v3Zstd.Write(v3Chunk, 0, positionsInChunk * V3_BYTES_PER_POS);
        totalPositions += positionsInChunk;
      }

      return totalPositions;
    }

    /// <summary>
    /// In-record byte-level upgrade. Spec:
    ///   - bytes [0 .. HEADER_BYTES)               → copy unchanged
    ///   - bytes [HEADER_BYTES .. HEADER_BYTES + 64*137) (V2 squares, contiguous)
    ///     → for each of 64 squares: copy 137 V2 bytes + append 3 aug bytes
    /// Total V3 layout: HEADER_BYTES + 64 * 140 = 610 + 8960 = 9570
    /// </summary>
    private static void UpgradeOnePosition(byte[] v2Buf, int v2Off, byte[] v3Buf, int v3Off)
    {
      // Copy pre-square header (610 bytes) unchanged.
      Buffer.BlockCopy(v2Buf, v2Off, v3Buf, v3Off, HEADER_BYTES);

      // Compute aug bytes from the V2 square block (PerSquareAttacks works on raw bytes).
      ReadOnlySpan<byte> v2Squares = new ReadOnlySpan<byte>(v2Buf, v2Off + HEADER_BYTES, 64 * SQ_BYTES_V2);
      Span<byte> ourAtt = stackalloc byte[64];
      Span<byte> oppAtt = stackalloc byte[64];
      PerSquareAttacks.ComputeFromTpgSquareBytes(v2Squares, ourAtt, oppAtt);

      // For each square: copy 137 base bytes + write 3 aug bytes in slot [137..140).
      // V3 byte encoding (matches Ceres TPGSquareRecord.WritePosPieces exactly):
      //   our_byte = count * 100 / 8
      //   opp_byte = count * 100 / 8
      //   net_byte = (our - opp + 8) * 100 / 16
      for (int sq = 0; sq < 64; sq++)
      {
        int v2SqOff = v2Off + HEADER_BYTES + sq * SQ_BYTES_V2;
        int v3SqOff = v3Off + HEADER_BYTES + sq * SQ_BYTES_V3;
        Buffer.BlockCopy(v2Buf, v2SqOff, v3Buf, v3SqOff, SQ_BYTES_V2);
        int ourCount = ourAtt[sq];
        int oppCount = oppAtt[sq];
        v3Buf[v3SqOff + SQ_BYTES_V2 + 0] = (byte)(ourCount * 100 / 8);
        v3Buf[v3SqOff + SQ_BYTES_V2 + 1] = (byte)(oppCount * 100 / 8);
        v3Buf[v3SqOff + SQ_BYTES_V2 + 2] = (byte)((ourCount - oppCount + 8) * 100 / 16);
      }
    }

    /// <summary>
    /// Upgrade every *.zst shard in `inputDir` to `outputDir`. Sidecar files
    /// (.options.txt, .summary.txt) are copied unchanged. Existing files in outputDir
    /// are overwritten.
    ///
    /// `maxFilesInParallel` defaults to 4 — a reasonable balance between throughput
    /// and disk I/O contention on a typical multi-shard corpus. Each file uses its
    /// own internal parallelism (Parallel.For over positions within a chunk).
    /// </summary>
    /// <returns>Total positions upgraded across all shards.</returns>
    public static long UpgradeDirectory(string inputDir, string outputDir,
                                         int parallelChunkSize = 8192,
                                         int maxFilesInParallel = 4,
                                         int zstdCompressionLevel = 5)
    {
      if (!Directory.Exists(inputDir)) throw new DirectoryNotFoundException(inputDir);
      Directory.CreateDirectory(outputDir);

      string[] zstFiles = Directory.GetFiles(inputDir, "*.zst").OrderBy(f => f).ToArray();
      string[] sidecarFiles = Directory.GetFiles(inputDir, "*.txt").OrderBy(f => f).ToArray();

      Console.WriteLine($"V2→V3 UPGRADE: {zstFiles.Length} shards from {inputDir} to {outputDir}");
      Console.WriteLine($"  parallelism: chunk={parallelChunkSize} positions, files={maxFilesInParallel}");

      long totalPositions = 0;
      Stopwatch swAll = Stopwatch.StartNew();
      object lockObj = new();

      Parallel.ForEach(zstFiles,
        new ParallelOptions { MaxDegreeOfParallelism = maxFilesInParallel },
        zstPath =>
      {
        string fileName = Path.GetFileName(zstPath);
        string outPath = Path.Combine(outputDir, fileName);
        Stopwatch swFile = Stopwatch.StartNew();
        long positions = UpgradeFile(zstPath, outPath, parallelChunkSize, zstdCompressionLevel);
        swFile.Stop();
        lock (lockObj)
        {
          totalPositions += positions;
          double rate = positions / Math.Max(0.001, swFile.Elapsed.TotalSeconds);
          Console.WriteLine($"  {fileName}: {positions:N0} positions in {swFile.Elapsed.TotalSeconds:F1}s ({rate:N0}/sec)");
        }
      });

      // Copy sidecar files (options.txt, summary.txt) unchanged
      foreach (string side in sidecarFiles)
      {
        string outSide = Path.Combine(outputDir, Path.GetFileName(side));
        File.Copy(side, outSide, overwrite: true);
      }

      swAll.Stop();
      double overallRate = totalPositions / Math.Max(0.001, swAll.Elapsed.TotalSeconds);
      Console.WriteLine($"V2→V3 DONE: {totalPositions:N0} positions in {swAll.Elapsed.TotalSeconds:F1}s ({overallRate:N0}/sec)");
      return totalPositions;
    }

    /// <summary>
    /// Read exactly `count` bytes from `stream` into `buffer`, returning the actual number
    /// of bytes read. Returns less than `count` only at end-of-stream.
    /// </summary>
    private static int ReadFully(Stream stream, byte[] buffer, int count)
    {
      int total = 0;
      while (total < count)
      {
        int n = stream.Read(buffer, total, count - total);
        if (n <= 0) break;
        total += n;
      }
      return total;
    }
  }
}
