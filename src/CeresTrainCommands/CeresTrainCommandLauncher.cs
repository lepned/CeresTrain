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
using System.IO;
using System.CommandLine;

using Ceres.Base.Misc;
using Ceres.Chess;

using CeresTrain.Examples;
using CeresTrain.UserSettings;
using CeresTrain.CeresTrainDefaults;
using CeresTrain.Tasks;
using CeresTrain.TrainingDataGenerator.GeneratorFromPuzzles;

#endregion

namespace CeresTrain.TrainCommands
{
    /// <summary>
    /// Runs console session (with command line arguments) for CeresTrain,
    /// allowing for training, evaluation, and running of Ceres networks.
    /// </summary>
    public static class CeresTrainCommandLauncher
  {
    static RootCommand rootCommand;

    static Option<string> configOption;
    static Option<long> numPosOption;
    static Option<int> numTPGSetsOption;
    static Option<long> genTpgNumPosOption;   // gen-tpg: default 0 = use num-sets path
    static Option<bool> frcOnlyOption;        // gen-tpg: when true, extract ONLY Chess960/FRC games (default false = standard only)
    static Option<string> piecesOptionRequired;
    static Option<string> piecesOptionOptional;
    static Option<string> netSpecificationOption;
    static Option<string> netSpecificationOptionalOption;
    static Option<string> netSpecificationFillinOption;
    static Option<string> tpgDirOption;
    static Option<string> tarDirOption;
    static Option<string> packedZSTDirOption;
    static Option<bool> verboseOption;
    static Option<string> hostOption;
    static Option<string> searchLimitOptionDefaultBV;
    static Option<string> searchLimitOption;
    static Option<int[]> devicesOption;
    static Option<string> epdOrPgnFnOption;
    static Option<string> epdOrPgnOutputFileNameOption;

    static Command infoCommand;
    static Command trainCommand;
    static Command evalCommand;
    static Command sampleTrainCommand;
    static Command tournCommand;
    static Command evalLC0Command;
    static Command generateEndgameTPGCommand;
    static Command extractPositionsCommand;
    static Command uciCommand;
    static Command initCommand;

    static Command generateTPGCommand;
    static Command convertTARToPackedZSTCommand;

    static Option<string> puzzleConfigOption;
    static Command minePuzzlesCommand;
    static Command labelPuzzlesCommand;
    static Command puzzlesToTPGCommand;
    static Command puzzleReplayCommand;
    static Command evalLabeledCommand;
    static Command fastLabelCommand;
    static Command enrichValueCommand;
    static Command teacherValueCommand;
    static Command teacherChildrenCommand;
    static Command softLabelCommand;
    static Command enrichActionCommand;
    static Command enrichOppDefenceCommand;
    static Command oppDefDeepenSmokeCommand;

    /// <summary>
    /// Starts a console session for CeresTrain, reading command line arguments and executing the appropriate command.
    /// </summary>
    /// <param name="args"></param>
    /// <exception cref="Exception"></exception>
    public static void LaunchProcessCommandLine(string[] args)
    {
      rootCommand = new RootCommand("CeresTrain - command line executor for CeresTrain to train/evaluate/run Ceres networks.");

      configOption = new Option<string>("--config", "Configuration name") { IsRequired = true };
      numPosOption = new Option<long>("--num-pos", () => 2048, "Number of positions") { };
      numTPGSetsOption = new Option<int>("--num-tpg-sets", () => 1, "Number of sets of TPG positions to generate (~200mm positions per set)") { };
      genTpgNumPosOption = new Option<long>("--num-pos", () => 0, "Override num-sets: emit exactly this many positions (use for small quiet-anchor streams). 0 = use num-sets.") { };
      frcOnlyOption = new Option<bool>("--frc-only", () => false, "If true, INVERT variant filter: keep only Chess960/FRC games, skip standard. Default false (legacy: keep standard, drop FRC).") { };
      piecesOptionRequired = new Option<string>("--pieces", "Chess pieces (e.g. KRPkrp)") { IsRequired = true };
      piecesOptionOptional = new Option<string>("--pieces", "Chess pieces (e.g. KRPkrp)") { IsRequired = false };
      netSpecificationOption = new Option<string>("--net-spec", "LC0 network specification in Ceres format, e.g. LC0:703810") { IsRequired = true };
      netSpecificationOptionalOption = new Option<string>("--net-spec", "LC0 network specification used to compare performance against (or null for tablebase)") { IsRequired = false };
      netSpecificationFillinOption = new Option<string>("--net-spec-fillin", "LC0 network specification of network to use for noncovered positions") { IsRequired = false };
      tpgDirOption = new Option<string>("--tpg-dir", "Directory containing TPG training data files") { IsRequired = false };
      tarDirOption = new Option<string>("--tar-dir", "Directory containing TAR training data files") { IsRequired = false };
      packedZSTDirOption = new Option<string>("--zst-dir", "Directory containing packed ZST training data files") { IsRequired = true };
      verboseOption = new Option<bool>("--verbose", "If verbose information should be sent to Console (true of false).");
      hostOption = new Option<string>("--host", "Name of host (or WSL) on which to execute command.") { IsRequired = false };
      devicesOption = new Option<int[]>("--devices", "List of indices of devices to use.") { IsRequired = false };
      searchLimitOptionDefaultBV = new Option<string>("--search-limit", () => "bv", "Search limit to use (e.g. BV for best value).") { IsRequired = false };
      searchLimitOption = new Option<string>("--search-limit", () => null, "Search limit to use if tournament is to be r.") { IsRequired = false };
      epdOrPgnFnOption = new Option<string>("--pos-fn", "EPD or PGN file name used to source positions") { IsRequired = false };
      epdOrPgnOutputFileNameOption = new Option<string>("--pos-out-fn", "Name of PGN or EPD file from which to extract positions") { IsRequired = true };
      puzzleConfigOption = new Option<string>("--puzzle-config", "Path to PuzzleReplayOptions JSON file") { IsRequired = true };

      // Add commands
      initCommand = new Command("init", "Initializes new config with default values.                     [config]") { configOption };
      infoCommand = new Command("info", "Display information about a configuration.                      [config]") { configOption };
      trainCommand = new Command("train", "Start training using a configuration.                           [config] [pieces] [num-pos] [tpg-dir] [host] [devices]") { configOption, piecesOptionOptional, numPosOption, tpgDirOption, hostOption, devicesOption };
      evalCommand = new Command("eval", "Evaluate accuracy of last trained net.                          [config] [pieces] [num-pos] [pos-fn] [verbose] [net-spec]") { configOption, piecesOptionRequired, numPosOption, verboseOption, netSpecificationOptionalOption, epdOrPgnFnOption };
      tournCommand = new Command("tourn", "Run tournament between net and specified LC0 net (or TB).       [config] [pieces] [num-pos] [pos-fn] [verbose] [net-spec] [search-limit]") { configOption, piecesOptionRequired, numPosOption, epdOrPgnFnOption, verboseOption, netSpecificationOptionalOption, searchLimitOptionDefaultBV };
      sampleTrainCommand = new Command("sample-train", "Launches sample training session (needs C# configuration).") { };
      uciCommand = new Command("uci", "Launch trained net with specified configuration as UCI engine.  [config] [pieces] [net-spec-fillin]") { configOption, piecesOptionRequired, netSpecificationFillinOption };
      evalLC0Command = new Command("eval-lc0", "Evaluate vs LC0 with specific pieces and network.               [pieces] [net-spec] [num-pos] [search-limit] [pos-fn] [verbose]") { piecesOptionRequired, netSpecificationOption, numPosOption, searchLimitOption, epdOrPgnFnOption, verboseOption };
      extractPositionsCommand = new Command("extract-pos", "Generate EPD/PGN file with positions from specified PGN/EPD     [pieces] [num-pos] [pos-fn] [pos-out-fn]") { piecesOptionRequired, numPosOption, epdOrPgnFnOption, epdOrPgnOutputFileNameOption };
      generateEndgameTPGCommand = new Command("gen-endgame-tpg", "Generate TPG files with positions from specified pieces or \"*\"  [pieces] [num-pos] [tar-dir] [tpg-dir]") { piecesOptionRequired, numPosOption, tarDirOption, tpgDirOption };
      generateTPGCommand = new Command("gen-tpg", "Generate TPG files from TAR files.                              [tar-dir] [tpg-dir] [num-sets|num-pos] [--frc-only]") { tarDirOption, tpgDirOption, numTPGSetsOption, genTpgNumPosOption, frcOnlyOption };
      convertTARToPackedZSTCommand = new Command("convert-tar-to-zst", "Convert TAR files to packed ZST files.                          [tar-dir] [zst-dir]") { tarDirOption, packedZSTDirOption };

      minePuzzlesCommand = new Command("mine-puzzles", "Mine hard Lichess puzzles at nodes=1.                           [puzzle-config]") { puzzleConfigOption };
      labelPuzzlesCommand = new Command("label-puzzles", "Teacher-label hard puzzles via MCGS search.                     [puzzle-config]") { puzzleConfigOption };
      puzzlesToTPGCommand = new Command("puzzles-to-tpg", "Convert labeled puzzles to TPG training shards.                 [puzzle-config]") { puzzleConfigOption };
      puzzleReplayCommand = new Command("puzzle-replay", "Run full puzzle replay pipeline (mine + label + tpg).           [puzzle-config]") { puzzleConfigOption };
      evalLabeledCommand = new Command("eval-labeled", "Evaluate a trained net on the training set (labeled.jsonl).     [puzzle-config]") { puzzleConfigOption };
      fastLabelCommand = new Command("label-fast", "Direct-label puzzles from Lichess CSV (no NN search, theme-WDL). [puzzle-config]") { puzzleConfigOption };
      enrichValueCommand = new Command("enrich-value-labels", "Enrich labeled.jsonl with opp-to-move + counterfactual + pre-blunder records. [puzzle-config]") { puzzleConfigOption };
      teacherValueCommand = new Command("teacher-value-label", "Stage 3: teacher-label per-position WDL via NNEvaluator (NetSpec in config). [puzzle-config]") { puzzleConfigOption };
      teacherChildrenCommand = new Command("teacher-label-children", "Emit one OppDefence record per Standard with teacher WDL on child position. [puzzle-config]") { puzzleConfigOption };
      softLabelCommand = new Command("soft-label-puzzles", "Rank-1 soft-label Standard+OppDefence records using orig NN + ε margin. [puzzle-config]") { puzzleConfigOption };
      enrichActionCommand = new Command("enrich-action-head", "Emit OppDefence + K OAIS records per Standard parent using action-head teacher (e.g. C3-768). [puzzle-config]") { puzzleConfigOption };
      enrichOppDefenceCommand = new Command("enrich-opp-defence", "Add MCGS-search-backed OppDefence records (post-solver-move positions) to labeled.jsonl. Targets the value-head opp-to-move calibration gap. [puzzle-config]") { puzzleConfigOption };
      oppDefDeepenSmokeCommand = new Command("oppdef-deepen-smoke", "Re-search a sample of low-|Q| OppDef records at a higher node budget; report whether they resolve to sharper Q. [puzzle-config]") { puzzleConfigOption };

      rootCommand.AddCommand(initCommand);
      rootCommand.AddCommand(infoCommand);
      rootCommand.AddCommand(trainCommand);
      rootCommand.AddCommand(evalCommand);
      rootCommand.AddCommand(tournCommand);
      rootCommand.AddCommand(sampleTrainCommand);
      rootCommand.AddCommand(uciCommand);
      rootCommand.AddCommand(evalLC0Command);
      rootCommand.AddCommand(extractPositionsCommand);
      rootCommand.AddCommand(generateEndgameTPGCommand);
      rootCommand.AddCommand(generateTPGCommand);
      rootCommand.AddCommand(convertTARToPackedZSTCommand);
      rootCommand.AddCommand(minePuzzlesCommand);
      rootCommand.AddCommand(labelPuzzlesCommand);
      rootCommand.AddCommand(puzzlesToTPGCommand);
      rootCommand.AddCommand(puzzleReplayCommand);
      rootCommand.AddCommand(evalLabeledCommand);
      rootCommand.AddCommand(fastLabelCommand);
      rootCommand.AddCommand(enrichValueCommand);
      rootCommand.AddCommand(teacherValueCommand);
      rootCommand.AddCommand(teacherChildrenCommand);
      rootCommand.AddCommand(softLabelCommand);
      rootCommand.AddCommand(enrichActionCommand);
      rootCommand.AddCommand(enrichOppDefenceCommand);
      rootCommand.AddCommand(oppDefDeepenSmokeCommand);

      InstallCommandHandlers();

      rootCommand.Invoke(args);
    }


    /// <summary>
    /// Installs the handlers associated with all of the available commands.
    /// </summary>
    /// <exception cref="Exception"></exception>
    static void InstallCommandHandlers()
    {
      string configsDir = Path.Combine(CeresTrainUserSettingsManager.Settings.OutputsDir, "configs");
      string resultsDir = Path.Combine(CeresTrainUserSettingsManager.Settings.OutputsDir, "results");

      Console.WriteLine();

      initCommand.SetHandler((configID) =>
      {
        if (configID == null)
        {
          ConsoleUtils.WriteLineColored(ConsoleColor.Red, CeresTrainCommandUtils.NO_CONFIG_ERR_STR);
          throw new Exception();
        }

        CeresTrainCommands.ProcessInitCommand(in CeresTrainDefault.DEFAULT_CONFIG_TRAINING, configID);
      }, configOption);

      extractPositionsCommand.SetHandler((string pieces, long numPos, string epdOrPgnFnOption, string epdOrPgnOutputFileNameOption) =>
      {
        CeresTrainCommands.ExtractToEPD(epdOrPgnFnOption, new PieceList(pieces), epdOrPgnOutputFileNameOption, (int)numPos);
      }, piecesOptionRequired, numPosOption, epdOrPgnFnOption, epdOrPgnOutputFileNameOption);


      infoCommand.SetHandler((configID) =>
      {
        CeresTrainCommands.ProcessInfoCommand(configID, configsDir);
      }, configOption);


      uciCommand.SetHandler((configID, piecesStr, netSpecFillInStr) =>
      {
        CeresTrainCommands.ProcessUCICommand(configID, piecesStr, configsDir, netSpecFillInStr);
      }, configOption, piecesOptionRequired, netSpecificationFillinOption);


      convertTARToPackedZSTCommand.SetHandler((sourceDir, targetDir) =>
      {
        TPGConvertFromTAR.GeneratePackedZSTFromTARs(sourceDir, targetDir);
      }, tarDirOption, packedZSTDirOption); 


      generateTPGCommand.SetHandler((sourceDir, targetDir, numSets, numPos, frcOnly) =>
      {
        string variantSuffix = frcOnly ? " (FRC-only extraction)" : "";
        if (numPos > 0)
        {
          // explicit position count overrides numSets — emit exactly numPos positions
          // NOTE: GenerateTPGCustomSize does not yet accept frcOnly; if --frc-only is
          // requested alongside --num-pos we route through the long-form GenerateTPG
          // entry which does accept it (passing numPos as numPositionsTotal).
          if (frcOnly)
          {
            TPGConvertFromTAR.GenerateTPG(sourceDir, targetDir, numPos, debugMode: false,
                                          description: $"FRC-only extraction ({numPos} positions)",
                                          extractOnlyFRC: true);
          }
          else
          {
            TPGConvertFromTAR.GenerateTPGCustomSize(sourceDir, targetDir, numPos, $"Converted using TPGConvertFromTAR.GenerateTPGCustomSize ({numPos} positions)");
          }
        }
        else
        {
          TPGConvertFromTAR.GenerateTPG(sourceDir, targetDir, numSets, "Converted using TPGConvertFromTAR.GenerateTPG" + variantSuffix, extractOnlyFRC: frcOnly);
        }
      }, tarDirOption, tpgDirOption, numTPGSetsOption, genTpgNumPosOption, frcOnlyOption);


      generateEndgameTPGCommand.SetHandler((piecesStr, numPos, tarDirectory, outDirectory) =>
      {
        if (tarDirectory == null)
        {
          CeresNetEvaluation.GenerateTPGFilesFromTablebasePositions(piecesStr, numPos, outDirectory);
        }
        else
        {
          CeresNetEvaluation.GenerateTPGFilesFromLC0TrainingData(piecesStr, numPos, tarDirectory, outDirectory);
        }
      }, piecesOptionRequired, numPosOption, tarDirOption, tpgDirOption);


      trainCommand.SetHandler((configID, piecesStr, numPos, tpgDir, hostName, devices) =>
      {
        CeresTrainCommands.ProcessTrainCommand(configID, piecesStr, numPos, hostName, tpgDir, devices, null);
      }, configOption, piecesOptionOptional, numPosOption, tpgDirOption, hostOption, devicesOption);


      evalCommand.SetHandler((configID, piecesStr, numPos, epdOrPgnFN, verbose, netSpecification) =>
      {
        CeresTrainCommands.RunEvalOrTournament(configID, piecesStr, numPos, epdOrPgnFN, verbose, netSpecification, false, configsDir, false, default);
      }, configOption, piecesOptionRequired, numPosOption, epdOrPgnFnOption,  verboseOption, netSpecificationOptionalOption);

      sampleTrainCommand.SetHandler(() =>
      {
        LaunchDistributedTraining.Run();
      });

      tournCommand.SetHandler((configID, piecesStr, numPos, epdOrPgnFN, verbose, compareLC0NetSpec, searchLimitSpec) =>
      {
        SearchLimit searchLimit = SearchLimitSpecificationString.Parse(searchLimitSpec);
        bool enableOpponentTB = compareLC0NetSpec == null;
        CeresTrainCommands.RunEvalOrTournament(configID, piecesStr, numPos, epdOrPgnFN, verbose, compareLC0NetSpec, enableOpponentTB, configsDir, true, searchLimit);
      }, configOption, piecesOptionRequired, numPosOption, epdOrPgnFnOption, verboseOption, netSpecificationOptionalOption, searchLimitOptionDefaultBV);


      evalLC0Command.SetHandler((piecesStr, netID, numPos, searchLimitSpec, epdOrPgnFN, verbose) =>
      {
        SearchLimit searchLimit = searchLimitSpec == null ? default : SearchLimitSpecificationString.Parse(searchLimitSpec);
        CeresTrainCommands.ProcessEvalLC0Command(piecesStr, netID, numPos, epdOrPgnFN, searchLimit, verbose);
      }, piecesOptionRequired, netSpecificationOption, numPosOption, searchLimitOption, epdOrPgnFnOption, verboseOption);

      minePuzzlesCommand.SetHandler((configPath) =>
      {
        PuzzleReplayOptions opts = PuzzleReplayOptions.Load(configPath);
        PuzzleMiner.Run(opts);
      }, puzzleConfigOption);

      labelPuzzlesCommand.SetHandler((configPath) =>
      {
        PuzzleReplayOptions opts = PuzzleReplayOptions.Load(configPath);
        PuzzleTeacherLabeler.Run(opts);
      }, puzzleConfigOption);

      puzzlesToTPGCommand.SetHandler((configPath) =>
      {
        PuzzleReplayOptions opts = PuzzleReplayOptions.Load(configPath);
        PuzzleToTPGGenerator.Run(opts);
      }, puzzleConfigOption);

      puzzleReplayCommand.SetHandler((configPath) =>
      {
        PuzzleReplayOptions opts = PuzzleReplayOptions.Load(configPath);
        PuzzleReplayPipeline.Run(opts);
      }, puzzleConfigOption);

      evalLabeledCommand.SetHandler((configPath) =>
      {
        PuzzleReplayOptions opts = PuzzleReplayOptions.Load(configPath);
        PuzzleEvalOnLabeled.Run(opts);
      }, puzzleConfigOption);

      fastLabelCommand.SetHandler((configPath) =>
      {
        PuzzleReplayOptions opts = PuzzleReplayOptions.Load(configPath);
        PuzzleFastLabeler.Run(opts);
      }, puzzleConfigOption);

      enrichValueCommand.SetHandler((configPath) =>
      {
        PuzzleReplayOptions opts = PuzzleReplayOptions.Load(configPath);
        string inputPath = opts.LabeledJsonlPath;
        string outputPath = System.IO.Path.Combine(opts.OutDir, "labeled_enriched.jsonl");
        PuzzleValueEnricher.Run(opts, inputPath, outputPath);
      }, puzzleConfigOption);

      teacherValueCommand.SetHandler((configPath) =>
      {
        PuzzleReplayOptions opts = PuzzleReplayOptions.Load(configPath);
        string outputPath = System.IO.Path.Combine(opts.OutDir, "labeled_teacher.jsonl");
        PuzzleValueLabeler.Run(opts, outputPath);
      }, puzzleConfigOption);

      teacherChildrenCommand.SetHandler((configPath) =>
      {
        PuzzleReplayOptions opts = PuzzleReplayOptions.Load(configPath);
        string inputPath = opts.LabeledJsonlPath;
        string outputPath = System.IO.Path.Combine(opts.OutDir, "labeled_teacher_plus.jsonl");
        PuzzleValueLabelerChildren.Run(opts, inputPath, outputPath);
      }, puzzleConfigOption);

      softLabelCommand.SetHandler((configPath) =>
      {
        PuzzleReplayOptions opts = PuzzleReplayOptions.Load(configPath);
        string outputPath = System.IO.Path.Combine(opts.OutDir, "labeled_soft.jsonl");
        PuzzleSoftLabeler.Run(opts, outputPath);
      }, puzzleConfigOption);

      enrichActionCommand.SetHandler((configPath) =>
      {
        PuzzleReplayOptions opts = PuzzleReplayOptions.Load(configPath);
        string inputPath = opts.LabeledJsonlPath;
        string outputPath = System.IO.Path.Combine(opts.OutDir, "labeled_action_enriched.jsonl");
        PuzzleValueLabelerActionChildren.Run(opts, inputPath, outputPath);
      }, puzzleConfigOption);

      enrichOppDefenceCommand.SetHandler((configPath) =>
      {
        PuzzleReplayOptions opts = PuzzleReplayOptions.Load(configPath);
        string outputPath = System.IO.Path.Combine(opts.OutDir, "labeled_with_oppdef.jsonl");
        PuzzleOppDefenceEnricher.Run(opts, outputPath);
      }, puzzleConfigOption);

      oppDefDeepenSmokeCommand.SetHandler((configPath) =>
      {
        PuzzleReplayOptions opts = PuzzleReplayOptions.Load(configPath);
        PuzzleOppDefenceDeepenSmoke.Run(opts);
      }, puzzleConfigOption);
    }

  }
}
