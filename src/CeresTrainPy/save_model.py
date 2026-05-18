# License Notice

"""
This file is part of the CeresTrain project at https://github.com/dje-dev/CeresTrain.
Copyright (C) 2023- by David Elliott and the CeresTrain Authors.

Ceres is free software distributed under the terms of the GNU General Public License v3.0.
You should have received a copy of the GNU General Public License along with CeresTrain.
If not, see <http://www.gnu.org/licenses/>.
"""

# End of License Notice

import os
from typing import Dict, Any

import torch

from config import Configuration
from lora import collect_and_save_lora_parameters


def save_checkpoint(NAME : str,
               OUTPUTS_DIR : str,
               config : Configuration,
               model_nocompile,
               state : Dict[str, Any],
               num_pos : str):

  # In head-LoRA-only mode we don't save full checkpoints
  # (save_model emits the head LoRA bin instead). But when ANY env-var LoRA
  # is active (body / head-front / smolgen), those lora_A/B/alpha params live
  # in the model state_dict and would be lost without a full ckpt save — so
  # we still emit the ckpt in that case.
  _body_attn = int(os.environ.get('CERES_LORA_ATTN_RANK_DIV', '0') or 0)
  _body_ffn  = int(os.environ.get('CERES_LORA_FFN_RANK_DIV',  '0') or 0)
  _body_legacy = int(os.environ.get('CERES_LORA_TRANSFORMER_RANK_DIV', '0') or 0)
  _headfront = int(os.environ.get('CERES_LORA_HEADFRONT_RANK_DIV', '0') or 0)
  _smolgen   = int(os.environ.get('CERES_LORA_SMOLGEN_RANK_DIV', '0') or 0)
  _gtab      = int(os.environ.get('CERES_GTAB', '0') or 0) > 0
  _body_lora_active = (_body_attn > 0 or _body_ffn > 0 or _body_legacy > 0
                       or _headfront > 0 or _smolgen > 0 or _gtab)
  if config.Opt_LoRARankDivisor > 0 and not _body_lora_active:
    return

  # Save PyTorch checkpoint. We persist *state dicts* rather than live objects:
  # Lightning's fabric.save() did this transparently, but plain torch.save() of
  # the model would try to pickle every attribute including non-picklable ones
  # (e.g. SummaryWriter's background thread lock). state_dict() captures only
  # the parameter tensors, which is the right thing to checkpoint anyway.
  # The load path in train.py already expects loaded["model"] / loaded["optimizer"]
  # to be state dicts (uses load_state_dict on both).
  SAVE_FULL_NAME = os.path.join(OUTPUTS_DIR, 'nets', "ckpt_" + NAME + "_" + num_pos)
  _opt = state.get('optimizer', None)
  state_no_compile = {
    "model": model_nocompile.state_dict(),
    "optimizer": _opt.state_dict() if hasattr(_opt, 'state_dict') else _opt,
    "num_pos": num_pos,
  }
  torch.save(state_no_compile, SAVE_FULL_NAME)
  print ('INFO: CHECKPOINT_FILENAME', SAVE_FULL_NAME)


def save_model(NAME : str,
               OUTPUTS_DIR : str,
               config : Configuration,
               model_nocompile,
               state : Dict[str, Any],
               num_pos : str,
               save_all_formats : str):

  # Resolve dtype/device from the model's parameters. (LightningModule used to
  # provide model.dtype / model.device; plain nn.Module does not.)
  _first_param = next(model_nocompile.parameters())
  _model_dtype = _first_param.dtype
  _model_device = _first_param.device

  with torch.no_grad():

    # If running in LoRA fine-tuning mode, first save the LoRA weights binary
    # (for potential use merging into a different base), then continue on to do
    # the normal ONNX export. The LoRA forward (base + alpha/sqrt(r) * A @ B @ x)
    # will be captured by torch.onnx.export's tracing and constant-folded so the
    # resulting ONNX contains the merged weights.
    _body_attn = int(os.environ.get('CERES_LORA_ATTN_RANK_DIV', '0') or 0)
    _body_ffn  = int(os.environ.get('CERES_LORA_FFN_RANK_DIV',  '0') or 0)
    _body_legacy = int(os.environ.get('CERES_LORA_TRANSFORMER_RANK_DIV', '0') or 0)
    _headfront = int(os.environ.get('CERES_LORA_HEADFRONT_RANK_DIV', '0') or 0)
    _smolgen   = int(os.environ.get('CERES_LORA_SMOLGEN_RANK_DIV', '0') or 0)
    _env_lora_active = (_body_attn > 0 or _body_ffn > 0 or _body_legacy > 0
                        or _headfront > 0 or _smolgen > 0)
    if config.Opt_LoRARankDivisor > 0 or _env_lora_active:
      SAVE_FULL_NAME_LORA_BIN = os.path.join(OUTPUTS_DIR, 'nets', NAME + ".lora_" + num_pos + '.bin')
      collect_and_save_lora_parameters(model_nocompile, SAVE_FULL_NAME_LORA_BIN)
      # Fall through and also produce a merged .onnx below.

    convert_type = _model_dtype
    model_nocompile.eval()


    # AOT export. Works (generates .so file), but seemingly slower than ONNX export options.
    if False and CONVERT_ONLY:
      try:
        #m = m.cuda().to(convert_type) # this might be necessary for AOT convert, but may cause subsequent failures if running net

        # get a device capabilities string (such as cuda_sm90)
        if torch.cuda.is_available():
          device = torch.cuda.get_device_properties(0)
          compute_capability = device.major, device.minor
          hardware_postfix = f"_cuda_sm{compute_capability[0]}{compute_capability[1]}" 
        else:
          hardware_postfix = "_cpu"

        #prepare output file name and directory
        aot_output_dir = "./" + TRAINING_ID
        aot_output_path = os.path.join(aot_output_dir, TRAINING_ID + hardware_postfix + ".so")
        if not os.path.exists(aot_output_dir):
          os.mkdir(aot_output_dir)
          
        batch_dim = torch.export.Dim("batch", min=1, max=1024)
        aot_example_inputs = (torch.rand(256, 64, 137).to(convert_type).to(_model_device),
                              torch.rand(256, 64, 4).to(convert_type).to(_model_device))
        with torch.no_grad():
          so_path = torch._export.aot_compile(model_nocompile,
                                            aot_example_inputs,
                                            dynamic_shapes={"squares": {0: batch_dim}, "prior_state": {0: batch_dim}},
                                            options={"aot_inductor.output_path": aot_output_path,
                                                    "max_autotune" : True,
#                                                    "max_autotune_gemm" : True,
#                                                    "max_autotune_pointwise" : True,
#                                                    "shape_padding" : True,
#                                                    "permute_fusion":True
                                                    })
        print('INFO: AOT_BINARY', so_path)
        exit(3)
      except Exception as e:
        print(f"Warning: torch._export.aot_compile save failed, skipping. Exception details: {e}")
  


    # below simpler method fails, probably due to use of .compile
    sample_inputs = [torch.rand(256, 64, 137).to(convert_type).to(_model_device),
                     torch.rand(256, 64, config.NetDef_PriorStateDim).to(convert_type).to(_model_device)]

    if False:
      try:
        SAVE_FULL_NAME = os.path.join(OUTPUTS_DIR, 'nets', NAME + "_" + num_pos + "_jit.ts")
        m_save = torch.jit.trace(model_nocompile, sample_inputs)        
        #m_save = torch.jit.script(m) # NOTE: fails for some common Pytorch operations such as einops
        m_save.save(SAVE_FULL_NAME)
        print('INFO: TS_JIT_FILENAME', SAVE_FULL_NAME)
      except Exception as e:
        print(f"Warning: torchscript save failed, skipping. Exception details: {e}")
    
    SAVE_TS = True
    SAVE_FULL_NAME = os.path.join(OUTPUTS_DIR, 'nets', NAME + ".ts_" + num_pos)
    if SAVE_TS:
      try:
        SAVE_FULL_NAME = os.path.join(OUTPUTS_DIR, 'nets', NAME + "_" + num_pos + ".ts")
        # to_torchscript was a LightningModule helper; equivalent on plain nn.Module:
        ts_module = torch.jit.trace(model_nocompile, sample_inputs)
        ts_module.save(SAVE_FULL_NAME)
        print('INFO: TS_FILENAME', SAVE_FULL_NAME )
        #model.to_onnx(SAVE_PATH + ".onnx", test_inputs_pytorch) #, export_params=True)
      except Exception as e:
        print(f"Warning: torchscript save failed, skipping. Exception details: {e}")

    if save_all_formats:
      # Still in beta testing as of PyTorch 2.3, not yet functional: torch.onnx.dynamo_export
      # TorchDynamo based export. Encountered warning/error on export.
      if False:
        try:
          SAVE_FULL_NAME = os.path.join(OUTPUTS_DIR, 'nets', NAME + "_" + num_pos + "_dynamo.onnx")
          export_options = torch.onnx.ExportOptions(dynamic_shapes=True)
          onnx_model = torch.onnx.dynamo_export(model_nocompile, sample_inputs[0], sample_inputs[1], export_options=export_options)
          onnx_model.save(SAVE_FULL_NAME)
          print('INFO: ONNX_DYNAMO_FILENAME', SAVE_FULL_NAME)
        except Exception as e:
          print(f"Warning: torch.onnx.dynamo_export save failed, skipping. Exception details: {e}")

      # Legacy ONNX export.
      if True:
        try:
          SAVE_FULL_NAME = os.path.join(OUTPUTS_DIR, 'nets', NAME + "_" + num_pos + ".onnx")
          # Output tensor previously named 'prior_state', same as the input — which is
          # an invalid ONNX graph (tensor can't be both named input and named output).
          # Rename the output tensor to 'prior_state_out' so the model is loadable.
          head_output_names = ['policy', 'value', 'mlh', 'unc', 'value2',
                               'q_deviation_lower', 'q_deviation_upper',
                               'uncertainty_policy', 'action', 'prior_state_out',
                               'action_uncertainty']
          output_axes = {'squares' : {0 : 'batch_size'},
                          'policy' : {0 : 'batch_size'},
                          'value' : {0 : 'batch_size'},
                          'mlh' : {0 : 'batch_size'},
                          'unc' : {0 : 'batch_size'},
                          'value2' : {0 : 'batch_size'},
                          'q_deviation_lower' : {0 : 'batch_size'},
                          'q_deviation_upper' : {0 : 'batch_size'},
                          'uncertainty_policy': {0 : 'batch_size'},
                          'action': {0 : 'batch_size'},
                          'prior_state_out': {0 : 'batch_size'},
                          'prior_state': {0 : 'batch_size'},
                          'action_uncertainty': {0 : 'batch_size'},
                          }
          # Export an FP32 ONNX first, then convert the entire graph (weights + IO)
          # to FP16 via onnxconverter-common below. Keeping the initial export FP32
          # avoids mixed-type MatMul errors that occur when inputs are declared FP16
          # while internal weights are still FP32.
          sample_inputs = (torch.rand(256, 64, 137).to(torch.float32).to(_model_device),
                            torch.rand(256, 64, config.NetDef_PriorStateDim).to(torch.float32).to(_model_device))
          # Ceres's TRT inference backend (TensorRTWrapper.cpp:2209, TRT_InferAsync)
          # only calls setTensorAddress on inputNames[0] — additional inputs are
          # never bound, causing enqueueV3 to fail with "Address is not set for
          # input tensor <name>". To stay compatible with that single-input
          # assumption, when PriorStateDim==0 we wrap the model so only `squares`
          # is an ONNX input; the zero prior_state is constructed internally.
          if config.NetDef_PriorStateDim == 0:
            import torch.nn as _nn
            class _SquaresOnlyWrapper(_nn.Module):
              def __init__(self, inner, prior_state_dim):
                super().__init__(); self.inner = inner; self.prior_state_dim = prior_state_dim
              def forward(self, squares):
                ps = torch.zeros(squares.shape[0], 64, self.prior_state_dim,
                                 dtype=squares.dtype, device=squares.device)
                return self.inner(squares, ps)
            _export_model = _SquaresOnlyWrapper(model_nocompile, config.NetDef_PriorStateDim).eval()
            _export_inputs = (sample_inputs[0],)
            _input_names = ['squares']
            _output_axes_single = {k: v for k, v in output_axes.items() if k != 'prior_state'}
          else:
            _export_model = model_nocompile
            _export_inputs = (sample_inputs[0], sample_inputs[1])
            _input_names = ['squares', 'prior_state']
            _output_axes_single = output_axes

          # Opset 18 matches what current PyTorch (>=2.4) actually emits — older
          # `opset_version=17` triggers an internal 18→17 downgrade pass which
          # can raise `axes_input_to_attribute.h:68 adapt: Assertion node->hasAttribute(kaxes)`
          # in onnx's C version_converter and abort the export, losing the .onnx
          # pointer file (the .onnx.data sidecar may still be written). Ceres
          # consumes opset-18 ONNX fine, so we target 18 directly.
          torch.onnx.export(_export_model,
                            _export_inputs,
                            SAVE_FULL_NAME,
                            do_constant_folding=True,
                            export_params=True,
                            opset_version=23,
                            input_names = _input_names,
                            output_names = head_output_names,
                            dynamic_axes=_output_axes_single)
          print('INFO: ONNX_FILENAME', SAVE_FULL_NAME)

          if True:
            # Convert the whole graph (weights + IO tensors) to FP16 and overwrite
            # the primary ONNX file. Ceres expects FP16 nets when running on GPU.
            # Module location moved between library versions — prefer onnxconverter-common
            # (the actively maintained one), fall back to the older onnxmltools path.
            try:
              from onnxconverter_common.float16 import convert_float_to_float16
            except ImportError:
              from onnxmltools.utils.float16_converter import convert_float_to_float16
            import onnx as _onnx
            onnx_model = _onnx.load(SAVE_FULL_NAME)
            # Lower min_positive_val from the default 1e-7: that default floors all
            # smaller-magnitude weights up to 1e-7, which catastrophically corrupts
            # well-trained models whose fine-tuned weights are often below 1e-7.
            # 1e-10 preserves the full FP16 representable range (subnormals ~6e-8,
            # below which hardware returns 0, which is the correct behavior).
            onnx_model_16 = convert_float_to_float16(
                onnx_model, keep_io_types=False,
                min_positive_val=1e-10, max_finite_val=1e4)
            # Embed weights inline in the .onnx file (same as production Ceres nets
            # like C3-768-30-pre3-I8.onnx). Our nets are well under the 2 GB protobuf
            # limit so no external data file is needed.
            _onnx.save(onnx_model_16, SAVE_FULL_NAME)
            # Clean up any leftover external-data sidecar from a previous export.
            _data_path = SAVE_FULL_NAME + ".data"
            if os.path.exists(_data_path):
              os.remove(_data_path)
            print('INFO: ONNX_FP16_CONVERSION_APPLIED', SAVE_FULL_NAME)

        except Exception as e:
          print(f"Warning: torch.onnx.export save failed, skipping. Exception details: {e}")       
