"""Resume from a checkpoint into a net that has MORE parameters (opt CheckpointResumeAllowNewParameters).

Used by train.py when a run continues from one of its checkpoints with additional net switches enabled, the switch
being an exact identity at insertion (NBT-BT6 hybrid: relation / pair / edge terms in more inner layers, shared
smolgen). Three pieces:

  check_state_dict       which keys the checkpoint lacks, and a refusal of every other mismatch;
  check_identity         the new parameters hold the values that make the switch an identity;
  remap_optimizer_state  the checkpoint's optimiser state re-keyed onto the larger parameter list: every existing
                         parameter keeps its state, every new parameter starts without state.

Optimiser state_dicts key the per-parameter state by POSITION in the concatenated parameter groups. Adding
parameters shifts the positions, so a plain load would either fail the group-size check (train.py then discards the
whole state) or, if sizes happened to agree, attach moments to the wrong tensors. The remap rebuilds the checkpoint's
positions from names: the old parameter list of a group is this run's list of that group minus the new names. That
holds because (1) the groups are built by rules on the parameter NAME and shape (wd_partition.py, train.py's Muon
scopes), which give an existing parameter the same group and the same relative order whatever else exists, and (2)
nothing but additions is allowed (check_state_dict). It is verified, not assumed: group sizes, the shape of every
state tensor and Muon's per-parameter routing flag must agree, else the resume is refused.
"""

import torch

# Reader parameters of a BT6 inner layer (bt6_net.BT6Layer): what a term-free layer gains when it becomes a term layer.
BT6_READER_SUFFIXES = (".t_node", ".t_edge", ".C1", ".C2", ".We", ".Wm", ".Wg", ".bg", ".Pk", ".Pq")


def is_switch_parameter(key: str, config) -> bool:
  """True for a state_dict key that an NBT-BT6 hybrid switch may ADD on resume:
     * the reader parameters of an inner layer (BT6TermsInnerLayers widened): transformer_layer.B.inner.J.<reader>
     * the shared smolgen (BT6SharedSmolgen): transformer_layer.B.shared_smolgen.* and smolgenPrepLayer.*"""
  if not getattr(config, 'NetDef_BT6Hybrid', False):
    return False
  if ".inner." in key and key.endswith(BT6_READER_SUFFIXES):
    return True
  if getattr(config, 'NetDef_BT6SharedSmolgen', False):
    return ".shared_smolgen." in key or key.startswith("smolgenPrepLayer.")
  return False


def check_state_dict(model_sd: dict, ckpt_sd: dict, explained) -> list:
  """Returns the sorted keys the checkpoint lacks; all must satisfy `explained(key)`. Raises ValueError listing every
  other difference: keys only in the checkpoint, tensors whose shape changed, missing keys no switch explains."""
  missing = sorted(k for k in model_sd if k not in ckpt_sd)
  unexpected = sorted(k for k in ckpt_sd if k not in model_sd)
  shapes = sorted(f"{k}: checkpoint {tuple(ckpt_sd[k].shape)} vs net {tuple(model_sd[k].shape)}"
                  for k in model_sd if k in ckpt_sd and tuple(ckpt_sd[k].shape) != tuple(model_sd[k].shape))
  unexplained = [k for k in missing if not explained(k)]
  if unexpected or shapes or unexplained:
    lines = ["CheckpointResumeAllowNewParameters: the checkpoint does not fit this net beyond the parameters a newly "
             "enabled switch adds; nothing was loaded."]
    if unexpected:
      lines.append(f"  {len(unexpected)} checkpoint tensors have no place in this net (a switch was turned OFF, or the "
                   f"parameters moved): {unexpected[:12]}{' ...' if len(unexpected) > 12 else ''}")
      if any(".inner." in k and k.endswith(BT6_READER_SUFFIXES) for k in unexpected) and any(".bt6_readers." in k for k in missing):
        lines.append("  This is BT6ShareReadersPerBlock switched on at resume: the checkpoint's per-layer readers would have "
                     "to become the block's shared set, which the other inner layers then also use -- they would not start "
                     "as an identity. Not supported (REPORT_hybrid_resume.txt).")
    if shapes:
      lines.append(f"  {len(shapes)} tensors changed shape: {shapes[:12]}{' ...' if len(shapes) > 12 else ''}")
    if unexplained:
      lines.append(f"  {len(unexplained)} tensors are missing from the checkpoint and are not parameters of a switch that "
                   f"may be enabled on resume: {unexplained[:12]}{' ...' if len(unexplained) > 12 else ''}")
    raise ValueError("\n".join(lines))
  return missing


def check_identity(model, new_keys) -> dict:
  """The new parameters must hold the values that make the switch an exact identity at insertion:
     reader parameters: t_node 1, t_edge 0, C1 = C2 = Wm = Wg = bg = Pk = Pq = 0 (We is free: it is multiplied by
     t_edge = 0); shared smolgen: prep layer weight and bias 0 (the generator is free).
  These ARE the from-scratch initial values; the check guards against a later change of an init. Returns counts."""
  sd = model.state_dict()
  bad, readers, smolgen = [], 0, 0
  for key in new_keys:
    value = sd[key].detach().float()
    if ".inner." in key and key.endswith(BT6_READER_SUFFIXES):
      readers += 1
      if key.endswith(".We"):
        continue
      target = 1.0 if key.endswith(".t_node") else 0.0
      if float((value - target).abs().max()) != 0.0:
        bad.append(key)
    elif ".shared_smolgen." in key or key.startswith("smolgenPrepLayer."):
      smolgen += 1
      if key.startswith("smolgenPrepLayer.") and float(value.abs().max()) != 0.0:
        bad.append(key)
  if bad:
    raise RuntimeError(f"CheckpointResumeAllowNewParameters: new parameters are not at their identity values, the net "
                       f"would not compute what the checkpoint computed: {bad[:12]}")
  return {"reader tensors": readers, "shared-smolgen tensors": smolgen}


def remap_optimizer_state(optimizer, loaded_state: dict, model, new_names) -> tuple:
  """`loaded_state` (a checkpoint's optimizer.state_dict(), built on the net WITHOUT `new_names`) -> a state_dict for
  `optimizer` (built on `model`, which has them): (state_dict, kept, fresh). Existing parameters keep their state
  entry unchanged; new parameters get none (the optimiser creates it on first use). Raises ValueError when the
  checkpoint's layout cannot be reproduced exactly from this run's groups."""
  new_names = set(new_names)
  name_of = {id(p): n for n, p in model.named_parameters()}
  current = optimizer.state_dict()                           # index form of this run's groups (and hyper-parameters)
  loaded_groups, loaded_states = loaded_state.get("param_groups", []), loaded_state.get("state", {})
  if len(loaded_groups) != len(optimizer.param_groups):
    raise ValueError(f"optimizer resume with new parameters: the checkpoint has {len(loaded_groups)} parameter groups, "
                     f"this run {len(optimizer.param_groups)} (a different optimizer or partition)")
  state, old_position, new_position, kept, fresh = {}, 0, 0, 0, 0
  for index, (group, loaded_group) in enumerate(zip(optimizer.param_groups, loaded_groups)):
    names = [name_of[id(p)] for p in group["params"]]
    old_names = [n for n in names if n not in new_names]
    if len(old_names) != len(loaded_group["params"]):
      raise ValueError(f"optimizer resume with new parameters: group {index} holds {len(old_names)} existing parameters in "
                       f"this run but {len(loaded_group['params'])} in the checkpoint: the grouping rules changed, the "
                       f"positional state cannot be re-keyed safely")
    old_index = {n: old_position + i for i, n in enumerate(old_names)}
    for offset, (name, p) in enumerate(zip(names, group["params"])):
      if name not in old_index:
        fresh += 1
        continue
      entry = loaded_states.get(old_index[name])
      if entry is None:
        continue
      for key, value in entry.items():
        if torch.is_tensor(value) and value.dim() > 0 and tuple(value.shape) != tuple(p.shape):
          raise ValueError(f"optimizer resume with new parameters: state '{key}' of {name} has shape {tuple(value.shape)}, "
                           f"the parameter {tuple(p.shape)}: the checkpoint's parameter order is not this run's")
      flag_now = optimizer.state[p].get("use_muon") if p in optimizer.state else None
      if flag_now is not None and entry.get("use_muon") is not None and bool(entry["use_muon"]) != bool(flag_now):
        raise ValueError(f"optimizer resume with new parameters: {name} is {'Muon' if flag_now else 'AdamW'} in this run but "
                         f"{'Muon' if entry['use_muon'] else 'AdamW'} in the checkpoint (MuonAdamWScope changed?)")
      state[new_position + offset] = entry
      kept += 1
    old_position += len(loaded_group["params"])
    new_position += len(group["params"])
  return {"param_groups": current["param_groups"], "state": state}, kept, fresh
