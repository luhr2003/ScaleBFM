"""Weight-space interpolation between the base and a fine-tuned checkpoint (WiSE-FT): theta = (1 - alpha) * base + alpha * ft.

A cheap post-hoc knob for the retention / terrain trade-off: alpha = 0 is the base model, alpha = 1 the fine-tuned one. Only
`model_state_dict` is interpolated (floating point tensors, in float64); everything else is copied from the fine-tuned checkpoint.
usage: interpolate_ckpt.py --base model_22200.pt --ft ft_v2/model_23000.pt --alpha 0.7 --out /path/model_wise0.7.pt
"""
import argparse

import torch

parser = argparse.ArgumentParser()
parser.add_argument("--base", required=True)
parser.add_argument("--ft", required=True)
parser.add_argument("--alpha", type=float, required=True)
parser.add_argument("--out", required=True)
args = parser.parse_args()

base = torch.load(args.base, map_location="cpu", weights_only=False)
ft = torch.load(args.ft, map_location="cpu", weights_only=False)
out = dict(ft)
state = {}
for key, value in ft["model_state_dict"].items():
    base_value = base["model_state_dict"][key]
    if value.is_floating_point():
        state[key] = ((1.0 - args.alpha) * base_value.double() + args.alpha * value.double()).to(value.dtype)
    else:
        state[key] = value
out["model_state_dict"] = state
torch.save(out, args.out)
print(f"wrote {args.out}: alpha={args.alpha}, {len(state)} tensors")
