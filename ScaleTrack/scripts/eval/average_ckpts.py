"""Average the policy weights of several checkpoints of ONE fine-tuning run (stochastic weight averaging / "model soup" along the trajectory).

The checkpoints of a plateaued PPO run jitter around a good solution: single checkpoints differ in a few fragile clips (for example the
success of individual dynamic BONES clips), the average of the last few is usually at least as good as the best of them and removes the jitter.
Only `model_state_dict` is averaged (floating point tensors, in float64); everything else (optimizer states, iteration, infos) is taken from the
last checkpoint of the list, so the result loads exactly like any other checkpoint of the run.

usage: average_ckpts.py --ckpts model_26000.pt model_26200.pt ... --out soup.pt
"""
import argparse

import torch

parser = argparse.ArgumentParser()
parser.add_argument("--ckpts", nargs="+", required=True)
parser.add_argument("--out", required=True)
args = parser.parse_args()

ckpts = [torch.load(p, map_location="cpu", weights_only=False) for p in args.ckpts]
out = dict(ckpts[-1])
state = {}
for key, last in ckpts[-1]["model_state_dict"].items():
    if last.is_floating_point():
        acc = torch.zeros_like(last, dtype=torch.float64)
        for c in ckpts:
            acc += c["model_state_dict"][key].double()
        state[key] = (acc / len(ckpts)).to(last.dtype)
    else:
        state[key] = last
out["model_state_dict"] = state
out["infos"] = {"averaged_checkpoints": [p.split("/")[-1] for p in args.ckpts]}
torch.save(out, args.out)
spread = max(
    float((c["model_state_dict"][k].double() - state[k].double()).abs().max())
    for c in ckpts
    for k, v in state.items()
    if v.is_floating_point()
)
print(f"wrote {args.out}: average of {len(ckpts)} checkpoints ({args.ckpts[0].split('/')[-1]} .. {args.ckpts[-1].split('/')[-1]}), "
      f"max |checkpoint - average| over all weights = {spread:.3g}")
