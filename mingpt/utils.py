# pyright: basic

import json
import os
import random
import sys
from ast import literal_eval

import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logging(config):
    work_dir = config.system.work_dir
    os.makedirs(work_dir, exist_ok=True)
    with open(os.path.join(work_dir, "args.txt"), "w") as f:
        f.write(" ".join(sys.argv))

    with open(os.path.join(work_dir, "config.json"), "w") as f:
        f.write(json.dumps(config.to_dict(), index=4))


class CfgNode:
    """A lighweight configuration class inpsired by yacs."""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __str__(self):
        return self._str_helper(0)

    def _str_helper(self, indent):
        parts = []
        for k, v in self.__dict__.items():
            if isinstance(v, CfgNode):
                parts.append(f"{str(k)}:\n")
                parts.append(f"{v._str_helper(indent + 1)}")
            else:
                parts.append(f"{str(k)}: {str(v)}\n")

        parts = [" " * (indent * 4) + p for p in parts]

        return "".join(parts)

    def to_dict(self):
        return {
            k: v.to_dict() if isinstance(v, CfgNode) else v
            for k, v in self.__dict__.items()
        }

    def merge_from_dict(self, d):
        self.__dict__.update(d)

    def merge_from_args(self, args):
        for arg in args:
            keyval = arg.split("=")
            assert (
                len(keyval) == 2
            ), f"Expecting each override arg to be of form --arg=value, got {arg}"
            key, val = keyval

            try:
                val = literal_eval(val)
            except ValueError:
                pass

            assert key[:2] == "--"
            key = key[2:]
            keys = key.split(".")
            obj = self
            for k in keys[:-1]:
                obj = getattr(obj, k)

            leaf_key = keys[-1]

            assert hasattr(
                obj, leaf_key
            ), f"{key} is not an attribute that exists in the config"

            print(
                f"command line overwriting config attribute {str(key)} with {str(val)}"
            )
            setattr(obj, leaf_key, val)
