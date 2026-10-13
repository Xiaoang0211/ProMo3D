from __future__ import annotations

import sys

from ..dispatch import _pop_model_arg, run_legacy_entry
from ._shared_args import rewrite_blockwise_probing_args


def main() -> None:
    model_name, forwarded_args = _pop_model_arg(sys.argv[1:])
    forwarded_args = rewrite_blockwise_probing_args(model_name=model_name, argv=forwarded_args)
    run_legacy_entry("run_isdynamic_blockwise_sweeps.py", forwarded_args=forwarded_args, model_name=model_name)


if __name__ == "__main__":
    main()
