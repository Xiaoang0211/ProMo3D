from __future__ import annotations

import sys

from ..dispatch import _pop_model_arg, run_legacy_entry
from ._shared_args import rewrite_export_args


def main() -> None:
    model_name, forwarded_args = _pop_model_arg(sys.argv[1:])
    forwarded_args = rewrite_export_args(model_name=model_name, script_filename="build_davis_isdynamic_npz.py", argv=forwarded_args)
    run_legacy_entry("build_davis_isdynamic_npz.py", forwarded_args=forwarded_args, model_name=model_name)


if __name__ == "__main__":
    main()
