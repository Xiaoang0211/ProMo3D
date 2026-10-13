from __future__ import annotations

import argparse
from pathlib import Path

from .layout import build_layout_lines, ensure_output_skeleton, project_root_from_package


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Shared IsDynamic probing utilities.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("print-layout", help="Print the recommended shared project layout.")

    init_parser = subparsers.add_parser(
        "init-output-layout",
        help="Create canonical output skeleton directories under the project root.",
    )
    init_parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Optional explicit project root. Defaults to the parent of the isdynamic package.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    package_root = Path(__file__).resolve().parent
    project_root = args.project_root.resolve() if getattr(args, "project_root", None) else project_root_from_package(package_root)

    if args.command == "print-layout":
        print("\n".join(build_layout_lines()))
        return

    if args.command == "init-output-layout":
        for path in ensure_output_skeleton(project_root):
            print(path)
        return

    parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
