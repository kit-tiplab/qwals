"""Command-line interface for ``qwals``.

Run from a folder containing ``wals-data.csv`` (and optionally
``WALS_feature_order.csv``):

    python -m qwals compare pl en
    python -m qwals nearest Polish --n 5
    python -m qwals pairwise --out distances.csv
    python -m qwals shared pl en --limit 20
    python -m qwals features Polish

After ``pip install``, an entry-point ``qwals`` is also wired up in
``pyproject.toml``, so ``qwals compare pl en`` works from any directory
containing the CSVs.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from . import __version__, QwalsCalculator


# ---------- shared helpers -------------------------------------------------

def _calc(args: argparse.Namespace) -> QwalsCalculator:
    """Build a calculator from the args common to every subcommand."""
    folder = Path(args.data).expanduser()
    return QwalsCalculator.from_folder(
        folder,
        data_filename=args.data_filename,
        order_filename=args.order_filename,
        cache=not args.no_cache,
    )


def _err(msg: str, code: int = 2) -> int:
    print(msg, file=sys.stderr)
    return code


# ---------- subcommands ----------------------------------------------------

def cmd_compare(args: argparse.Namespace) -> int:
    calc = _calc(args)
    try:
        d = calc.distance(args.language1, args.language2, method=args.method)
    except ValueError as exc:
        return _err(f"error: {exc}")
    l1 = calc.resolve_language(args.language1)
    l2 = calc.resolve_language(args.language2)
    print(f"{l1}\t{l2}\t{d:.6f}")
    return 0


def cmd_nearest(args: argparse.Namespace) -> int:
    calc = _calc(args)
    try:
        items = calc.nearest(
            args.language,
            n=args.n,
            method=args.method,
            min_shared=args.min_shared,
        )
        target = calc.resolve_language(args.language)
    except ValueError as exc:
        return _err(f"error: {exc}")
    if not items:
        thresh = args.min_shared if args.min_shared is not None else calc.NEAREST_MIN_SHARED
        return _err(
            f"warning: no neighbours of '{target}' share ≥ {thresh} features. "
            f"Try passing --min-shared with a smaller value (0 disables the filter).",
            code=0,
        )
    print(f"# {len(items)} nearest to {target} (method={args.method})")
    for name, d in items:
        print(f"{name}\t{d:.6f}")
    return 0


def cmd_pairwise(args: argparse.Namespace) -> int:
    calc = _calc(args)
    langs = args.languages or None
    if args.out:
        calc.save_pairwise_matrix(args.out, languages=langs, method=args.method)
        print(f"Wrote pairwise matrix ({len(langs) if langs else len(calc.languages)} × "
              f"{len(langs) if langs else len(calc.languages)}) to {args.out}")
    else:
        m = calc.pairwise_matrix(langs, method=args.method)
        m.to_csv(sys.stdout)
    return 0


def cmd_shared(args: argparse.Namespace) -> int:
    calc = _calc(args)
    try:
        feats = calc.shared_features(args.language1, args.language2)
        l1 = calc.resolve_language(args.language1)
        l2 = calc.resolve_language(args.language2)
    except ValueError as exc:
        return _err(f"error: {exc}")
    if args.limit and args.limit > 0:
        feats = feats[: args.limit]
    print(f"# {len(feats)} shared features between {l1} and {l2}")
    for f in feats:
        print(f)
    return 0


def cmd_features(args: argparse.Namespace) -> int:
    calc = _calc(args)
    try:
        feats = calc.features_for(args.language)
        target = calc.resolve_language(args.language)
    except ValueError as exc:
        return _err(f"error: {exc}")
    if args.limit and args.limit > 0:
        feats = feats[: args.limit]
    print(f"# {len(feats)} features for {target}")
    for f in feats:
        print(f)
    return 0


# ---------- argparse wiring -------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m qwals",
        description="Linguistic distance from WALS-style data.",
    )
    p.add_argument(
        "--data", default=".",
        help="folder containing the WALS CSVs (default: current directory)",
    )
    p.add_argument("--data-filename", default="wals-data.csv")
    p.add_argument("--order-filename", default="WALS_feature_order.csv")
    p.add_argument("--no-cache", action="store_true",
                   help="don't read or write the on-disk matrix cache")
    p.add_argument("--version", action="version", version=__version__)

    sub = p.add_subparsers(dest="cmd", required=True)

    pc = sub.add_parser("compare", help="distance between two languages")
    pc.add_argument("language1")
    pc.add_argument("language2")
    pc.add_argument("--method", choices=["ordinal", "onehot"], default="ordinal")
    pc.set_defaults(fn=cmd_compare)

    pn = sub.add_parser("nearest", help="N nearest languages to a target")
    pn.add_argument("language")
    pn.add_argument("--n", type=int, default=10)
    pn.add_argument("--method", choices=["ordinal", "onehot"], default="ordinal")
    pn.add_argument(
        "--min-shared", type=int, default=None,
        help=("only consider languages sharing at least this many features "
              "with the target (default: 50; pass 0 to disable the filter)"),
    )
    pn.set_defaults(fn=cmd_nearest)

    pp = sub.add_parser("pairwise", help="pairwise distance matrix")
    pp.add_argument("--method", choices=["ordinal", "onehot"], default="ordinal")
    pp.add_argument("--out", help="CSV path; default writes to stdout")
    pp.add_argument("languages", nargs="*", help="optional subset; default = all")
    pp.set_defaults(fn=cmd_pairwise)

    ps = sub.add_parser("shared", help="features shared between two languages")
    ps.add_argument("language1")
    ps.add_argument("language2")
    ps.add_argument("--limit", type=int, default=0,
                    help="truncate to first N (0 = all, default)")
    ps.set_defaults(fn=cmd_shared)

    pf = sub.add_parser("features", help="features available for one language")
    pf.add_argument("language")
    pf.add_argument("--limit", type=int, default=0)
    pf.set_defaults(fn=cmd_features)

    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
