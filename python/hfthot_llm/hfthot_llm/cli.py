"""``python -m hfthot_llm`` entry point.

Subcommands:
    tables                          list lakehouse datasets.* tables
    stats  --table <name>           show rows/columns/bytes for a table
    ingest --historia <db> --thotbook <db>   run llm-corpus ingest
"""

from __future__ import annotations

import argparse
import sys

from . import __version__
from .corpus import CorpusClient


def _print(label: str, value: object) -> None:
    print(f"{label:<18}: {value}")


def cmd_tables(args: argparse.Namespace) -> int:
    client = CorpusClient(args.lakehouse)
    tables = client.tables()
    _print("mode", "RÉEL (lakehouse parquet present)")
    _print("lakehouse", client.lakehouse_dir)
    _print("tables", len(tables))
    for table in tables:
        print(f"  - {table}")
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    client = CorpusClient(args.lakehouse)
    stats = client.stats(args.table)
    _print("mode", "RÉEL (lakehouse parquet present)")
    _print("table", args.table)
    _print("rows", stats["rows"])
    _print("columns", stats["columns"])
    _print("bytes", stats["bytes"])
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    client = CorpusClient(args.lakehouse)
    result = client.ingest(args.historia, args.thotbook, lakehouse_dir=args.lakehouse)
    _print("mode", "RÉEL (SQLite sources present)")
    _print("returncode", result.returncode)
    if result.stdout:
        sys.stdout.write(result.stdout)
    if result.stderr:
        sys.stderr.write(result.stderr)
    return result.returncode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hfthot_llm",
        description="hfthot-llm Python connectors (Rust pipeline).",
    )
    parser.add_argument("--version", action="version", version=f"hfthot-llm {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_tables = sub.add_parser("tables", help="list lakehouse datasets.* tables")
    p_tables.add_argument("--lakehouse", default=None, help="lakehouse dir (default: project data/lakehouse)")
    p_tables.set_defaults(func=cmd_tables)

    p_stats = sub.add_parser("stats", help="show rows/columns/bytes for a lakehouse table")
    p_stats.add_argument("--table", default="datasets.sessions", help="table name (default: datasets.sessions)")
    p_stats.add_argument("--lakehouse", default=None, help="lakehouse dir (default: project data/lakehouse)")
    p_stats.set_defaults(func=cmd_stats)

    p_ingest = sub.add_parser("ingest", help="run llm-corpus ingest from SQLite sources into the lakehouse")
    p_ingest.add_argument("--historia", required=True, help="path to historia.db (opencode session documents)")
    p_ingest.add_argument("--thotbook", required=True, help="path to thotbook.db (research index)")
    p_ingest.add_argument("--lakehouse", default=None, help="lakehouse dir (default: project data/lakehouse)")
    p_ingest.set_defaults(func=cmd_ingest)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
