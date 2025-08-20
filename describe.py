#!/usr/bin/env python3
"""
describe.py — Write a text file listing all files under a root, then append
an ASCII tree of the directory structure.

Usage:
  python describe.py [root_dir] [-o OUTPUT] [--no-hidden] [--follow] [--max-depth N]

Examples:
  python describe.py
  python describe.py /path/to/project -o files_and_tree.txt
  python describe.py . --no-hidden --max-depth 3
"""

from __future__ import annotations
import os
import argparse
from pathlib import Path
from typing import Iterable, List, Optional


def iter_files(
    root: str,
    include_hidden: bool = True,
    follow_symlinks: bool = False,
) -> Iterable[str]:
    """Yield relative file paths under root, sorted, honoring hidden/symlinks."""
    root = os.path.abspath(root)
    for dirpath, dirnames, filenames in os.walk(
        root, followlinks=follow_symlinks, onerror=lambda e: None
    ):
        # Optionally filter out hidden directories/files (dot-prefixed)
        if not include_hidden:
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]
            filenames = [f for f in filenames if not f.startswith(".")]

        for name in sorted(filenames):
            full = os.path.join(dirpath, name)
            # Guard: skip broken symlinks or unreadables
            try:
                # Ensure it’s a file (not a dir that slipped in)
                if not os.path.islink(full) and not os.path.isfile(full):
                    continue
            except OSError:
                continue
            yield os.path.relpath(full, root)


def build_tree(
    root: str,
    include_hidden: bool = True,
    follow_symlinks: bool = False,
    max_depth: Optional[int] = None,
) -> str:
    """Return an ASCII tree representation starting at root."""
    root_path = Path(root).resolve()

    def list_entries(dir_path: Path) -> List[Path]:
        try:
            entries = sorted(dir_path.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
        except Exception:
            return []
        if not include_hidden:
            entries = [e for e in entries if not e.name.startswith(".")]
        return entries

    def render(dir_path: Path, prefix: str = "", depth: int = 0) -> List[str]:
        lines: List[str] = []
        entries = list_entries(dir_path)
        total = len(entries)
        for i, entry in enumerate(entries):
            connector = "└── " if i == total - 1 else "├── "
            lines.append(f"{prefix}{connector}{entry.name}")

            is_dir = False
            try:
                is_dir = entry.is_dir() and (follow_symlinks or not entry.is_symlink())
            except OSError:
                is_dir = False

            if is_dir and (max_depth is None or depth < max_depth):
                extension = "    " if i == total - 1 else "│   "
                lines.extend(render(entry, prefix + extension, depth + 1))
        return lines

    header = [str(root_path)]
    body = render(root_path)
    # If empty, still show root
    return "\n".join(header + body)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write all file paths to a text file and append a directory tree."
    )
    parser.add_argument(
        "root",
        nargs="?",
        default=".",
        help="Root directory to scan (default: current directory).",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="files_and_tree.txt",
        help="Output text filename (default: files_and_tree.txt).",
    )
    parser.add_argument(
        "--no-hidden",
        action="store_true",
        help="Exclude dotfiles and dot-directories.",
    )
    parser.add_argument(
        "--follow",
        action="store_true",
        help="Follow symlinks to directories (may be slow or loop if cyclic).",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="Limit tree depth (directories). Default: unlimited.",
    )

    args = parser.parse_args()
    root = os.path.abspath(args.root)
    include_hidden = not args.no_hidden

    # Ensure output directory exists
    out_path = Path(args.output).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"# File list for {root}\n\n")
        for rel in iter_files(root, include_hidden=include_hidden, follow_symlinks=args.follow):
            f.write(rel + "\n")
            count += 1

        f.write("\n\n# Directory tree\n\n")
        f.write(build_tree(root, include_hidden=include_hidden, follow_symlinks=args.follow, max_depth=args.max_depth))
        f.write("\n")

    print(f"Wrote {count} file paths and the directory tree to: {out_path}")


if __name__ == "__main__":
    main()
