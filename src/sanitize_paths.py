"""
sanitize_paths.py — Pre-commit local path scrubber

Scans all .csv and .md files in the repo for absolute local machine paths
and replaces them with portable placeholders before git staging.

Usage:
    python src/sanitize_paths.py [--dry-run] [--root <repo-root>]

    --dry-run   Print matches without modifying files.
    --root      Repo root directory (default: parent of this script).
"""

import argparse
import glob
import os
import re
import sys


# Patterns that identify local machine paths in output files
_LOCAL_PATTERNS = [
    # Absolute home directory paths (any username)
    (re.compile(r"/Users/[^/\s\"',]+"), "<LOCAL_PATH>"),
    (re.compile(r"/home/[^/\s\"',]+"),  "<LOCAL_PATH>"),
    # Windows-style paths that may end up via WSL2
    (re.compile(r"C:\\\\?Users\\\\?[^\\s\"',]+"), "<LOCAL_PATH>"),
    # Absolute venv paths that contain a username
    (re.compile(r"/Users/[^/\s\"',]+/venv/bin/python[\d.]*"), "<VENV_PYTHON>"),
    (re.compile(r"/home/[^/\s\"',]+/venv/bin/python[\d.]*"),  "<VENV_PYTHON>"),
]

_SCAN_EXTENSIONS = ["*.csv", "*.md"]


def scan_and_sanitize(root: str, dry_run: bool) -> int:
    """
    Scan all CSV and Markdown files under root for local paths.
    Returns the number of matches found.
    """
    total_matches = 0
    files_modified = 0

    for ext in _SCAN_EXTENSIONS:
        for filepath in glob.glob(os.path.join(root, "**", ext), recursive=True):
            # Skip the brain/artifacts directory (those are agent-internal)
            if ".gemini" in filepath or "brain" in filepath:
                continue

            with open(filepath, encoding="utf-8", errors="ignore") as f:
                original = f.read()

            modified = original
            file_matches = 0

            for pattern, replacement in _LOCAL_PATTERNS:
                hits = pattern.findall(modified)
                if hits:
                    file_matches += len(hits)
                    for hit in set(hits):
                        print(f"  {'[DRY-RUN] ' if dry_run else ''}MATCH in {os.path.relpath(filepath, root)}: {hit!r} → {replacement!r}")
                    modified = pattern.sub(replacement, modified)

            if file_matches > 0:
                total_matches += file_matches
                if not dry_run and modified != original:
                    with open(filepath, "w", encoding="utf-8") as f:
                        f.write(modified)
                    files_modified += 1
                    print(f"  ✅ Sanitized: {os.path.relpath(filepath, root)}")

    return total_matches


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-commit local path scrubber")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print matches without modifying files.")
    parser.add_argument("--root", type=str,
                        default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        help="Repo root directory to scan.")
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    mode = "DRY-RUN" if args.dry_run else "LIVE"

    print(f"\n[sanitize_paths.py — {mode}]")
    print(f"  Scanning: {root}")
    print(f"  File types: {', '.join(_SCAN_EXTENSIONS)}\n")

    total = scan_and_sanitize(root, dry_run=args.dry_run)

    if total == 0:
        print("  ✅ No local paths found. Safe to commit.")
    else:
        if args.dry_run:
            print(f"\n  ⚠️  {total} local path(s) found. Run without --dry-run to sanitize.")
        else:
            print(f"\n  ✅ {total} local path(s) sanitized.")

    sys.exit(0 if total == 0 or not args.dry_run else 1)


if __name__ == "__main__":
    main()
