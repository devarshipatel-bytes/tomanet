#!/usr/bin/env python3
"""Register TomaNet modules inside the installed ultralytics package.

Ultralytics has no plugin registry (the "Python in YAML" proposal was closed on security
grounds), and `parse_model` decides channel injection and depth scaling from two
frozensets that are *local variables*. The only supported way to add a custom module is
to edit the source, so this script does exactly that - minimally, idempotently, and
reversibly.

    python scripts/patch_ultralytics.py            # apply
    python scripts/patch_ultralytics.py --check    # report status, change nothing
    python scripts/patch_ultralytics.py --revert   # remove the patch

Re-run after every `pip install -U ultralytics`.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

MARKER = "TOMANET PATCH"

IMPORT_BLOCK = """
# --- TOMANET PATCH (imports) ---
try:
    from tomanet.modules import CoordAtt, TomaBlock, TomaLayer

    _TOMANET_BASE = frozenset({TomaBlock, TomaLayer})
    _TOMANET_REPEAT = frozenset({TomaLayer})
except Exception:  # tomanet not installed - leave ultralytics fully usable
    _TOMANET_BASE = frozenset()
    _TOMANET_REPEAT = frozenset()
# --- END TOMANET PATCH (imports) ---
"""

USAGE_BLOCK = """{indent}# --- TOMANET PATCH (module sets) ---
{indent}base_modules = base_modules | _TOMANET_BASE
{indent}repeat_modules = repeat_modules | _TOMANET_REPEAT
{indent}# --- END TOMANET PATCH (module sets) ---
"""

# First line inside parse_model that reads base_modules; both frozensets exist by then.
USAGE_ANCHOR = re.compile(r"^(\s*)(?:if\s+)?m\s+in\s+base_modules\b", re.MULTILINE)


def tasks_path() -> Path:
    try:
        import ultralytics.nn.tasks as ul_tasks
    except ImportError:
        sys.exit("ultralytics is not installed - run `pip install -r requirements.txt` first")
    return Path(ul_tasks.__file__)


def find_import_anchor(source: str) -> int:
    """Character offset just after the last top-level import line."""
    last = 0
    for match in re.finditer(r"^(?:from|import)\s+\S+.*$", source, re.MULTILINE):
        last = max(last, match.end())
    if last == 0:
        sys.exit("could not locate the import block in tasks.py - patch aborted")
    return last


def apply(path: Path) -> None:
    source = path.read_text(encoding="utf-8")
    if MARKER in source:
        print(f"already patched: {path}")
        return

    usage = USAGE_ANCHOR.search(source)
    if not usage:
        sys.exit(
            "could not find `m in base_modules` in parse_model - ultralytics has changed "
            "its internals. Update USAGE_ANCHOR in this script before continuing."
        )

    backup = path.with_suffix(path.suffix + ".tomanet-bak")
    if not backup.exists():
        shutil.copy2(path, backup)

    line_start = source.rfind("\n", 0, usage.start()) + 1
    indent = usage.group(1)
    patched = (
        source[: find_import_anchor(source)]
        + "\n"
        + IMPORT_BLOCK
        + source[find_import_anchor(source) : line_start]
        + USAGE_BLOCK.format(indent=indent)
        + source[line_start:]
    )

    path.write_text(patched, encoding="utf-8")
    print(f"patched:  {path}")
    print(f"backup:   {backup}")


def revert(path: Path) -> None:
    backup = path.with_suffix(path.suffix + ".tomanet-bak")
    if not backup.exists():
        sys.exit(f"no backup at {backup} - nothing to revert")
    shutil.copy2(backup, path)
    backup.unlink()
    print(f"reverted: {path}")


def check(path: Path) -> None:
    patched = MARKER in path.read_text(encoding="utf-8")
    print(f"{'PATCHED' if patched else 'NOT PATCHED'}: {path}")
    sys.exit(0 if patched else 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--check", action="store_true", help="report status, change nothing")
    group.add_argument("--revert", action="store_true", help="restore the original file")
    args = parser.parse_args()

    path = tasks_path()
    if args.check:
        check(path)
    elif args.revert:
        revert(path)
    else:
        apply(path)


if __name__ == "__main__":
    main()
