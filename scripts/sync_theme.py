#!/usr/bin/env python3
"""Synchronize canonical SeroVerse Zensical theme components to sibling repositories.

Copies `docs/stylesheets/extra.css`, `docs/overrides/main.html`,
`docs/overrides/partials/comments.html`, `scripts/fetch_announcement.py`, and
the pillar-specific `docs/assets/favicon.svg` from `SeroVerse` into
`../SeroRef`, `../KBank`, `../SeroGen`, and `../SeroEpi`.
"""

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CANONICAL_FILES = (
    Path("docs/stylesheets/extra.css"),
    Path("docs/overrides/main.html"),
    Path("docs/overrides/partials/comments.html"),
    Path("scripts/fetch_announcement.py"),
)
PILLAR_FAVICONS = {
    "seroref": Path("docs/assets/favicon-seroref.svg"),
    "kbank": Path("docs/assets/favicon-kbank.svg"),
    "serogen": Path("docs/assets/favicon-serogen.svg"),
    "seroepi": Path("docs/assets/favicon-seroepi.svg"),
}
DEFAULT_TARGETS = (
    ROOT.parent / "SeroRef",
    ROOT.parent / "KBank",
    ROOT.parent / "SeroGen",
    ROOT.parent / "SeroEpi",
)


def _sync_pair(src: Path, dst: Path, label: str, *, check_only: bool) -> str:
    src_bytes = src.read_bytes()
    dst_bytes = dst.read_bytes() if dst.exists() else None

    if dst_bytes == src_bytes:
        return f"[OK]   {label} is up to date"
    if check_only:
        return f"[DIFF] {label} differs from canonical"

    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return f"[SYNC] {label} updated"


def sync_target(target_repo: Path, *, check_only: bool = False) -> list[str]:
    """Sync canonical theme files into a target repository."""
    messages: list[str] = []
    if not target_repo.is_dir():
        messages.append(f"[SKIP] {target_repo} does not exist")
        return messages

    for rel_path in CANONICAL_FILES:
        src = ROOT / rel_path
        dst = target_repo / rel_path
        messages.append(_sync_pair(src, dst, f"{target_repo.name}/{rel_path}", check_only=check_only))

    pillar_key = target_repo.name.lower()
    favicon_src_rel = PILLAR_FAVICONS.get(pillar_key)
    if favicon_src_rel is not None:
        src = ROOT / favicon_src_rel
        dst = target_repo / "docs/assets/favicon.svg"
        if src.exists():
            messages.append(_sync_pair(src, dst, f"{target_repo.name}/docs/assets/favicon.svg", check_only=check_only))

    return messages


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify that sibling repositories match canonical theme files without modifying them.",
    )
    parser.add_argument(
        "targets",
        nargs="*",
        type=Path,
        default=list(DEFAULT_TARGETS),
        help="Target repository paths (defaults to ../SeroRef, ../KBank, ../SeroGen, ../SeroEpi).",
    )
    args = parser.parse_args()

    has_diff = False
    for target in args.targets:
        for line in sync_target(target.resolve(), check_only=args.check):
            print(line)
            if line.startswith("[DIFF]"):
                has_diff = True

    return 1 if (args.check and has_diff) else 0


if __name__ == "__main__":
    raise SystemExit(main())
