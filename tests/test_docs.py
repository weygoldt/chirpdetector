"""The Markdown documentation's relative links resolve.

Every link and image in ``README.md`` and ``docs/*.md`` that is not a URL
(Markdown links and images, and the ``src``/``srcset`` of HTML tags such as
``<picture>``) must point to an existing file or directory of the repository, and a
``#fragment`` must name a heading of the target page (GitHub's anchors).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PAGES = sorted([ROOT / "README.md", *(ROOT / "docs").glob("*.md")])
LINK = re.compile(r"!?\[[^\]]*\]\(<?([^)\s>]+)>?(?:\s+\"[^\"]*\")?\)")
HTML_SRC = re.compile(r"\b(?:src|srcset)=\"([^\"\s]+)")
FENCE = re.compile(r"^(```|~~~)")


def _prose(text: str) -> list[str]:
    """The page's lines outside fenced code blocks, inline code removed."""
    lines, fenced = [], False
    for line in text.splitlines():
        if FENCE.match(line.lstrip()):
            fenced = not fenced
            continue
        if not fenced:
            lines.append(re.sub(r"`[^`]*`", "", line))
    return lines


def anchors(page: Path) -> set[str]:
    """GitHub's heading anchors of a Markdown page."""
    found: set[str] = set()
    counts: dict[str, int] = {}
    fenced = False
    for line in page.read_text().splitlines():
        if FENCE.match(line.lstrip()):
            fenced = not fenced
            continue
        match = re.match(r"^#{1,6}\s+(.*?)\s*#*\s*$", line)
        if fenced or not match:
            continue
        text = re.sub(r"[`*_]|\[([^\]]*)\]\([^)]*\)", r"\1", match.group(1))
        slug = re.sub(r"[^\w\- ]", "", text.strip().lower()).replace(" ", "-")
        n = counts.get(slug, 0)
        counts[slug] = n + 1
        found.add(slug if n == 0 else f"{slug}-{n}")
    return found


def links(page: Path) -> list[str]:
    """Relative link targets of a page (URLs and mail links excluded)."""
    targets = []
    for line in _prose(page.read_text()):
        for target in [*LINK.findall(line), *HTML_SRC.findall(line)]:
            if not re.match(r"^[a-z][a-z0-9+.-]*:", target):
                targets.append(target)
    return targets


def test_there_are_pages() -> None:
    assert len(PAGES) > 1
    assert (ROOT / "docs" / "README.md").exists()


@pytest.mark.parametrize("page", PAGES, ids=lambda p: str(p.relative_to(ROOT)))
def test_relative_links_resolve(page: Path) -> None:
    broken = []
    for target in links(page):
        path, _, fragment = target.partition("#")
        resolved = (page.parent / path).resolve() if path else page
        if not resolved.exists():
            broken.append(f"{target}: no such file")
        elif (
            fragment and resolved.suffix == ".md" and fragment not in anchors(resolved)
        ):
            broken.append(f"{target}: no heading #{fragment}")
    assert not broken, f"{page.relative_to(ROOT)}: " + "; ".join(broken)
