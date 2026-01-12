# -*- coding: utf-8 -*-
# --------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------

"""
MkDocs build-time generator for PTO Tile Lib.

We intentionally keep MkDocs config under `docs/mkdocs/` and generate a *mirror*
of repository markdown into `docs/mkdocs/src/` so the site can browse markdown
across the entire repo (README files under kernels/, tests/, scripts/, etc.).

Key property:
- Generated pages preserve original repository paths, so existing repo-relative
  links like `docs/...` or `kernels/...` keep working in the site.
"""

from __future__ import annotations

from pathlib import Path

import mkdocs_gen_files


REPO_ROOT = Path(__file__).resolve().parents[2]

SKIP_PREFIXES = (
    ".git/",
    ".github/",
    ".gitcode/",
    ".venv/",
    ".venv-mkdocs/",
    "site/",
    "build/",
    "build_tests/",
    ".idea/",
    ".vscode/",
)

SKIP_CONTAINS = (
    "/__pycache__/",
    "/CMakeFiles/",
)

ASSET_EXTS = {
    ".svg",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
}


def _should_skip(rel_posix: str) -> bool:
    if rel_posix.startswith("docs/mkdocs/"):
        return True
    if rel_posix.endswith("/mkdocs.yml"):
        return True
    if rel_posix.startswith(".venv"):
        return True
    if "site-packages/" in rel_posix:
        return True
    if any(rel_posix.startswith(p) for p in SKIP_PREFIXES):
        return True
    if any(s in rel_posix for s in SKIP_CONTAINS):
        return True
    if rel_posix.endswith((".pyc",)):
        return True
    return False


def main() -> None:
    copied_md: list[str] = []

    # Mirror markdown files into the MkDocs virtual filesystem, preserving paths.
    for src in REPO_ROOT.rglob("*.md"):
        rel = src.relative_to(REPO_ROOT).as_posix()
        if _should_skip(rel):
            continue
        text = src.read_text(encoding="utf-8", errors="replace")
        with mkdocs_gen_files.open(rel, "w") as f:
            f.write(f"<!-- Generated from `{rel}` -->\n\n")
            f.write(text)
        copied_md.append(rel)

    # Generate a per-instruction reference index for docs/isa/*.md.
    isa_dir = REPO_ROOT / "docs" / "isa"
    isa_pages: list[tuple[str, str]] = []

    def extract_first_heading(md_path: Path) -> str:
        try:
            text = md_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return md_path.stem
        for line in text.splitlines():
            if line.startswith("#"):
                return line.lstrip("#").strip()
        return md_path.stem

    if isa_dir.exists():
        for p in sorted(isa_dir.glob("*.md")):
            if p.name in ("README.md", "conventions.md"):
                continue
            instr = p.stem
            title = extract_first_heading(p)
            isa_pages.append((instr, title))

    with mkdocs_gen_files.open("manual/isa-reference.md", "w") as f:
        f.write("# Instruction Reference Pages\n\n")
        f.write("This page is generated at build time.\n\n")
        f.write("- Instruction index: `docs/isa/README.md`\n")
        f.write("- ISA conventions: `docs/isa/conventions.md`\n\n")
        if not isa_pages:
            f.write("No instruction pages were found under `docs/isa/`.\n")
        else:
            f.write("## All instructions\n\n")
            for instr, title in isa_pages:
                # This page lives under `manual/`, so use `../` to link back to root-level docs.
                link = f"../docs/isa/{instr}.md"
                suffix = "" if title.strip() == instr else f" — {title}"
                f.write(f"- [{instr}]({link}){suffix}\n")
            f.write("\n")

    # Generate a simple index page that links to all mirrored markdown.
    copied_md = sorted(set(copied_md))
    sections: dict[str, list[str]] = {}
    for rel in copied_md:
        top = rel.split("/", 1)[0] if "/" in rel else "(root)"
        sections.setdefault(top, []).append(rel)

    with mkdocs_gen_files.open("all-pages.md", "w") as f:
        f.write("# All Markdown Pages\n\n")
        f.write("This page is generated at build time and lists markdown files mirrored into the site.\n\n")
        for top in sorted(sections.keys()):
            f.write(f"## {top}\n\n")
            for rel in sections[top]:
                # Prefer a short label but keep it unambiguous.
                label = rel if top == "(root)" else rel[len(top) + 1 :]
                f.write(f"- [{label}]({rel})\n")
            f.write("\n")

    # Mirror commonly referenced doc assets (images) so docs render cleanly.
    for src in REPO_ROOT.rglob("*"):
        if not src.is_file():
            continue
        if src.suffix.lower() not in ASSET_EXTS:
            continue
        rel = src.relative_to(REPO_ROOT).as_posix()
        if _should_skip(rel):
            continue
        with mkdocs_gen_files.open(rel, "wb") as f:
            f.write(src.read_bytes())


main()
