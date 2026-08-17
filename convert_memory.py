#!/usr/bin/env python3
"""
Convert project memory files to Claude Code persistent memory format.
Reads from /workspace/Video_Enhancement/memory/*.md
Writes to /root/.claude/projects/-workspace-Video-Enhancement/memory/
"""
import os
import re
import yaml
from pathlib import Path

SRC_DIR = Path("/workspace/Video_Enhancement/memory")
DST_DIR = Path("/root/.claude/projects/-workspace-Video-Enhancement/memory")
MEMORY_INDEX_PATH = DST_DIR / "MEMORY.md"

# Files to skip
SKIP_FILES = {"MEMORY.md"}

def parse_frontmatter(content: str):
    """Parse YAML frontmatter from markdown content."""
    if not content.startswith("---"):
        return {}, content

    parts = content.split("---", 2)
    if len(parts) < 3:
        return {}, content

    frontmatter_str = parts[1].strip()
    body = parts[2].strip()

    try:
        frontmatter = yaml.safe_load(frontmatter_str) or {}
    except yaml.YAMLError:
        frontmatter = {}

    return frontmatter, body

def normalize_frontmatter(frontmatter: dict, filename: str, body: str) -> dict:
    """Normalize frontmatter to standard Claude Code memory format."""
    result = {}

    # name: use existing or derive from filename
    name = frontmatter.get("name", "")
    if not name:
        name = filename.replace(".md", "").replace("_", "-")
    result["name"] = name

    # description
    desc = frontmatter.get("description", "")
    if not desc:
        # Take first non-empty line of body as description
        for line in body.split("\n"):
            line = line.strip().lstrip("#").strip()
            if line and not line.startswith("[") and not line.startswith("!"):
                desc = line[:200]
                break
    result["description"] = desc

    # metadata.type
    metadata = frontmatter.get("metadata", {})
    if isinstance(metadata, dict):
        mem_type = metadata.get("type", "")
    else:
        mem_type = ""

    if not mem_type:
        mem_type = frontmatter.get("type", "project")

    # Preserve other metadata fields
    result_metadata = {"type": mem_type}
    if isinstance(metadata, dict):
        for k, v in metadata.items():
            if k != "type" and not k.startswith("origin"):
                result_metadata[k] = v

    result["metadata"] = result_metadata

    return result

def format_frontmatter(fm: dict) -> str:
    """Format frontmatter dict back to YAML string."""
    # Use custom formatting for clean output
    lines = ["---"]
    lines.append(f"name: {fm['name']}")
    lines.append(f"description: {fm['description']}")
    lines.append("metadata:")
    for k, v in fm["metadata"].items():
        if isinstance(v, str):
            lines.append(f"  {k}: {v}")
        elif isinstance(v, bool):
            lines.append(f"  {k}: {'true' if v else 'false'}")
        elif isinstance(v, (int, float)):
            lines.append(f"  {k}: {v}")
    lines.append("---")
    return "\n".join(lines)

def main():
    os.makedirs(DST_DIR, exist_ok=True)

    files = sorted([f for f in SRC_DIR.glob("*.md") if f.name not in SKIP_FILES])

    index_entries = []
    converted = 0
    errors = []

    for filepath in files:
        try:
            content = filepath.read_text(encoding="utf-8")
            frontmatter, body = parse_frontmatter(content)
            normalized_fm = normalize_frontmatter(frontmatter, filepath.name, body)

            # Build output content
            fm_str = format_frontmatter(normalized_fm)
            output = f"{fm_str}\n\n{body}\n"

            # Write to destination
            dest_path = DST_DIR / filepath.name
            dest_path.write_text(output, encoding="utf-8")

            # Build index entry
            name = normalized_fm["name"]
            desc = normalized_fm["description"]
            index_entries.append((filepath.name, name, desc))
            converted += 1
            print(f"  ✅ {filepath.name}")

        except Exception as e:
            errors.append((filepath.name, str(e)))
            print(f"  ❌ {filepath.name}: {e}")

    # Write MEMORY.md index
    index_lines = []
    for filename, name, desc in index_entries:
        # Extract first part of description for hook
        hook = desc[:80] + ("..." if len(desc) > 80 else "")
        index_lines.append(f"- [{name}]({filename}) — {hook}")

    index_content = "\n".join(index_lines) + "\n"
    MEMORY_INDEX_PATH.write_text(index_content, encoding="utf-8")

    print(f"\n📊 Summary: {converted}/{len(files)} converted, {len(errors)} errors")
    print(f"📄 Index written to: {MEMORY_INDEX_PATH}")

if __name__ == "__main__":
    main()
