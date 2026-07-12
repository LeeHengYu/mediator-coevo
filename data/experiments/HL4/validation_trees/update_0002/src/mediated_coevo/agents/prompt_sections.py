"""Helpers for deriving runtime prompt sections from Markdown skills."""

from __future__ import annotations


def select_markdown_sections(markdown: str, headings: list[str]) -> str | None:
    """Return requested level-two Markdown sections in their original order."""
    selected_headings = set(headings)
    selected_sections: list[str] = []
    current_heading: str | None = None
    current_lines: list[str] = []

    def flush_current_section() -> None:
        if current_heading in selected_headings and current_lines:
            section = "\n".join(current_lines).strip()
            if section:
                selected_sections.append(section)

    for line in markdown.splitlines():
        if line.startswith("## "):
            flush_current_section()
            current_heading = line[3:].strip()
            current_lines = [line]
            continue
        if current_heading is not None:
            current_lines.append(line)
    flush_current_section()

    if not selected_sections:
        return None
    return "\n\n".join(selected_sections)


def select_markdown_section(markdown: str, heading: str) -> str | None:
    """Return one level-two Markdown section, including its heading."""
    current_heading: str | None = None
    current_lines: list[str] = []

    for line in markdown.splitlines():
        if line.startswith("## "):
            if current_heading == heading and current_lines:
                return "\n".join(current_lines).strip()
            current_heading = line[3:].strip()
            current_lines = [line]
            continue
        if current_heading is not None:
            current_lines.append(line)

    if current_heading == heading and current_lines:
        return "\n".join(current_lines).strip()
    return None
