from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import html
import re
from typing import Dict, Iterable, List, Optional

try:
    from bs4 import BeautifulSoup
except ImportError:  # pragma: no cover
    BeautifulSoup = None  # type: ignore

BOOK_SOURCE_FILES = [
    "index.html",
    "guidebook.html",
    "note_from_author.html",
    "measurement_registers.html",
    "chapter_1.html",
    "chapter_2.html",
    "chapter_3.html",
    "chapter_4.html",
    "quantitative_appendix.html",
    "references.html",
]

BASE_DIR = Path(__file__).resolve().parent
BOOK_CONTEXT_DIR = BASE_DIR / "book_context"

HEADING_TAGS = {"h1", "h2", "h3", "h4"}
TEXT_TAGS = {
    "p",
    "li",
    "figcaption",
    "td",
    "th",
    "summary",
    "blockquote",
    "pre",
}


@dataclass(frozen=True)
class BookChunk:
    chunk_id: str
    source_file: str
    page_title: str
    section_title: str
    anchor_id: str
    chapter: str
    content_type: str
    content: str


LAST_LOAD_STATUS: Dict[str, object] = {
    "base_dir": str(BASE_DIR),
    "book_context_dir": str(BOOK_CONTEXT_DIR),
    "source_files_found": 0,
    "filenames_found": [],
    "cleaned_char_count_by_file": {},
    "chunks_created_by_file": {},
    "chunks_loaded": 0,
    "error": "",
}


@dataclass
class _SectionBuffer:
    source_file: str
    page_title: str
    section_title: str
    anchor_id: str
    lines: List[str]

    def materialize(self) -> str:
        merged = "\n".join(line for line in self.lines if line)
        return _normalize_whitespace(merged)


def _normalize_whitespace(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _detect_content_type(section_title: str, text: str) -> str:
    title_lower = section_title.lower()
    text_lower = text.lower()
    if "equation" in title_lower or "equation" in text_lower or "\\(" in text:
        return "equation"
    if "table" in title_lower:
        return "table"
    if "figure" in title_lower or "fig." in text_lower:
        return "figure_caption"
    if "reference" in title_lower:
        return "reference"
    return "text"


def _chapter_label(source_file: str, page_title: str) -> str:
    stem = Path(source_file).stem
    if stem.startswith("chapter_"):
        return stem.replace("_", " ").title()
    return page_title or stem.replace("_", " ").title()


def _chunk_section_text(text: str, target_chars: int = 1200, overlap_chars: int = 180) -> List[str]:
    if len(text) <= target_chars:
        return [text]

    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks: List[str] = []
    current = ""

    for para in paragraphs:
        if not current:
            current = para
            continue

        candidate = f"{current}\n{para}"
        if len(candidate) <= target_chars:
            current = candidate
            continue

        chunks.append(current)
        tail = current[-overlap_chars:].strip()
        current = f"{tail}\n{para}" if tail else para

    if current:
        chunks.append(current)

    return chunks


def _clean_text_block(text: str) -> str:
    return _normalize_whitespace(html.unescape(text))


def _extract_sections_from_html(file_path: Path) -> tuple[List[_SectionBuffer], str]:
    raw = file_path.read_text(encoding="utf-8", errors="ignore")
    if BeautifulSoup is None:
        return _extract_sections_without_bs4(file_path, raw)

    soup = BeautifulSoup(raw, "html.parser")

    for removable in soup.find_all(["header", "nav", "footer", "script", "style", "noscript", "aside"]):
        removable.decompose()

    main = soup.find("main") or soup.body or soup
    page_title = _normalize_whitespace(soup.title.get_text(" ", strip=True) if soup.title else file_path.stem)
    cleaned_page_text = _clean_text_block(main.get_text(" ", strip=True))

    sections: List[_SectionBuffer] = []
    current = _SectionBuffer(
        source_file=file_path.name,
        page_title=page_title,
        section_title="Overview",
        anchor_id="top",
        lines=[],
    )

    for node in main.descendants:
        if getattr(node, "name", None) in HEADING_TAGS:
            if current.lines:
                sections.append(current)
            heading_text = _normalize_whitespace(node.get_text(" ", strip=True))
            heading_text = heading_text or "Untitled Section"
            current = _SectionBuffer(
                source_file=file_path.name,
                page_title=page_title,
                section_title=heading_text,
                anchor_id=node.get("id") or heading_text.lower().replace(" ", "-")[:80],
                lines=[],
            )
            continue

        if getattr(node, "name", None) in TEXT_TAGS:
            line = _normalize_whitespace(node.get_text(" ", strip=True))
            if len(line) >= 20:
                current.lines.append(line)

    if current.lines:
        sections.append(current)

    if not sections and cleaned_page_text:
        sections = [
            _SectionBuffer(
                source_file=file_path.name,
                page_title=page_title,
                section_title=page_title,
                anchor_id="top",
                lines=[cleaned_page_text],
            )
        ]

    return sections, cleaned_page_text


def _extract_sections_without_bs4(file_path: Path, raw: str) -> tuple[List[_SectionBuffer], str]:
    title_match = re.search(r"<title[^>]*>(.*?)</title>", raw, flags=re.IGNORECASE | re.DOTALL)
    title_raw = title_match.group(1) if title_match else file_path.stem
    page_title = _normalize_whitespace(html.unescape(_strip_html_tags(title_raw)))
    cleaned_page_text = _normalize_whitespace(
        html.unescape(
            _strip_html_tags(
                re.sub(
                    r"(?is)<(script|style|nav|noscript|header|footer|aside)[^>]*>.*?</\1>",
                    " ",
                    raw,
                )
            )
        )
    )

    sections: List[_SectionBuffer] = []
    current = _SectionBuffer(
        source_file=file_path.name,
        page_title=page_title,
        section_title="Overview",
        anchor_id="top",
        lines=[],
    )

    pattern = re.compile(
        r"<(h[1-4]|p|li|figcaption|td|th|summary|blockquote|pre)\b([^>]*)>(.*?)</\1>",
        flags=re.IGNORECASE | re.DOTALL,
    )

    for match in pattern.finditer(raw):
        tag = match.group(1).lower()
        attrs = match.group(2) or ""
        inner = match.group(3) or ""
        text = _normalize_whitespace(html.unescape(_strip_html_tags(inner)))

        if tag in HEADING_TAGS:
            if current.lines:
                sections.append(current)
            heading_text = text or "Untitled Section"
            anchor_id = _extract_attr(attrs, "id") or heading_text.lower().replace(" ", "-")[:80]
            current = _SectionBuffer(
                source_file=file_path.name,
                page_title=page_title,
                section_title=heading_text,
                anchor_id=anchor_id,
                lines=[],
            )
            continue

        if len(text) >= 20:
            current.lines.append(text)

    if current.lines:
        sections.append(current)
    if not sections and cleaned_page_text:
        sections = [
            _SectionBuffer(
                source_file=file_path.name,
                page_title=page_title,
                section_title=page_title,
                anchor_id="top",
                lines=[cleaned_page_text],
            )
        ]
    return sections, cleaned_page_text


def _extract_attr(attrs: str, name: str) -> str:
    pattern = rf'{name}\s*=\s*["\']([^"\']+)["\']'
    match = re.search(pattern, attrs, flags=re.IGNORECASE)
    return match.group(1).strip() if match else ""


def _strip_html_tags(text: str) -> str:
    return re.sub(r"<[^>]+>", " ", text, flags=re.DOTALL)


def _resolve_candidate_file(base_dir: Path, source_name: str) -> Optional[Path]:
    first = base_dir / "book_context" / source_name
    if first.exists():
        return first
    fallback = base_dir / source_name
    if fallback.exists():
        return fallback
    return None


def get_last_load_status() -> Dict[str, object]:
    return dict(LAST_LOAD_STATUS)


def load_book_chunks(base_dir: Optional[Path] = None, source_files: Optional[Iterable[str]] = None) -> List[BookChunk]:
    resolved_base = (base_dir or BASE_DIR).resolve()
    files = list(source_files or BOOK_SOURCE_FILES)
    all_chunks: List[BookChunk] = []
    found_file_paths: List[Path] = []

    LAST_LOAD_STATUS.update(
        {
            "base_dir": str(resolved_base),
            "book_context_dir": str((resolved_base / "book_context").resolve()),
            "source_files_found": 0,
            "filenames_found": [],
            "cleaned_char_count_by_file": {},
            "chunks_created_by_file": {},
            "chunks_loaded": 0,
            "error": "",
        }
    )

    for source_name in files:
        file_path = _resolve_candidate_file(resolved_base, source_name)
        if file_path is None:
            continue
        found_file_paths.append(file_path)

        sections, cleaned_text = _extract_sections_from_html(file_path)
        file_chunk_count = 0
        file_cleaned_chars = len(cleaned_text)

        for section_idx, section in enumerate(sections, start=1):
            section_text = section.materialize()
            if not section_text:
                continue

            split_chunks = _chunk_section_text(section_text)
            for split_idx, split_text in enumerate(split_chunks, start=1):
                chapter = _chapter_label(section.source_file, section.page_title)
                content_type = _detect_content_type(section.section_title, split_text)
                chunk_id = f"{section.source_file}:{section_idx}:{split_idx}"
                all_chunks.append(
                    BookChunk(
                        chunk_id=chunk_id,
                        source_file=section.source_file,
                        page_title=section.page_title,
                        section_title=section.section_title,
                        anchor_id=section.anchor_id,
                        chapter=chapter,
                        content_type=content_type,
                        content=split_text,
                    )
                )
                file_chunk_count += 1

        if file_chunk_count == 0 and cleaned_text:
            fallback_chunks = _chunk_section_text(cleaned_text)
            for split_idx, split_text in enumerate(fallback_chunks, start=1):
                chunk_id = f"{file_path.name}:fallback:{split_idx}"
                all_chunks.append(
                    BookChunk(
                        chunk_id=chunk_id,
                        source_file=file_path.name,
                        page_title=file_path.stem.replace("_", " ").title(),
                        section_title=file_path.stem.replace("_", " ").title(),
                        anchor_id="top",
                        chapter=file_path.stem.replace("_", " ").title(),
                        content_type=_detect_content_type(file_path.stem, split_text),
                        content=split_text,
                    )
                )
                file_chunk_count += 1

        cleaned_by_file = LAST_LOAD_STATUS.get("cleaned_char_count_by_file")
        chunks_by_file = LAST_LOAD_STATUS.get("chunks_created_by_file")
        if isinstance(cleaned_by_file, dict):
            cleaned_by_file[file_path.name] = file_cleaned_chars
        if isinstance(chunks_by_file, dict):
            chunks_by_file[file_path.name] = file_chunk_count

    LAST_LOAD_STATUS["source_files_found"] = len(found_file_paths)
    LAST_LOAD_STATUS["filenames_found"] = [path.name for path in found_file_paths]
    LAST_LOAD_STATUS["chunks_loaded"] = len(all_chunks)

    if not found_file_paths:
        message = (
            f"No book source files were found. Checked "
            f"'{(resolved_base / 'book_context').resolve()}' first, then '{resolved_base}'."
        )
        LAST_LOAD_STATUS["error"] = message
        raise RuntimeError(message)

    if not all_chunks:
        message = (
            f"Book source files were found ({len(found_file_paths)}), but 0 chunks were extracted. "
            "Check parser compatibility and source file contents."
        )
        LAST_LOAD_STATUS["error"] = message
        raise RuntimeError(message)

    return all_chunks
