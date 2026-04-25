"""TOC extraction and page label resolution for PDF chapter detection."""

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# Fallback chapter heading patterns (used when PDF has no embedded TOC)
_CHAPTER_PATTERNS = [
    re.compile(r"^(?:Chapter|CHAPTER|Ch\.)\s+(\d+)", re.MULTILINE),
    re.compile(r"^第\s*[一二三四五六七八九十百千\d]+\s*章", re.MULTILINE),
    re.compile(r"^(?:Part|PART)\s+([IVXLCDM\d]+)", re.MULTILINE),
    re.compile(r"^#\s+", re.MULTILINE),
    re.compile(r"^##\s+", re.MULTILINE),
]


@dataclass
class Chapter:
    """A single chapter/section entry from the PDF table of contents.

    Attributes:
        level: TOC nesting level (1 = top-level chapter, 2 = section, etc.)
        title: Chapter title text.
        logical_page: Page number as recorded in the TOC (may use roman
            numerals, offset numbering, or be outright wrong).
        phys_index: 0-based physical page index resolved from *logical_page*:
            the actual page position within the PDF file.
    """
    level: int
    title: str
    logical_page: int
    phys_index: int = 0


class TocExtractor:
    """Extract and resolve the table of contents of a PDF document.

    Handles the common mismatch between *logical* page numbers (the numbers
    displayed and used in the TOC) and *physical* 0-based page indices.

    Resolution strategy (tried in order):
    1. ``page.get_label()`` — uses the PDF's /PageLabels structure.
    2. Heuristic text scan — calibrates by scanning candidate pages for
       known chapter titles.
    3. Naive ``logical - 1`` — fallback when neither of the above works.
    """

    def extract(self, doc) -> list[Chapter]:
        """Extract and fully resolve the TOC.

        Returns a list of *Chapter* entries with correct *phys_index*
        values, or an empty list if the PDF has no TOC at all.

        Args:
            doc: An opened ``fitz.Document`` (PyMuPDF) object.
        """
        toc = doc.get_toc()
        if not toc:
            logger.info("No embedded TOC found; scanning text for chapter headings.")
            return self._scan_chapters(doc)

        chapters = [Chapter(level, title.strip(), logical or 1)
                    for level, title, logical in toc]

        # Try label-based mapping first, then heuristic
        resolved = self._resolve_by_labels(doc, chapters)
        if resolved:
            return resolved

        resolved = self._resolve_by_heuristic(doc, chapters)
        if resolved:
            return resolved

        # Fallback: assume logical_page is 1-based with no offset
        logger.warning("Could not calibrate TOC pages; using logical-1 as physical index.")
        for ch in chapters:
            ch.phys_index = max(0, ch.logical_page - 1)
        return chapters

    # ── Label-based resolution ────────────────────────────────────────

    def _resolve_by_labels(self, doc, chapters: list[Chapter]) -> list[Chapter]:
        """Match TOC logical pages to physical indices via *get_label()."""
        # Build reverse map: displayed label text → physical page index
        label_to_phys = {}
        for i in range(doc.page_count):
            label = doc[i].get_label()
            if label:
                label_to_phys[str(label)] = i

        if not label_to_phys:
            logger.debug("PDF has no /PageLabels; skipping label-based resolution.")
            return []

        resolved: list[Chapter] = []
        misses = 0
        for ch in chapters:
            key = str(ch.logical_page)
            phys = label_to_phys.get(key)
            if phys is not None:
                resolved.append(Chapter(ch.level, ch.title, ch.logical_page, phys))
            else:
                misses += 1
                resolved.append(ch)  # keep temporarily for later fixup

        if misses == 0:
            logger.info("All TOC pages resolved via /PageLabels.")
            return resolved

        if misses == len(chapters):
            logger.debug("No TOC entries matched /PageLabels. Labels: %s",
                         list(label_to_phys.keys())[:10])
            return []

        # Partial match — fill in misses by interpolation
        hits = [(i, r) for i, r in enumerate(resolved) if r.phys_index != 0]
        for i, ch in enumerate(resolved):
            if ch.phys_index != 0:
                continue
            prev = next((h for h in reversed(hits) if h[0] < i), None)
            nxt = next((h for h in hits if h[0] > i), None)
            if prev and nxt:
                _, prev_ch = prev
                _, next_ch = nxt
                avg_step = (next_ch.phys_index - prev_ch.phys_index) / (next_ch.logical_page - prev_ch.logical_page)
                ch.phys_index = round(prev_ch.phys_index + (ch.logical_page - prev_ch.logical_page) * avg_step)
            elif prev:
                _, prev_ch = prev
                ch.phys_index = prev_ch.phys_index + (ch.logical_page - prev_ch.logical_page)
            elif nxt:
                _, next_ch = nxt
                ch.phys_index = next_ch.phys_index - (next_ch.logical_page - ch.logical_page)

        return resolved

    # ── Heuristic calibration ─────────────────────────────────────────

    def _resolve_by_heuristic(self, doc, chapters: list[Chapter]) -> list[Chapter]:
        """Scan candidate pages to find the offset between logical and physical pages.

        Only the first few top-level chapters are used for calibration.
        """
        top = [ch for ch in chapters if ch.level == 1][:3]
        if not top:
            return []

        search_range = min(30, doc.page_count)
        candidate_pages = list(range(search_range))

        for ch in top:
            title_keywords = self._title_keywords(ch.title)
            if not title_keywords:
                continue

            for phys in candidate_pages:
                text = self._page_preview(doc, phys)
                if any(kw in text for kw in title_keywords):
                    offset = phys - (ch.logical_page - 1)
                    logger.info("Heuristic calibration: offset=%d (title=%r at phys page %d)",
                                offset, ch.title[:40], phys)
                    result: list[Chapter] = []
                    for c in chapters:
                        result.append(Chapter(c.level, c.title, c.logical_page,
                                              max(0, c.logical_page - 1 + offset)))
                    return result

        logger.debug("Heuristic calibration failed; no chapter titles found in first %d pages.",
                     search_range)
        return []

    @staticmethod
    def _title_keywords(title: str) -> list[str]:
        """Break a chapter title into search keywords (3+ chars, skipping stopwords)."""
        stop = {"the", "a", "an", "of", "in", "to", "and", "is", "for", "on", "at", "by", "with"}
        words = re.findall(r"\w{3,}", title)
        return [w for w in words if w.lower() not in stop] or words[:3]

    @staticmethod
    def _page_preview(doc, phys_index: int, chars: int = 300) -> str:
        """Get the first *chars* characters from a physical page."""
        try:
            page = doc[phys_index]
            return page.get_text()[:chars]
        except Exception:
            return ""

    # ── No-TOC fallback: text scanning ────────────────────────────────

    def _scan_chapters(self, doc) -> list[Chapter]:
        """Scan every page for chapter-like headings when no TOC exists."""
        chapters: list[Chapter] = []
        for i in range(doc.page_count):
            try:
                text = doc[i].get_text()[:500]
            except Exception:
                continue
            for pattern in _CHAPTER_PATTERNS:
                match = pattern.search(text)
                if match:
                    title = match.group().strip()
                    # Avoid duplicates from adjacent page matches
                    if not chapters or chapters[-1].phys_index != i:
                        chapters.append(Chapter(1, title, i + 1, i))
                    break

        if chapters:
            logger.info("Found %d chapter heading(s) via text scan.", len(chapters))
        return chapters
