import curses
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class CursesLayout:
    header_rows: int = 2
    footer_rows: int = 1


@dataclass(frozen=True)
class WrappedChunk:
    text: str
    lines: List[str]
    line_ranges: List[Tuple[int, int]]


class CursesKaraokeScreen:
    def __init__(self, stdscr, layout: Optional[CursesLayout] = None, debug: bool = False):
        self.stdscr = stdscr
        self.layout = layout or CursesLayout()
        self.debug = bool(debug)
        self.term_height, self.term_width = self.stdscr.getmaxyx()
        self.body_height = max(1, self.term_height - self.layout.header_rows - self.layout.footer_rows)
        self.body_pad = None
        self.pad_height = 0
        self.pad_width = 0
        self.header_win = None
        self.footer_win = None
        self.colors: Dict[str, int] = {}
        self._init_curses()
        self._init_colors()
        self._rebuild_windows()

    def _init_curses(self):
        curses.curs_set(0)
        curses.noecho()
        curses.cbreak()
        self.stdscr.keypad(True)

    def _init_colors(self):
        if curses.has_colors():
            curses.start_color()
            curses.use_default_colors()
            try:
                curses.init_pair(1, curses.COLOR_CYAN, -1)
                curses.init_pair(2, curses.COLOR_WHITE, -1)
                curses.init_pair(3, curses.COLOR_GREEN, -1)
                
                self.colors = {
                    "current": curses.A_NORMAL,
                    "future": curses.A_NORMAL,
                    "past": curses.color_pair(2) | curses.A_DIM,
                    "current_word": curses.color_pair(1) | curses.A_REVERSE | curses.A_BOLD,
                    "spoken": curses.color_pair(2) | curses.A_DIM,
                    "info": curses.color_pair(3),
                    "header": curses.A_BOLD | curses.color_pair(3),
                }
            except Exception:
                 self.colors = {
                    "current": curses.A_NORMAL,
                    "future": curses.A_NORMAL,
                    "past": curses.A_DIM,
                    "current_word": curses.A_BOLD | curses.A_REVERSE,
                    "spoken": curses.A_DIM,
                    "info": curses.A_DIM,
                    "header": curses.A_BOLD,
                }
        else:
            self.colors = {
                "current": curses.A_NORMAL,
                "future": curses.A_NORMAL,
                "past": curses.A_DIM,
                "current_word": curses.A_BOLD | curses.A_REVERSE,
                "spoken": curses.A_DIM,
                "info": curses.A_DIM,
                "header": curses.A_BOLD,
            }

    def _rebuild_windows(self):
        self.term_height, self.term_width = self.stdscr.getmaxyx()
        self.body_height = max(1, self.term_height - self.layout.header_rows - self.layout.footer_rows)
        self.header_win = self.stdscr.derwin(self.layout.header_rows, self.term_width, 0, 0)
        self.footer_win = self.stdscr.derwin(
            self.layout.footer_rows, self.term_width, self.term_height - self.layout.footer_rows, 0
        )
        
        self.max_text_width = 90
        self.text_width = min(self.max_text_width, self.term_width - 4)
        self.margin_left = (self.term_width - self.text_width) // 2
        
        self._ensure_pad(self.body_height)

    def _ensure_pad(self, needed_lines: int):
        needed = max(needed_lines, self.body_height, 1)
        if self.body_pad is None or needed > self.pad_height or self.term_width != self.pad_width:
            self.pad_height = needed
            self.pad_width = self.term_width
            self.body_pad = curses.newpad(self.pad_height, self.pad_width)

    def handle_resize(self) -> bool:
        new_h, new_w = self.stdscr.getmaxyx()
        if (new_h, new_w) == (self.term_height, self.term_width):
            return False
        curses.resizeterm(new_h, new_w)
        self._rebuild_windows()
        return True

    def clear(self):
        self.stdscr.erase()
        if self.header_win is not None:
            self.header_win.erase()
        if self.footer_win is not None:
            self.footer_win.erase()
        if self.body_pad is not None:
            self.body_pad.erase()

    def _safe_addnstr(self, win, y: int, x: int, text: str, attr: int = 0):
        if win is None:
            return
        try:
            win.addnstr(y, x, text, max(0, self.term_width - x - 1), attr)
        except curses.error:
            pass

    def draw_header(self, lines: List[str]):
        if self.header_win is None:
            return
        self.header_win.erase()
        margin = getattr(self, "margin_left", 0)
        width = getattr(self, "text_width", self.term_width)
        
        for idx in range(min(len(lines), self.layout.header_rows)):
            text = lines[idx]
            if len(text) > width:
                text = text[:width]
            
            try:
                self.header_win.addstr(idx, margin, text, self.colors.get("info", 0))
            except curses.error:
                pass

    def draw_footer(self, text: str, attr: int = 0):
        if self.footer_win is None:
            return
        self.footer_win.erase()
        margin = getattr(self, "margin_left", 0)
        width = getattr(self, "text_width", self.term_width)
        
        if len(text) > width:
            text = text[:width]
            
        try:
            self.footer_win.addstr(0, margin, text, attr or self.colors.get("info", 0))
        except curses.error:
            pass

    def draw_pad_lines(self, lines: List[Tuple[str, int, int]]):
        self._ensure_pad(len(lines))
        if self.body_pad is None:
            return
        self.body_pad.erase()
        margin = getattr(self, "margin_left", 0)
        for y, (text, attr, indent) in enumerate(lines):
            self._safe_addnstr(self.body_pad, y, margin + indent, text, attr)

    def apply_span(
        self,
        chunk_start_line: int,
        line_ranges: List[Tuple[int, int]],
        span_start: Optional[int],
        span_end: Optional[int],
        lines: List[Tuple[str, int, int]],
        attr: int,
    ):
        if self.body_pad is None or not lines:
            return
        segments = self.span_to_segments(line_ranges, span_start, span_end)
        if not segments:
            return
        margin = getattr(self, "margin_left", 0)
        for line_no, col_start, col_end in segments:
            pad_line = chunk_start_line + line_no
            if pad_line < 0 or pad_line >= len(lines):
                continue
            line_text = lines[pad_line][0]
            indent = lines[pad_line][2]
            if not line_text or col_start >= len(line_text):
                continue
            end = min(col_end, len(line_text))
            if end <= col_start:
                continue
            self._safe_addnstr(self.body_pad, pad_line, margin + indent + col_start, line_text[col_start:end], attr)

    def refresh(self, pad_top: int = 0):
        if self.header_win is not None:
            self.header_win.noutrefresh()
        if self.footer_win is not None:
            self.footer_win.noutrefresh()
        if self.body_pad is not None:
            top = max(0, min(pad_top, max(0, self.pad_height - self.body_height)))
            self.body_pad.noutrefresh(
                top,
                0,
                self.layout.header_rows,
                0,
                self.layout.header_rows + self.body_height - 1,
                self.term_width - 1,
            )
        curses.doupdate()

    def wrap_text_basic(self, text: str, width: int) -> List[str]:
        lines: List[str] = []
        for raw_line in (text or "").split("\n"):
            if not raw_line:
                lines.append("")
                continue
            start = 0
            while start < len(raw_line):
                lines.append(raw_line[start : start + width])
                start += width
        return lines if lines else [""]

    def wrap_text_with_ranges(self, text: str, width: int) -> Tuple[List[str], List[Tuple[int, int]], List[int]]:
        """Wrap text while preserving exact indices, returning lines, ranges, and indents."""
        if width <= 0:
            width = 1
        text = text or ""
        if not text:
            return [""], [(0, 0)], [0]
        lines: List[str] = []
        ranges: List[Tuple[int, int]] = []
        indents: List[int] = []
        i = 0
        text_len = len(text)
        
        while i < text_len:
            current_hanging_indent = 0
            remaining_text = text[i:]
            match = re.match(r'^(\s*[•\-\*]|\d+\.)\s+', remaining_text)
            if match:
                current_hanging_indent = len(match.group(0))
                if current_hanging_indent > width // 2:
                    current_hanging_indent = 0
            
            first_line_of_logical = True
            while i < text_len:
                line_start = i
                
                effective_width = width
                if not first_line_of_logical:
                    effective_width = width - current_hanging_indent
                if effective_width < 1: 
                    effective_width = 1
                
                line_len = 0
                last_space_pos = None
                last_space_idx = None
                broke_on_newline = False
                
                while i < text_len:
                    ch = text[i]
                    if ch == "\n":
                        broke_on_newline = True
                        break
                    line_len += 1
                    if ch.isspace():
                        last_space_pos = line_len - 1
                        last_space_idx = i
                    i += 1
                    if line_len >= effective_width:
                        break

                line_end = i
                if i < text_len and (not broke_on_newline) and line_len >= effective_width:
                    if last_space_pos is not None and last_space_pos != 0:
                        rollback = line_end - (last_space_idx + 1)
                        if rollback > 0:
                            i -= rollback
                            line_end -= rollback
                            line_len -= rollback

                lines.append(text[line_start:line_end])
                ranges.append((line_start, line_end))
                indents.append(current_hanging_indent if not first_line_of_logical else 0)
                
                first_line_of_logical = False

                if i < text_len and text[i] == "\n":
                    i += 1
                    break

        return lines if lines else [""], ranges if ranges else [(0, 0)], indents if indents else [0]

    @staticmethod
    def span_to_segments(
        line_ranges: List[Tuple[int, int]],
        span_start: Optional[int],
        span_end: Optional[int],
    ) -> List[Tuple[int, int, int]]:
        """Map a [start,end) span into (line, col_start, col_end) segments."""
        segments: List[Tuple[int, int, int]] = []
        if span_start is None or span_end is None:
            return segments
        try:
            start = int(span_start)
            end = int(span_end)
        except Exception:
            return segments
        if end <= start:
            return segments
        for line_no, (line_start, line_end) in enumerate(line_ranges):
            if line_end <= start or line_start >= end:
                continue
            seg_start = max(start, line_start) - line_start
            seg_end = min(end, line_end) - line_start
            if seg_end > seg_start:
                segments.append((line_no, seg_start, seg_end))
        return segments

    def build_fullpage_lines(
        self,
        chunks: List[str],
        current_index: int,
    ) -> Tuple[List[Tuple[str, int, int]], List[Tuple[int, int]]]:
        lines, ranges, _ = self.build_fullpage_lines_with_ranges(chunks, current_index)
        return lines, ranges

    def build_fullpage_lines_with_ranges(
        self,
        chunks: List[str],
        current_index: int,
    ) -> Tuple[
        List[Tuple[str, int, int]],
        List[Tuple[int, int]],
        List[List[Tuple[int, int]]],
    ]:
        lines: List[Tuple[str, int, int]] = []
        ranges: List[Tuple[int, int]] = []
        line_ranges_by_chunk: List[List[Tuple[int, int]]] = []
        width = getattr(self, "text_width", max(1, self.term_width - 1))
        
        for idx, chunk in enumerate(chunks):
            start_line = len(lines)
            attr = self.colors.get("past", 0) if idx < current_index else self.colors.get("future", 0)
            if idx == current_index:
                attr = self.colors.get("current", 0)
            
            is_header = False
            stripped = chunk.strip()
            if len(stripped) < 60 and "\n" not in stripped:
                if stripped.endswith(":") or (stripped.isupper() and len(stripped) > 4):
                     is_header = True
            
            if is_header:
                attr = self.colors.get("header", 0)
            
            wrapped_lines, line_ranges, indents = self.wrap_text_with_ranges(chunk, width)
            for i, line in enumerate(wrapped_lines):
                indent = indents[i] if i < len(indents) else 0
                lines.append((line, attr, indent))
            
            ranges.append((start_line, len(lines)))
            line_ranges_by_chunk.append(line_ranges)
            lines.append(("", attr, 0))
        return lines, ranges, line_ranges_by_chunk
