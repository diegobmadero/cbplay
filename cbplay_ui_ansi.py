from dataclasses import dataclass
import re

ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


@dataclass(frozen=True)
class AnsiTheme:
    active_color: str
    prev_color: str
    info_color: str
    reset_color: str
    box_tl: str
    box_tr: str
    box_bl: str
    box_br: str
    box_h: str
    box_v: str
    progress_full: str
    progress_empty: str


@dataclass(frozen=True)
class UiLayout:
    term_width: int
    display_width: int


def visible_len(text: str) -> int:
    """Length of text without ANSI codes."""
    return len(ANSI_ESCAPE_RE.sub("", text or ""))


def wrap_text_for_box(text: str, inner_width: int) -> list:
    """Wrap text to fit inside a box, preserving ANSI codes."""
    lines = []
    for line in (text or "").split("\n"):
        if not line:
            lines.append("")
            continue
        words = line.split(" ")
        current_line = ""
        for word in words:
            test_line = f"{current_line} {word}".strip() if current_line else word
            if visible_len(test_line) <= inner_width:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
                while visible_len(current_line) > inner_width:
                    lines.append(current_line[:inner_width])
                    current_line = current_line[inner_width:]
        if current_line:
            lines.append(current_line)
    return lines if lines else [""]


def hard_wrap_ansi(text: str, width: int):
    """Wrap a string by visible width while preserving ANSI codes."""
    if width <= 0:
        return [text or ""]
    lines = []
    buf = []
    visible = 0
    i = 0
    text = text or ""
    while i < len(text):
        ch = text[i]
        if ch == "\n":
            lines.append("".join(buf))
            buf = []
            visible = 0
            i += 1
            continue
        if ch == "\x1b":
            m = ANSI_ESCAPE_RE.match(text, i)
            if m:
                buf.append(m.group(0))
                i = m.end()
                continue
        buf.append(ch)
        visible += 1
        i += 1
        if visible >= width:
            lines.append("".join(buf))
            buf = []
            visible = 0
    lines.append("".join(buf))
    return lines


def build_highlight_spans(context_span, current_word_span, dim_span=None):
    """Build karaoke-style highlight spans (ANSI codes)."""
    spans = []
    if dim_span and isinstance(dim_span, (tuple, list)) and len(dim_span) == 2:
        try:
            start, end = int(dim_span[0]), int(dim_span[1])
        except Exception:
            start, end = None, None
        if start is not None and end is not None and 0 <= start < end:
            spans.append((start, end, "\033[38;5;242m", "\033[39m"))

    if context_span and isinstance(context_span, (tuple, list)) and len(context_span) == 2:
        try:
            start, end = int(context_span[0]), int(context_span[1])
        except Exception:
            start, end = None, None
        if start is not None and end is not None and 0 <= start < end:
            spans.append((start, end, "\033[4m", "\033[24m"))

    if current_word_span and isinstance(current_word_span, (tuple, list)) and len(current_word_span) == 2:
        try:
            start, end = int(current_word_span[0]), int(current_word_span[1])
        except Exception:
            start, end = None, None
        if start is not None and end is not None and 0 <= start < end:
            spans.append((start, end, "\033[1;7;96m", "\033[22;27;39m"))
    return spans if spans else None


class AnsiKaraokeRenderer:
    def __init__(self, theme: AnsiTheme, layout: UiLayout):
        self.theme = theme
        self.layout = layout

    def draw_box_top(self, title: str, width: int, color: str = "") -> str:
        reset = self.theme.reset_color if color else ""
        title_part = f" {title} " if title else self.theme.box_h
        remaining = width - 2 - visible_len(title_part)
        left_pad = 1
        right_pad = remaining - left_pad
        return (
            f"{color}{self.theme.box_tl}{self.theme.box_h * left_pad}"
            f"{title_part}{self.theme.box_h * max(0, right_pad)}{self.theme.box_tr}{reset}"
        )

    def draw_box_bottom(self, width: int, color: str = "") -> str:
        reset = self.theme.reset_color if color else ""
        return f"{color}{self.theme.box_bl}{self.theme.box_h * (width - 2)}{self.theme.box_br}{reset}"

    def draw_box_line(self, content: str, width: int, color: str = "", content_color: str = "") -> str:
        reset = self.theme.reset_color if color else ""
        content_reset = self.theme.reset_color if content_color else ""
        inner_width = width - 4
        vis_len = visible_len(content)
        padding = max(0, inner_width - vis_len)
        return (
            f"{color}{self.theme.box_v}{reset} {content_color}{content}"
            f"{content_reset}{' ' * padding} {color}{self.theme.box_v}{reset}"
        )

    def render_progress_bar(self, current: float, total: float, width: int = 20) -> str:
        if total <= 0:
            return self.theme.progress_empty * width
        ratio = min(1.0, max(0.0, current / total))
        filled = int(width * ratio)
        return self.theme.progress_full * filled + self.theme.progress_empty * (width - filled)

    def render_current_region(
        self,
        text: str,
        highlight_spans=None,
        paused: bool = False,
        debug_line: str = "",
        chunk_index: int = 0,
        total_chunks: int = 0,
        playhead: float = 0.0,
        duration: float = 0.0,
        status_icon: str = "",
    ) -> list:
        if total_chunks > 0:
            title = f"{status_icon} Now Playing [{chunk_index + 1}/{total_chunks}]"
        else:
            title = f"{status_icon} Now Playing"
        if paused:
            title += " PAUSED"

        box_width = min(self.layout.display_width, self.layout.term_width - 2)
        inner_width = box_width - 4

        rendered_text = text or ""
        if highlight_spans:
            events = {}
            for span in highlight_spans:
                if not span or len(span) != 4:
                    continue
                try:
                    start, end, start_code, end_code = span
                    start = int(start)
                    end = int(end)
                    start_code = str(start_code)
                    end_code = str(end_code)
                except Exception:
                    continue
                if start_code:
                    events.setdefault(start, []).append(start_code)
                if end_code:
                    events.setdefault(end, []).append(end_code)
            for idx in sorted(events.keys(), reverse=True):
                if 0 <= idx <= len(rendered_text):
                    rendered_text = rendered_text[:idx] + "".join(events[idx]) + rendered_text[idx:]

        lines = []
        box_color = self.theme.active_color
        lines.append(self.draw_box_top(title, box_width, box_color))
        wrapped_lines = wrap_text_for_box(rendered_text, inner_width)
        for line in wrapped_lines:
            lines.append(self.draw_box_line(line, box_width, box_color))

        if duration > 0:
            progress = self.render_progress_bar(playhead, duration, width=min(30, inner_width - 10))
            time_str = f"{playhead:.1f}s / {duration:.1f}s"
            progress_line = f"{progress}  {time_str}"
            lines.append(self.draw_box_line("", box_width, box_color))
            lines.append(self.draw_box_line(progress_line, box_width, box_color, self.theme.info_color))

        lines.append(self.draw_box_bottom(box_width, box_color))
        if debug_line:
            lines.append(f"{self.theme.info_color}{debug_line}{self.theme.reset_color}")
        return lines

    def render_previous_section(self, text: str, max_lines: int = 4) -> list:
        if not text:
            return []

        box_width = min(self.layout.display_width, self.layout.term_width - 2)
        inner_width = box_width - 4

        lines = []
        lines.append(self.draw_box_top("Previous", box_width, self.theme.prev_color))

        wrapped = wrap_text_for_box(text, inner_width)
        if len(wrapped) > max_lines:
            more = len(wrapped) - max_lines + 1
            wrapped = wrapped[: max_lines - 1] + [f"... ({more} more lines)"]

        for line in wrapped:
            lines.append(self.draw_box_line(line, box_width, self.theme.prev_color, self.theme.prev_color))

        lines.append(self.draw_box_bottom(box_width, self.theme.prev_color))
        return lines
