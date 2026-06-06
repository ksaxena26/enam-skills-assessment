"""Export the Claude conversation JSONL to a styled PDF."""
import json
import re
import textwrap
from datetime import datetime, timezone
from pathlib import Path

from fpdf import FPDF
from fpdf.enums import XPos, YPos
import matplotlib.font_manager as _fm

# Resolve DejaVu font paths from matplotlib's bundled fonts
def _find_font(name: str) -> str:
    for f in _fm.fontManager.ttflist:
        if name.lower() in f.fname.lower():
            return f.fname
    raise FileNotFoundError(f"Font not found: {name}")

_FONT_DIR = Path(_find_font("DejaVuSans.ttf")).parent
_SANS         = str(_FONT_DIR / "DejaVuSans.ttf")
_SANS_BOLD    = str(_FONT_DIR / "DejaVuSans-Bold.ttf")
_SANS_OBLIQUE = str(_FONT_DIR / "DejaVuSans-Oblique.ttf")
_MONO         = str(_FONT_DIR / "DejaVuSansMono.ttf")

JSONL_PATH = (
    Path(r"C:\Users\Kushagra Saxena\.claude\projects"
         r"\C--Users-Kushagra-Saxena-Documents-enam-skills-assessment"
         r"\7cd1235d-79e3-4f1b-99fc-8211295569e5.jsonl")
)
OUT_DIR = Path(__file__).parent
SESSION_LABEL = "enam-skills-assessment — engineering.py session"

# ── Colour palette ────────────────────────────────────────────────────────────
C_USER_BG   = (230, 242, 255)   # light blue
C_ASST_BG   = (240, 250, 240)   # light green
C_CODE_BG   = (245, 245, 245)   # light grey
C_USER_HDR  = (30,  90, 160)
C_ASST_HDR  = (30, 130,  60)
C_TEXT      = (30,  30,  30)
C_CODE_TEXT = (50,  50,  50)


def _extract_text(content) -> str:
    """Pull plain text out of a message content (str or list of blocks)."""
    if isinstance(content, str):
        return content
    parts = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(block["text"])
    return "\n".join(parts)


def _is_tool_result_turn(entry: dict) -> bool:
    return bool(entry.get("toolUseResult") or entry.get("sourceToolAssistantUUID"))


def _load_turns(jsonl_path: Path) -> list[dict]:
    """Return list of {role, text, ts} dicts — only real human/assistant turns."""
    turns = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            e = json.loads(line)
            role = e.get("type")
            if role not in ("user", "assistant"):
                continue
            if role == "user" and _is_tool_result_turn(e):
                continue

            text = _extract_text(e["message"]["content"]).strip()
            if not text:
                continue

            # Skip pure system injections (very long, no newline-delimited conversation)
            if role == "user" and text.startswith("# Role:"):
                continue

            ts_raw = e.get("timestamp", "")
            try:
                ts = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
                ts_str = ts.astimezone().strftime("%Y-%m-%d %H:%M")
            except Exception:
                ts_str = ""

            turns.append({"role": role, "text": text, "ts": ts_str})

    return turns


# ── PDF renderer ─────────────────────────────────────────────────────────────

class ChatPDF(FPDF):
    def __init__(self, title: str):
        super().__init__(orientation="P", unit="mm", format="A4")
        self.title_str = title
        self.set_auto_page_break(auto=True, margin=15)
        self.set_margins(15, 15, 15)
        # Register Unicode fonts
        self.add_font("Sans",      style="",  fname=_SANS)
        self.add_font("Sans",      style="B", fname=_SANS_BOLD)
        self.add_font("Sans",      style="I", fname=_SANS_OBLIQUE)
        self.add_font("Mono",      style="",  fname=_MONO)

    def header(self):
        self.set_font("Sans", "B", 9)
        self.set_text_color(*C_USER_HDR)
        self.cell(0, 6, self.title_str, align="L",
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_draw_color(180, 180, 180)
        self.line(15, self.get_y(), 195, self.get_y())
        self.ln(2)

    def footer(self):
        self.set_y(-12)
        self.set_font("Sans", "", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 6, f"Page {self.page_no()}", align="C")

    def _set_fill(self, role: str):
        bg = C_USER_BG if role == "user" else C_ASST_BG
        self.set_fill_color(*bg)

    def render_turn(self, role: str, text: str, ts: str):
        label = "You" if role == "user" else "Claude"
        hdr_colour = C_USER_HDR if role == "user" else C_ASST_HDR

        # Header bar
        self._set_fill(role)
        self.set_font("Sans", "B", 9)
        self.set_text_color(*hdr_colour)
        header_line = f"  {label}   {ts}"
        self.set_fill_color(*(C_USER_BG if role == "user" else C_ASST_BG))
        self.cell(0, 7, header_line, fill=True,
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        # Body — split on code fences
        self._render_body(text, role)
        self.ln(3)

    def _render_body(self, text: str, role: str):
        """Render text with code blocks in monospace and prose in normal font."""
        segments = re.split(r"(```[^\n]*\n.*?```)", text, flags=re.DOTALL)
        bg = C_USER_BG if role == "user" else C_ASST_BG
        self.set_fill_color(*bg)

        for seg in segments:
            if seg.startswith("```"):
                # Strip fence markers
                body = re.sub(r"^```[^\n]*\n", "", seg).rstrip("`").rstrip()
                self._render_code(body)
            else:
                self._render_prose(seg, bg)

    def _render_prose(self, text: str, bg_colour: tuple):
        self.set_font("Sans", "", 9)
        self.set_text_color(*C_TEXT)
        self.set_fill_color(*bg_colour)
        usable_w = self.w - self.l_margin - self.r_margin - 4
        for line in text.split("\n"):
            line = line.strip()
            if not line:
                self.ln(2)
                continue
            # Bold for markdown headers
            if line.startswith("#"):
                line = re.sub(r"^#+\s*", "", line)
                self.set_font("Sans", "B", 9)
            else:
                self.set_font("Sans", "", 9)
            # Wrap long lines
            wrapped = textwrap.wrap(line, width=110) or [""]
            for wline in wrapped:
                self.cell(4, 0, "")   # left indent
                self.multi_cell(usable_w, 5, wline, fill=True,
                                new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    def _render_code(self, code: str):
        self.set_font("Mono", "", 7.5)
        self.set_text_color(*C_CODE_TEXT)
        self.set_fill_color(*C_CODE_BG)
        self.set_draw_color(200, 200, 200)
        usable_w = self.w - self.l_margin - self.r_margin - 8

        x0 = self.get_x()
        y0 = self.get_y()
        self.ln(1)
        for line in code.split("\n"):
            self.cell(8, 0, "")   # indent
            self.multi_cell(usable_w, 4.5, line, fill=True,
                            new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(1)
        self.set_text_color(*C_TEXT)


def export(jsonl_path: Path, out_dir: Path, label: str):
    turns = _load_turns(jsonl_path)
    if not turns:
        print("No turns found.")
        return

    pdf = ChatPDF(label)
    pdf.add_page()

    # Cover info
    pdf.set_font("Sans", "B", 14)
    pdf.set_text_color(*C_USER_HDR)
    pdf.cell(0, 10, label, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Sans", "", 9)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 6, f"Exported {datetime.now().strftime('%Y-%m-%d %H:%M')}  |  {len(turns)} turns",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(4)

    for turn in turns:
        pdf.render_turn(turn["role"], turn["text"], turn["ts"])

    slug = datetime.now().strftime("%Y%m%d_%H%M")
    out_path = out_dir / f"chat_{slug}.pdf"
    pdf.output(str(out_path))
    print(f"Saved -> {out_path}  ({len(turns)} turns)")


if __name__ == "__main__":
    export(JSONL_PATH, OUT_DIR, SESSION_LABEL)
