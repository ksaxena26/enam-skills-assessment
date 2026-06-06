import json
import re
from pathlib import Path
from fpdf import FPDF

JSONL = Path(r"C:\Users\Kushagra Saxena\.claude\projects\C--Users-Kushagra-Saxena-Documents-enam-skills-assessment\a7321368-f8d4-4ae9-8db6-d8d5bd517f05.jsonl")
OUT   = Path(r"C:\Users\Kushagra Saxena\Documents\enam-skills-assessment\chats\chat_history.pdf")


# ── Parse messages ─────────────────────────────────────────────────────────────
def extract_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type", "")
            if btype == "text":
                parts.append(block.get("text", ""))
            elif btype == "tool_use":
                name = block.get("name", "")
                inp  = block.get("input", {})
                desc = inp.get("description", inp.get("command", inp.get("file_path", str(inp)[:80])))
                parts.append(f"[Tool call: {name} — {str(desc)[:120]}]")
            elif btype == "tool_result":
                inner = block.get("content", "")
                if isinstance(inner, list):
                    inner = " ".join(b.get("text", "") for b in inner if isinstance(b, dict))
                parts.append(f"[Tool result: {str(inner)[:200]}]")
        return "\n".join(p for p in parts if p.strip())
    return ""


messages = []
for raw in JSONL.read_text(encoding="utf-8").splitlines():
    raw = raw.strip()
    if not raw:
        continue
    try:
        obj = json.loads(raw)
    except Exception:
        continue
    # Structure: {"type": "user"|"assistant", "message": {"role": ..., "content": ...}}
    event_type = obj.get("type", "")
    if event_type not in ("user", "assistant"):
        continue
    msg = obj.get("message", {})
    if not isinstance(msg, dict):
        continue
    content = extract_text(msg.get("content", ""))
    if content.strip():
        messages.append((event_type, content.strip()))

print(f"Parsed {len(messages)} messages")


# ── Text sanitiser (latin-1 safe + markdown stripped) ─────────────────────────
REPLACEMENTS = {
    "—": "--",   # em dash
    "–": "-",    # en dash
    "‘": "'",    # left single quote
    "’": "'",    # right single quote
    "“": '"',    # left double quote
    "”": '"',    # right double quote
    "•": "-",    # bullet
    "…": "...",  # ellipsis
    "é": "e",    # é
    "è": "e",    # è
    "ê": "e",    # ê
    "à": "a",    # à
    "â": "a",    # â
    "û": "u",    # û
    "₹": "Rs.",  # rupee sign
    "→": "->",   # arrow
    "←": "<-",
    "°": " deg",
    "×": "x",
}

def sanitise(text: str) -> str:
    # Replace known special chars
    for char, rep in REPLACEMENTS.items():
        text = text.replace(char, rep)
    # Strip markdown
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text, flags=re.S)
    text = re.sub(r"\*(.+?)\*",     r"\1", text, flags=re.S)
    text = re.sub(r"```[^\n]*\n?",  "",    text)
    text = re.sub(r"`(.+?)`",       r'"\1"', text)
    text = re.sub(r"^#{1,6} +",     "",    text, flags=re.M)
    text = re.sub(r"\n{3,}",        "\n\n", text)
    # Drop any remaining non-latin-1 chars
    text = text.encode("latin-1", errors="replace").decode("latin-1")
    return text.strip()


# ── PDF ────────────────────────────────────────────────────────────────────────
class ChatPDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(60, 60, 60)
        self.cell(0, 8, "Wisdom Trader -- Full Chat History",
                  align="C", new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(180, 180, 180)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(4)

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(160, 160, 160)
        self.cell(0, 6, f"Page {self.page_no()}", align="C")


pdf = ChatPDF()
pdf.set_auto_page_break(auto=True, margin=18)
pdf.set_margins(15, 18, 15)
pdf.add_page()

LABEL_W = 18

for i, (role, content) in enumerate(messages):
    is_user = role == "user"
    label   = "You" if is_user else "Claude"

    # Coloured role label
    if is_user:
        pdf.set_fill_color(227, 242, 253)
        pdf.set_text_color(13, 71, 161)
    else:
        pdf.set_fill_color(232, 245, 233)
        pdf.set_text_color(27, 94, 32)
    pdf.set_font("Helvetica", "B", 8)
    pdf.cell(LABEL_W, 6, f" {label}", fill=True, new_x="LMARGIN", new_y="NEXT")

    # Body
    pdf.set_text_color(20, 20, 20)
    pdf.set_font("Helvetica", "", 8)
    pdf.multi_cell(0, 4.5, sanitise(content), new_x="LMARGIN", new_y="NEXT")

    pdf.ln(2)
    pdf.set_draw_color(215, 215, 215)
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
    pdf.ln(4)

pdf.output(str(OUT))
kb = OUT.stat().st_size // 1024
print(f"Saved  : {OUT}")
print(f"Size   : {kb} KB  |  {pdf.page} pages  |  {len(messages)} messages")
