"""Build the MedScriptAI lab manual from docs/guide/*.md.

Outputs:
  docs/build/blocks/<tab>.json      blocks for the Google Docs writer (gdocs_writer.py)
  docs/MedScriptAI_Lab_Manual.html  printable HTML (then PDF via Chromium)

Mini-markup: '# ', '## ', '### ' headings; '- ' bullets; ``` code fences; | tables |;
'@@include path' inserts a repo file as a code block; '{blue}' in a table header cell = light-blue column;
'{w=80}' in a header cell = fixed column width (pt) in Google Docs. Every other non-blank line = one paragraph.
"""
import html
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GUIDE = ROOT / "docs" / "guide"
OUT_BLOCKS = ROOT / "docs" / "build" / "blocks"
TABS = [  # (source file, tab key, Google Docs tab title or None = existing MEDSCRIPT AI tab)
    ("00_start.md", "main", None),
    ("02a_hyperstack.md", "p2a", "MS Part 2A - Hyperstack + SSH + VS Code + WinSCP"),
    ("02b_project.md", "p2b", "MS Part 2B - Project Folder + Tools on the VM"),
    ("03_core.md", "p3", "MS Part 3 - Core Idea"),
    ("04_basics.md", "p4", "MS Part 4 - Basics"),
    ("05_intermediate.md", "p5", "MS Part 5 - Intermediate"),
    ("06a_dpo_api.md", "p6a", "MS Part 6A - DPO + vLLM + API"),
    ("06b_deploy.md", "p6b", "MS Part 6B - Docker + Cloud Deploy"),
    ("07_elite.md", "p7", "MS Part 7 - Elite"),
    ("08_reference.md", "p8", "MS Parts 8-14 - Reference"),
]


SPECIAL_TOKEN = re.compile(r"<\|[a-z_]+\|>")


def split_row(line):
    """Split '| a | b |' on pipes that are NOT inside `code`, <|special_tokens|> or escaped as \\|."""
    line = SPECIAL_TOKEN.sub(lambda m: m.group(0).replace("|", "\x00"), line.replace("\\|", "\x00"))
    cells, cur, in_code = [], "", False
    for ch in line.strip().strip("|"):
        if ch == "`":
            in_code = not in_code
        if ch == "|" and not in_code:
            cells.append(cur)
            cur = ""
        else:
            cur += ch
    cells.append(cur)
    return [c.strip().replace("\x00", "|") for c in cells]


def parse(text):
    blocks, lines, i = [], text.splitlines(), 0
    while i < len(lines):
        line = lines[i]
        s = line.strip()
        if not s:
            i += 1
            continue
        if s.startswith("```"):
            j, code = i + 1, []
            while j < len(lines) and not lines[j].strip().startswith("```"):
                code.append(lines[j])
                j += 1
            blocks.append({"t": "code", "text": "\n".join(code)})
            i = j + 1
            continue
        if s.startswith("@@include "):
            path = ROOT / s.split(" ", 1)[1].strip()
            blocks.append({"t": "code", "text": path.read_text().rstrip("\n"), "file": str(path.relative_to(ROOT))})
            i += 1
            continue
        if s.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = split_row(lines[i].strip())
                if rows and len(cells) != len(rows[0]):
                    break                                   # a different column count starts a NEW table
                if not all(re.fullmatch(r":?-{2,}:?", c) for c in cells):
                    rows.append(cells)
                i += 1
            blue, widths = [], []
            for c, cell in enumerate(rows[0]):
                if "{blue}" in cell:
                    blue.append(c)
                m = re.search(r"\{w=(\d+)\}", cell)
                widths.append(int(m.group(1)) if m else 0)
                rows[0][c] = re.sub(r"\s*\{(blue|w=\d+)\}", "", cell)
            rows = [[c.replace("\\n", "\n") for c in r] for r in rows]
            uneven = [r for r in rows if len(r) != len(rows[0])]
            assert not uneven, f"table under {rows[0]} has a row with a different column count: {uneven[0][:3]}"
            block = {"t": "table", "rows": rows, "blue": blue}
            if any(widths):
                block["widths"] = widths
            blocks.append(block)
            continue
        if s.startswith("- "):
            items = []
            while i < len(lines) and lines[i].strip().startswith("- "):
                items.append(lines[i].strip()[2:])
                i += 1
            blocks.append({"t": "bullets", "items": items})
            continue
        m = re.match(r"(#{1,3}) (.*)", s)
        if m:
            blocks.append({"t": f"h{len(m.group(1))}", "text": m.group(2)})
        else:
            blocks.append({"t": "p", "text": s})
        i += 1
    return blocks


INLINE = re.compile(r"\*\*(.+?)\*\*|`([^`]+)`|(https?://[^\s)\]|,]+)")
LABELS = {"WHERE:": "where", "GOAL:": "where", "You should see:": "see", "Watch out:": "watch",
          "CHECKPOINT:": "check", "If not:": "check"}


def inline_html(text):
    s = inline_spans(text)
    for label, cls in LABELS.items():
        if text.startswith(label):
            return f'<p class="lbl {cls}"><b>{html.escape(label)}</b>{s[len(html.escape(label)):]}</p>'
    return f"<p>{s}</p>"


def inline_spans(text):
    """Inline markup -> HTML without any paragraph wrapper (used inside table cells and list items)."""
    out, pos = [], 0
    for m in INLINE.finditer(text):
        out.append(html.escape(text[pos:m.start()]))
        if m.group(1) is not None:
            out.append(f"<b>{html.escape(m.group(1))}</b>")
        elif m.group(2) is not None:
            out.append(f"<code>{html.escape(m.group(2))}</code>")
        else:
            u = html.escape(m.group(3))
            out.append(f'<a href="{u}">{u}</a>')
        pos = m.end()
    out.append(html.escape(text[pos:]))
    return "".join(out)


def to_html(all_blocks):
    toc, body, n = [], [], 0
    for blocks in all_blocks:
        for b in blocks:
            t = b["t"]
            if t in ("h1", "h2", "h3"):
                n += 1
                anchor = f"s{n}"
                if t != "h3":
                    toc.append(f'<li class="toc-{t}"><a href="#{anchor}">{html.escape(b["text"])}</a></li>')
                body.append(f'<{t} id="{anchor}">{html.escape(b["text"])}</{t}>')
            elif t == "p":
                body.append(inline_html(b["text"]))
            elif t == "bullets":
                body.append("<ul>" + "".join(f"<li>{inline_spans(x)}</li>" for x in b["items"]) + "</ul>")
            elif t == "code":
                body.append(f"<pre>{html.escape(b['text'])}</pre>")
            elif t == "table":
                blue = set(b.get("blue", []))
                rows = []
                for r, row in enumerate(b["rows"]):
                    tag = "th" if r == 0 else "td"
                    cells = "".join(
                        f'<{tag}{" class=blue" if c in blue else ""}>{inline_spans(cell).replace(chr(10), "<br>")}</{tag}>'
                        for c, cell in enumerate(row))
                    rows.append(f"<tr>{cells}</tr>")
                body.append("<table>" + "".join(rows) + "</table>")
    css = (ROOT / "docs" / "build" / "manual.css").read_text()
    return (f"<!doctype html><html><head><meta charset='utf-8'><title>MedScriptAI Lab Manual</title>"
            f"<style>{css}</style></head><body>"
            f"<div class='cover'><h1 class='title'>MedScriptAI</h1><p class='subtitle'>Zero-to-Elite Lab Manual: "
            f"medical LLM post-training (QLoRA SFT + DPO) and production deployment</p>"
            f"<p class='meta'>Llama-3.1-8B-Instruct · Unsloth · TRL · PEFT · distilabel · vLLM multi-LoRA · FastAPI · "
            f"LangSmith · Docker · FakeCloud / AWS<br>Version: October 2026</p></div>"
            f"<nav class='toc'><h2>Table of contents</h2><ul>{''.join(toc)}</ul></nav>"
            + "".join(body) + "</body></html>")


if __name__ == "__main__":
    OUT_BLOCKS.mkdir(parents=True, exist_ok=True)
    everything, manifest = [], []
    for src, key, title in TABS:
        blocks = parse((GUIDE / src).read_text())
        (OUT_BLOCKS / f"{key}.json").write_text(json.dumps(blocks, ensure_ascii=False))
        manifest.append({"key": key, "title": title, "blocks": len(blocks)})
        everything.append(blocks)
        print(f"{src:22} -> {key:6} {len(blocks):5} blocks")
    (OUT_BLOCKS / "manifest.json").write_text(json.dumps(manifest, indent=1))
    (ROOT / "docs" / "MedScriptAI_Lab_Manual.html").write_text(to_html(everything))
    print("wrote docs/MedScriptAI_Lab_Manual.html")
