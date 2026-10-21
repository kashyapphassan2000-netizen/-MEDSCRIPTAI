"""Turn guide blocks into Google Docs batchUpdate requests with simulated indices (no re-fetch needed)."""
import re

GREY_CODE = {"red": 0.949, "green": 0.949, "blue": 0.949}
HEADER_BG = {"red": 0.929, "green": 0.929, "blue": 0.929}
BLUE = {"red": 221 / 255, "green": 235 / 255, "blue": 247 / 255}      # light blue #DDEBF7
BLUE_HEAD = {"red": 189 / 255, "green": 215 / 255, "blue": 238 / 255}  # #BDD7EE
CALLOUT = {
    "WHERE:": {"red": 0.10, "green": 0.33, "blue": 0.62},
    "GOAL:": {"red": 0.10, "green": 0.33, "blue": 0.62},
    "You should see:": {"red": 0.11, "green": 0.50, "blue": 0.22},
    "Watch out:": {"red": 0.75, "green": 0.38, "blue": 0.0},
    "CHECKPOINT:": {"red": 0.45, "green": 0.20, "blue": 0.70},
    "If not:": {"red": 0.45, "green": 0.20, "blue": 0.70},
}
INLINE = re.compile(r"\*\*(.+?)\*\*|`([^`]+)`|(https?://[^\s)\]|,]+)")


def u16(s):
    """Length in UTF-16 code units (what Google Docs indices count)."""
    return len(s.encode("utf-16-le")) // 2


def parse_inline(text):
    """'**b** and `c` http://x' -> plain text + spans [(start, end, kind, url)] in UTF-16 offsets."""
    out, spans, pos = [], [], 0
    for m in INLINE.finditer(text):
        out.append(text[pos:m.start()])
        start = u16("".join(out))
        if m.group(1) is not None:
            piece, kind, url = m.group(1), "bold", None
        elif m.group(2) is not None:
            piece, kind, url = m.group(2), "code", None
        else:
            piece, kind, url = m.group(3), "link", m.group(3)
        out.append(piece)
        spans.append((start, start + u16(piece), kind, url))
        pos = m.end()
    out.append(text[pos:])
    plain = "".join(out)
    for label, color in CALLOUT.items():                 # colour the lab-manual labels
        if plain.startswith(label):
            spans.append((0, u16(label), "label", color))
    return plain, spans


def span_requests(spans, base, tab):
    reqs = []
    for s, e, kind, extra in spans:
        rng = {"startIndex": base + s, "endIndex": base + e, "tabId": tab}
        if e <= s:
            continue
        if kind == "bold":
            reqs.append({"updateTextStyle": {"range": rng, "textStyle": {"bold": True}, "fields": "bold"}})
        elif kind == "code":
            reqs.append({"updateTextStyle": {"range": rng, "textStyle": {
                "weightedFontFamily": {"fontFamily": "Roboto Mono"},
                "backgroundColor": {"color": {"rgbColor": GREY_CODE}}},
                "fields": "weightedFontFamily,backgroundColor"}})
        elif kind == "link":
            reqs.append({"updateTextStyle": {"range": rng, "textStyle": {"link": {"url": extra}}, "fields": "link"}})
        elif kind == "label":
            reqs.append({"updateTextStyle": {"range": rng, "textStyle": {
                "bold": True, "foregroundColor": {"color": {"rgbColor": extra}}},
                "fields": "bold,foregroundColor"}})
    return reqs


def reset(rng, named="NORMAL_TEXT"):
    return [
        {"updateParagraphStyle": {"range": rng, "paragraphStyle": {"namedStyleType": named},
                                  "fields": "namedStyleType,shading,indentStart,indentFirstLine,spaceAbove,spaceBelow"}},
        {"updateTextStyle": {"range": rng, "textStyle": {},
                             "fields": "bold,italic,weightedFontFamily,fontSize,backgroundColor,foregroundColor,link"}},
    ]


def build(blocks, tab, cursor):
    """Return (requests, new_cursor). `cursor` = start index of the tab's final empty paragraph."""
    reqs = []
    for b in blocks:
        t = b["t"]
        if t in ("h1", "h2", "h3", "p"):
            plain, spans = parse_inline(b["text"])
            text = plain + "\n"
            reqs.append({"insertText": {"location": {"index": cursor, "tabId": tab}, "text": text}})
            rng = {"startIndex": cursor, "endIndex": cursor + u16(text), "tabId": tab}
            named = {"h1": "HEADING_1", "h2": "HEADING_2", "h3": "HEADING_3", "p": "NORMAL_TEXT"}[t]
            reqs += reset(rng, named)
            reqs += span_requests(spans, cursor, tab)
            cursor += u16(text)
        elif t == "bullets":
            start = cursor
            all_spans = []
            for item in b["items"]:
                plain, spans = parse_inline(item)
                all_spans.append((cursor, spans))
                reqs.append({"insertText": {"location": {"index": cursor, "tabId": tab}, "text": plain + "\n"}})
                cursor += u16(plain + "\n")
            rng = {"startIndex": start, "endIndex": cursor, "tabId": tab}
            reqs += reset(rng)
            reqs.append({"createParagraphBullets": {"range": rng, "bulletPreset": "BULLET_DISC_CIRCLE_SQUARE"}})
            for base, spans in all_spans:
                reqs += span_requests(spans, base, tab)
        elif t == "code":
            text = b["text"].rstrip("\n") + "\n"
            reqs.append({"insertText": {"location": {"index": cursor, "tabId": tab}, "text": text}})
            rng = {"startIndex": cursor, "endIndex": cursor + u16(text), "tabId": tab}
            reqs += reset(rng)
            reqs.append({"updateParagraphStyle": {"range": rng, "paragraphStyle": {
                "shading": {"backgroundColor": {"color": {"rgbColor": GREY_CODE}}},
                "spaceAbove": {"magnitude": 0, "unit": "PT"}, "spaceBelow": {"magnitude": 0, "unit": "PT"}},
                "fields": "shading,spaceAbove,spaceBelow"}})
            reqs.append({"updateTextStyle": {"range": rng, "textStyle": {
                "weightedFontFamily": {"fontFamily": "Roboto Mono"},
                "fontSize": {"magnitude": 9, "unit": "PT"}},
                "fields": "weightedFontFamily,fontSize"}})
            cursor += u16(text)
        elif t == "table":
            rows = b["rows"]
            R, C = len(rows), max(len(r) for r in rows)
            rows = [r + [""] * (C - len(r)) for r in rows]
            reqs.append({"insertTable": {"rows": R, "columns": C, "location": {"index": cursor, "tabId": tab}}})
            T = cursor + 1                                       # Docs adds a newline before the table
            parsed = [[parse_inline(cell) for cell in row] for row in rows]
            base = lambda r, c: T + 3 + r * (1 + 2 * C) + 2 * c   # empty-cell paragraph index
            for r in reversed(range(R)):                         # fill from the END so indices stay valid
                for c in reversed(range(C)):
                    plain = parsed[r][c][0]
                    if plain:
                        reqs.append({"insertText": {"location": {"index": base(r, c), "tabId": tab}, "text": plain}})
            shift = 0
            for r in range(R):                                   # styles, with final (shifted) positions
                for c in range(C):
                    plain, spans = parsed[r][c]
                    start = base(r, c) + shift
                    if r == 0 and plain:
                        reqs.append({"updateTextStyle": {"range": {"startIndex": start, "endIndex": start + u16(plain),
                                     "tabId": tab}, "textStyle": {"bold": True}, "fields": "bold"}})
                    reqs += span_requests(spans, start, tab)
                    shift += u16(plain)
            blue = set(b.get("blue", []))
            for c in range(C):
                for r in range(R):
                    color = None
                    if c in blue:
                        color = BLUE_HEAD if r == 0 else BLUE
                    elif r == 0:
                        color = HEADER_BG
                    if color:
                        reqs.append({"updateTableCellStyle": {
                            "tableRange": {"tableCellLocation": {"tableStartLocation": {"index": T, "tabId": tab},
                                                                 "rowIndex": r, "columnIndex": c},
                                           "rowSpan": 1, "columnSpan": 1},
                            "tableCellStyle": {"backgroundColor": {"color": {"rgbColor": color}}},
                            "fields": "backgroundColor"}})
            if b.get("widths"):
                for c, w in enumerate(b["widths"]):
                    if w:
                        reqs.append({"updateTableColumnProperties": {
                            "tableStartLocation": {"index": T, "tabId": tab}, "columnIndices": [c],
                            "tableColumnProperties": {"widthType": "FIXED_WIDTH", "width": {"magnitude": w, "unit": "PT"}},
                            "fields": "widthType,width"}})
            cursor = T + 2 + R * (1 + 2 * C) + shift
    return reqs, cursor
