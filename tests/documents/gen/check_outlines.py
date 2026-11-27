"""Check (or write) the expected outlines in ../expected/outline/.

  python check_outlines.py           check every outline file against the corpus; prints mismatches
  python check_outlines.py --write   write the outline files from the parser below

Two checks are made for each outline file:
  1. every row's line number points at a line that is a heading of the stated level with the
     stated text (for a setext or underlined heading, the next line is the underline);
  2. the rows equal what the parser finds, so no heading is missing or extra.
Standard library only.
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.join(HERE, "..", "corpus")
OUTLINE = os.path.join(HERE, "..", "expected", "outline")

FILES = [
    "manual/aq40-manual.md",
    "textbook/ch1-heat-transfer.md",
    "textbook/ch2-simple-statistics.txt",
    "notes/2026-09-15-meeting.md",
    "notes/2026-10-04-log-review.md",
    "notes/2026-11-02-trial-summary.md",
    "notes/2026-10-09-site-visit-report.txt",
    "notes/2026-10-22-supplier-visit.txt",
]


def read_lines(rel):
    """Decode a corpus file (UTF-16 with BOM, UTF-8, or Windows-1252) into a list of lines."""
    data = open(os.path.join(CORPUS, rel), "rb").read()
    if data.startswith(b"\xff\xfe") or data.startswith(b"\xfe\xff"):
        text = data.decode("utf-16")
    elif data.startswith(b"\xef\xbb\xbf"):
        text = data[3:].decode("utf-8")
    else:
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            text = data.decode("cp1252")
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return lines


ATX = re.compile(r"^ {0,3}(#{1,6})(?:[ \t]+(.*?))?[ \t]*$")
FENCE = re.compile(r"^ {0,3}(```|~~~)")
SETEXT_EQ = re.compile(r"^ {0,3}=+[ \t]*$")
SETEXT_DASH = re.compile(r"^ {0,3}-+[ \t]*$")
NOT_PARA = re.compile(r"^\s*([|>*+-]|\d+[.)]\s|#)")


def atx_text(raw):
    raw = raw or ""
    raw = re.sub(r"(^|[ \t]+)#+[ \t]*$", "", raw)
    return raw.strip()


def outline_md(lines):
    out = []
    i = 0
    n = len(lines)
    if n and lines[0].strip() == "---":
        i = 1
        while i < n and lines[i].strip() != "---":
            i += 1
        i += 1
    in_code = False
    while i < n:
        line = lines[i]
        if FENCE.match(line):
            in_code = not in_code
            i += 1
            continue
        if in_code:
            i += 1
            continue
        m = ATX.match(line)
        if m:
            out.append((i + 1, len(m.group(1)), atx_text(m.group(2))))
            i += 1
            continue
        prev_blank = i == 0 or lines[i - 1].strip() == ""
        if (line.strip() and prev_blank and not NOT_PARA.match(line) and i + 1 < n):
            nxt = lines[i + 1]
            if SETEXT_EQ.match(nxt):
                out.append((i + 1, 1, line.strip()))
                i += 2
                continue
            if SETEXT_DASH.match(nxt):
                out.append((i + 1, 2, line.strip()))
                i += 2
                continue
        i += 1
    return out


NUMBERED = re.compile(r"^(\d+(?:\.\d+)*) +\S")


def outline_txt(lines):
    out = []
    n = len(lines)
    i = 0
    while i < n:
        line = lines[i]
        s = line.strip()
        if s and i + 1 < n and not set(s) <= set("=-"):
            nxt = lines[i + 1].strip()
            if len(nxt) >= 3 and set(nxt) == {"="}:
                out.append((i + 1, 1, s))
                i += 2
                continue
            if len(nxt) >= 3 and set(nxt) == {"-"}:
                out.append((i + 1, 2, s))
                i += 2
                continue
        m = NUMBERED.match(line)
        if (m and (i == 0 or lines[i - 1].strip() == "") and len(line) < 80
                and not s.endswith(".")):
            out.append((i + 1, m.group(1).count(".") + 1, s))
        i += 1
    return out


def outline(rel):
    lines = read_lines(rel)
    return outline_md(lines) if rel.endswith(".md") else outline_txt(lines)


def tsv_name(rel):
    return rel.replace("/", "__") + ".tsv"


def row_ok(rel, lines, num, level, text):
    """Independent check of one row against the file's own lines."""
    if not 1 <= num <= len(lines):
        return "line %d out of range" % num
    line = lines[num - 1]
    nxt = lines[num].strip() if num < len(lines) else ""
    if rel.endswith(".md"):
        m = ATX.match(line)
        if m:
            if len(m.group(1)) != level or atx_text(m.group(2)) != text:
                return "ATX heading differs: %r" % line
            return None
        if line.strip() == text and ((level == 1 and SETEXT_EQ.match(nxt)) or
                                     (level == 2 and SETEXT_DASH.match(nxt))):
            return None
        return "not a heading: %r" % line
    if line.strip() != text:
        return "text differs: %r" % line
    if nxt and set(nxt) == {"="} and len(nxt) >= 3:
        return None if level == 1 else "level should be 1"
    if nxt and set(nxt) == {"-"} and len(nxt) >= 3:
        return None if level == 2 else "level should be 2"
    m = NUMBERED.match(line)
    if m and m.group(1).count(".") + 1 == level and lines[num - 2].strip() == "":
        return None
    return "not a text heading of level %d: %r" % (level, line)


def main():
    write = "--write" in sys.argv
    os.makedirs(OUTLINE, exist_ok=True)
    bad = 0
    for rel in FILES:
        rows = outline(rel)
        path = os.path.join(OUTLINE, tsv_name(rel))
        if write:
            with open(path, "w", encoding="utf-8", newline="\n") as f:
                for num, level, text in rows:
                    f.write("%d\t%d\t%s\n" % (num, level, text))
            print("wrote %s (%d headings)" % (tsv_name(rel), len(rows)))
            continue
        lines = read_lines(rel)
        got = []
        for k, raw in enumerate(open(path, encoding="utf-8").read().split("\n"), 1):
            if not raw:
                continue
            parts = raw.split("\t")
            if len(parts) != 3:
                print("%s row %d: not three tab separated fields" % (tsv_name(rel), k))
                bad += 1
                continue
            num, level, text = int(parts[0]), int(parts[1]), parts[2]
            got.append((num, level, text))
            err = row_ok(rel, lines, num, level, text)
            if err:
                print("%s line %d: %s" % (tsv_name(rel), num, err))
                bad += 1
        if got != rows:
            print("%s: rows differ from the parser (%d in file, %d parsed)"
                  % (tsv_name(rel), len(got), len(rows)))
            bad += 1
        else:
            print("ok %s (%d headings)" % (tsv_name(rel), len(got)))
    if not write:
        print("mismatches: %d" % bad)
        sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
