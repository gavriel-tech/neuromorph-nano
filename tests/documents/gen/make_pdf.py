"""PDF test material for tests/documents/pdf/ and its expected text in tests/documents/expected/pdf/.

Steps (make_pdf.ps1 runs them in order):
  python make_pdf.py html    the manual as HTML pages (one and two columns, a page holding only a
                             picture), the lines for Microsoft Print to PDF, the expected text of
                             every page and the gold passages with their pages
  (Edge headless and Microsoft Print to PDF print the HTML and the lines; see make_pdf.ps1)
  python make_pdf.py craft   PDFs crafted byte by byte, marked as crafted: an encrypted copy of
                             the one-column manual for each standard security handler revision
                             the reader supports (RC4 40 bit, RC4 128 bit, AES-128, AES-256), all
                             with an empty user password and permissions only; a file using the
                             older filters, an object stream, a cross-reference stream, a simple
                             font with a /Differences encoding and a composite font with a
                             ToUnicode map; a file whose composite font has no way to map glyphs
                             to characters

Pages are made by construction: every HTML page is its own section with a forced page break, and
the blocks of the manual are given to pages by an estimate of their height that leaves room, so
the text of every page is known before printing. make_pdf.ps1 checks that each printed file has the
number of pages planned (counted in the file's bytes, not by the reader under test).

Expected text holds what each page shows as text: headings, paragraphs, list items with their
bullet or number, table rows with their cells in order, code lines; not the running header and
footer (which the reader must remove). Long words carry a soft hyphen in the HTML, so the printer
breaks some of them at line ends with a visible hyphen; the expected text holds them whole.
Standard library only. Everything is invented.
"""
import hashlib
import html
import os
import re
import struct
import sys
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
DOCS = os.path.dirname(HERE)
MANUAL = os.path.join(DOCS, "corpus", "manual", "aq40-manual.md")
OUT = os.path.join(DOCS, "pdf")
EXP = os.path.join(DOCS, "expected", "pdf")
SRC = os.path.join(EXP, "src")
HEADER = "Halvern AQ-40 Service and Maintenance Manual"


def write(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    mode = "wb" if isinstance(data, bytes) else "w"
    kw = {} if isinstance(data, bytes) else {"encoding": "utf-8", "newline": "\n"}
    with open(path, mode, **kw) as f:
        f.write(data)


# ---------------------------------------------------------------- the manual as blocks

def inline(md, shy):
    """(html, text) of one line of Markdown inline text: bold and code marks, soft hyphens."""
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", md)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    out = []
    pos = 0
    for m in re.finditer(r"\*\*(.+?)\*\*|`([^`]+)`", md):
        out.append(("t", md[pos:m.start()]))
        if m.group(1) is not None:
            out.append(("b", m.group(1)))
        else:
            out.append(("c", m.group(2)))
        pos = m.end()
    out.append(("t", md[pos:]))
    parts = []
    for kind, s in out:
        h = html.escape(s, quote=False)
        if shy and kind == "t":
            h = re.sub(r"[A-Za-z]{11,}", soft, h)
        if kind == "b":
            h = "<b>" + h + "</b>"
        elif kind == "c":
            h = "<code>" + h + "</code>"
        parts.append(h)
    return "".join(parts), text


def soft(m):
    """A soft hyphen in the middle of a long word, after a vowel followed by a consonant."""
    w = m.group(0)
    mid = len(w) // 2
    for d in range(0, mid - 2):
        for i in (mid + d, mid - d):
            if 3 <= i <= len(w) - 3 and w[i - 1] in "aeiouy" and w[i] not in "aeiouy":
                return w[:i] + "&shy;" + w[i:]
    return w


def blocks_of(md):
    """The manual as a list of blocks: (kind, html, text, weight in estimated lines)."""
    lines = md.split("\n")
    out = []
    i = 0
    para = []

    def flush():
        if para:
            h, t = inline(" ".join(para), True)
            out.append(("p", "<p>" + h + "</p>", t))
            para.clear()

    while i < len(lines):
        l = lines[i]
        if not l.strip():
            flush()
            i += 1
            continue
        m = re.match(r"(#{1,3}) (.*)", l)
        if m:
            flush()
            lv = len(m.group(1))
            h, t = inline(m.group(2), False)
            out.append(("h%d" % lv, "<h%d>%s</h%d>" % (lv, h, lv), t))
            i += 1
            continue
        if l.startswith("> "):
            flush()
            q = []
            while i < len(lines) and lines[i].startswith(">"):
                q.append(lines[i][1:].strip())
                i += 1
            h, t = inline(" ".join(q), True)
            out.append(("note", '<div class="box">' + h + "</div>", t))
            continue
        if l.startswith("- "):
            flush()
            while i < len(lines) and lines[i].startswith("- "):
                h, t = inline(lines[i][2:], True)
                out.append(("li", h, "• " + t))
                i += 1
            continue
        m = re.match(r"(\d+)\. (.*)", l)
        if m:
            flush()
            while i < len(lines) and re.match(r"\d+\. ", lines[i]):
                m = re.match(r"(\d+)\. (.*)", lines[i])
                item = [m.group(2)]
                i += 1
                while i < len(lines) and lines[i].startswith("   "):
                    item.append(lines[i].strip())
                    i += 1
                h, t = inline(" ".join(item), True)
                out.append(("oli:" + m.group(1), h, m.group(1) + ". " + t))
            continue
        if l.startswith("|"):
            flush()
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r"-+", c) for c in cells):
                    rows.append(cells)
                i += 1
            out.append(("table", rows, "\n".join(" ".join(inline(c, False)[1] for c in r) for r in rows)))
            continue
        if l.startswith("```"):
            flush()
            i += 1
            code = []
            while i < len(lines) and not lines[i].startswith("```"):
                code.append(lines[i])
                i += 1
            i += 1
            out.append(("pre", "<pre>" + html.escape("\n".join(code), quote=False) + "</pre>", "\n".join(code)))
            continue
        para.append(l.strip())
        i += 1
    flush()
    return out


def weight(b, cpl):
    """Estimated height of a block in body lines at `cpl` characters per line."""
    kind, _, text = b
    if kind == "h1":
        return 3.2
    if kind in ("h2", "h3"):
        return 2.4
    if kind == "table":
        return 1.6 * (text.count("\n") + 1) + 1.0
    if kind == "pre":
        return 1.15 * (text.count("\n") + 1) + 1.4
    n = max(1, -(-len(text) // cpl))
    if kind == "note":
        return n + 2.0
    if kind == "li" or kind.startswith("oli"):
        return n + 0.35
    return n + 0.9


def paginate(blocks, cpl, cap):
    """Blocks given to pages in order; a page takes blocks while their estimated height fits."""
    pages, cur, h = [], [], 0.0
    for b in blocks:
        w = weight(b, cpl)
        # A heading is never left alone at the bottom of a page.
        if cur and (h + w > cap or (b[0].startswith("h") and h + w + 3 > cap)):
            pages.append(cur)
            cur, h = [], 0.0
        cur.append(b)
        h += w
    if cur:
        pages.append(cur)
    return pages


def page_html(page, two):
    out = []
    i = 0
    while i < len(page):
        kind, h, t = page[i]
        if kind == "li":
            items = []
            while i < len(page) and page[i][0] == "li":
                items.append("<li>" + page[i][1] + "</li>")
                i += 1
            out.append("<ul>" + "".join(items) + "</ul>")
            continue
        if kind.startswith("oli:"):
            start = kind.split(":")[1]
            items = []
            while i < len(page) and page[i][0].startswith("oli:"):
                items.append("<li>" + page[i][1] + "</li>")
                i += 1
            out.append('<ol start="%s">%s</ol>' % (start, "".join(items)))
            continue
        if kind == "table":
            rows = h
            cells = ["<tr>" + "".join("<th>%s</th>" % inline(c, False)[0] for c in rows[0]) + "</tr>"]
            for r in rows[1:]:
                cells.append("<tr>" + "".join("<td>%s</td>" % inline(c, False)[0] for c in r) + "</tr>")
            out.append("<table>" + "".join(cells) + "</table>")
            i += 1
            continue
        out.append(h)
        i += 1
    return '<section class="page">\n' + "\n".join(out) + "\n</section>"


CSS = """
@page { size: A4; margin: 22mm 18mm 20mm 18mm;
  @top-left { content: "%s"; font: 8pt Arial; color: #555; }
  @top-right { content: "Edition 3"; font: 8pt Arial; color: #555; }
  @bottom-center { content: "Page " counter(page) " of " counter(pages); font: 8pt Arial; color: #555; } }
body { font: 10.5pt/1.38 Georgia, serif; margin: 0; }
h1 { font: bold 17pt Arial, sans-serif; margin: 0 0 8pt 0; }
h2 { font: bold 13pt Arial, sans-serif; margin: 10pt 0 5pt 0; }
h3 { font: bold 11.5pt Arial, sans-serif; margin: 8pt 0 4pt 0; }
p { margin: 0 0 6pt 0; text-align: left; }
.box { border: 1px solid #888; padding: 4pt 6pt; margin: 4pt 0 8pt 0; background: #f2f2f2; }
ul, ol { margin: 0 0 6pt 0; padding-left: 18pt; }
li { margin: 0 0 2pt 0; }
table { border-collapse: collapse; margin: 4pt 0 8pt 0; font: 9pt Arial, sans-serif; }
th, td { border: 1px solid #999; padding: 2pt 5pt; text-align: left; white-space: nowrap; }
pre, code { font: 9pt Consolas, monospace; }
pre { margin: 4pt 0 8pt 0; }
section.page { break-after: page; }
section.page:last-child { break-after: auto; }
""" % HEADER

CSS2 = """
section.page { column-count: 2; column-gap: 9mm; column-fill: balance; }
h1, table, pre { column-span: all; }
body { font-size: 10pt; }
table { font-size: 8.5pt; }
"""


def doc_html(pages, two, title):
    css = CSS + (CSS2 if two else "")
    body = "\n".join(page_html(p, two) for p in pages)
    return ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n<title>%s</title>\n<style>%s</style>\n</head>\n<body>\n%s\n</body>\n</html>\n'
            % (html.escape(title), css, body))


def page_texts(pages):
    return ["\n".join(b[2] for b in p) for p in pages]


def expected_file(texts):
    out = []
    for n, t in enumerate(texts, 1):
        out.append("--- page %d ---" % n)
        out.append(t)
    return "\n".join(out) + "\n"


# Gold passages: a phrase that must be found on the page that holds it. Each is also the anchor
# of a development question (questions/pdf.json).
GOLD = [
    "keep the key with you while you work",
    "Running without water for more than 20 seconds",
    "report it to the dealer within seven",
    "Shaft seal type",
    "tighten them to 12 N m",
    "Within 60 seconds the pressure should rise",
    "Grease the motor bearings",
    "Leave the solution in the pump for 45 minutes",
    "hours = 1240",
    "It sounds like gravel in the pump",
    "the spiral shaped body of the pump",
    "Parts that fail under warranty are returned through the dealer",
]


def gold_rows(name, texts):
    rows = []
    for g in GOLD:
        hit = [n for n, t in enumerate(texts, 1) if g in t.replace("\n", " ")]
        if len(hit) != 1:
            sys.exit("gold passage %r found on pages %s of %s" % (g, hit, name))
        rows.append("%s\t%d\t%s" % (name, hit[0], g))
    return rows


# ---------------------------------------------------------------- a picture page

def png(width, height, rows):
    """A greyscale PNG from a list of byte rows."""
    raw = b"".join(b"\x00" + bytes(r) for r in rows)

    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


def scan_png():
    """A picture that looks like a scanned page: light paper with dark bars where lines of print
    would be."""
    w, h = 310, 430
    rows = []
    for y in range(h):
        line = (y - 30) // 11
        in_line = 30 <= y < 400 and (y - 30) % 11 < 6
        row = []
        for x in range(w):
            stop = 280 - (line * 37 % 70)
            word = (x // 5 + line * 3) % 7 != 0
            row.append(50 if in_line and 25 <= x < stop and word else 236)
        rows.append(row)
    return png(w, h, rows)


def insert_html(img_rel):
    p1 = ("<h1>Pressure vessel inspection record</h1><p>This record is kept with the AQ-40 service file. "
          "The inspection of the pressure vessel is carried out once a year by the installer, who "
          "checks the air charge, the membrane and the mounting bracket.</p><p>The next page is a "
          "scanned copy of the paper form signed at the last visit. It is a picture only and holds "
          "no text that a reader can take from it.</p>")
    p3 = ("<h1>Notes after the inspection</h1><p>The air charge was found at 1.3 bar and was raised to "
          "the correct value. The membrane showed no sign of wear. The mounting bracket bolts were "
          "tightened to 10 N m.</p>")
    css = ("@page { size: A4; margin: 20mm; } body { font: 11pt/1.4 Georgia, serif; } "
           "h1 { font: bold 16pt Arial; } section { break-after: page; } section:last-child { break-after: auto; } "
           "img { width: 160mm; height: 222mm; display: block; }")
    body = ("<section>%s</section>\n<section><img src=\"%s\" alt=\"\"></section>\n<section>%s</section>" % (p1, img_rel, p3))
    texts = [re.sub(r"<[^>]+>", "\n", p1), "", re.sub(r"<[^>]+>", "\n", p3)]
    texts = ["\n".join(x for x in t.split("\n") if x.strip()) for t in texts]
    return ('<!doctype html>\n<html lang="en"><head><meta charset="utf-8"><title>Inspection record</title><style>%s</style></head><body>\n%s\n</body></html>\n'
            % (css, body)), texts


# ---------------------------------------------------------------- Microsoft Print to PDF lines

def wrap(text, width):
    out = []
    for para in text.split("\n"):
        line = ""
        for w in para.split(" "):
            if line and len(line) + 1 + len(w) > width:
                out.append(line)
                line = w
            else:
                line = w if not line else line + " " + w
        out.append(line)
    return out


def step_html():
    md = open(MANUAL, encoding="utf-8").read()
    blocks = blocks_of(md)
    gold = []
    plan = []
    for name, two, cpl, cap in (("manual-1col", False, 92, 46.0), ("manual-2col", True, 44, 86.0)):
        pages = paginate(blocks, cpl, cap)
        write(os.path.join(SRC, name + ".html"), doc_html(pages, two, HEADER))
        texts = page_texts(pages)
        write(os.path.join(EXP, name + ".txt"), expected_file(texts))
        gold += gold_rows(name + ".pdf", texts)
        plan.append("%s.pdf\t%d" % (name, len(pages)))
    # The same pages printed by Microsoft Print to PDF from plain lines (no columns, no header).
    one = paginate(blocks, 92, 46.0)
    texts = page_texts(one)
    lines = []
    for n, t in enumerate(texts, 1):
        lines.append("--- page %d ---" % n)
        lines += wrap(t, 96)
    write(os.path.join(SRC, "manual-msprint.txt"), "\n".join(lines) + "\n")
    write(os.path.join(EXP, "manual-msprint.txt"), expected_file(texts))
    gold += gold_rows("manual-msprint.pdf", texts)
    plan.append("manual-msprint.pdf\t%d" % len(texts))
    # A record whose middle page is a picture only.
    write(os.path.join(SRC, "scan.png"), scan_png())
    h, texts = insert_html("scan.png")
    write(os.path.join(SRC, "inspection-scan.html"), h)
    write(os.path.join(EXP, "inspection-scan.txt"), expected_file(texts))
    plan.append("inspection-scan.pdf\t3")
    write(os.path.join(EXP, "gold.tsv"), "file\tpage\tpassage\n" + "\n".join(gold) + "\n")
    write(os.path.join(EXP, "pages.tsv"), "file\tpages\n" + "\n".join(plan) + "\n")
    print("html, expected text and gold passages written:", ", ".join(plan))


# ---------------------------------------------------------------- crafted PDFs

def rc4(key, data):
    s = list(range(256))
    j = 0
    for i in range(256):
        j = (j + s[i] + key[i % len(key)]) & 255
        s[i], s[j] = s[j], s[i]
    out = bytearray()
    i = j = 0
    for b in data:
        i = (i + 1) & 255
        j = (j + s[i]) & 255
        s[i], s[j] = s[j], s[i]
        out.append(b ^ s[(s[i] + s[j]) & 255])
    return bytes(out)


SBOX = []


def aes_tables():
    if SBOX:
        return
    p = q = 1
    sb = [0] * 256
    while True:
        p = p ^ ((p << 1) & 0xFF) ^ (0x1B if p & 0x80 else 0)
        q ^= q << 1
        q ^= q << 2
        q ^= q << 4
        q &= 0xFF
        if q & 0x80:
            q ^= 0x09
        x = q ^ ((q << 1) | (q >> 7)) & 0xFF ^ ((q << 2) | (q >> 6)) & 0xFF ^ ((q << 3) | (q >> 5)) & 0xFF ^ ((q << 4) | (q >> 4)) & 0xFF
        sb[p] = (x ^ 0x63) & 0xFF
        if p == 1:
            break
    sb[0] = 0x63
    SBOX.extend(sb)


def xt(a):
    return ((a << 1) ^ 0x1B) & 0xFF if a & 0x80 else a << 1


def aes_encrypt_block(key, block):
    aes_tables()
    nk = len(key) // 4
    nr = nk + 6
    w = [list(key[4 * i:4 * i + 4]) for i in range(nk)]
    rcon = 1
    for i in range(nk, 4 * (nr + 1)):
        t = list(w[i - 1])
        if i % nk == 0:
            t = [SBOX[b] for b in t[1:] + t[:1]]
            t[0] ^= rcon
            rcon = xt(rcon)
        elif nk > 6 and i % nk == 4:
            t = [SBOX[b] for b in t]
        w.append([a ^ b for a, b in zip(w[i - nk], t)])
    s = [list(block[4 * c:4 * c + 4]) for c in range(4)]

    def add(r):
        for c in range(4):
            for k in range(4):
                s[c][k] ^= w[4 * r + c][k]

    add(0)
    for r in range(1, nr + 1):
        for c in range(4):
            s[c] = [SBOX[b] for b in s[c]]
        t = [[s[(c + k) % 4][k] for k in range(4)] for c in range(4)]
        s[:] = t
        if r != nr:
            for c in range(4):
                a = s[c]
                x = a[0] ^ a[1] ^ a[2] ^ a[3]
                s[c] = [a[k] ^ x ^ xt(a[k] ^ a[(k + 1) % 4]) for k in range(4)]
        add(r)
    return bytes(b for c in s for b in c)


def aes_cbc(key, iv, data, pad=True):
    if pad:
        n = 16 - len(data) % 16
        data += bytes([n]) * n
    out = bytearray()
    prev = iv
    for i in range(0, len(data), 16):
        blk = bytes(a ^ b for a, b in zip(data[i:i + 16], prev))
        prev = aes_encrypt_block(key, blk)
        out += prev
    return bytes(out)


PAD = bytes.fromhex("28BF4E5E4E758A4164004E56FFFA01082E2E00B6D0683E802F0CA9FE6453697A")
FILE_ID = hashlib.md5(b"halvern aq-40 crafted").digest()


def iv_for(n):
    return hashlib.md5(b"iv %d" % n).digest()


def handler(rev):
    """(encrypt dictionary text, file key, method) for an empty user password and a fixed owner
    password, permissions: no printing, no copying (bits 3 and 5 clear)."""
    # Every bit set except printing (bit 3) and copying (bit 5); bits 1 and 2 are always clear.
    perms = struct.unpack("<i", struct.pack("<I", 0xFFFFFFFF & ~0x3 & ~0x4 & ~0x10))[0]
    owner_pw = b"halvern-owner"
    if rev == 6:
        file_key = hashlib.sha256(b"aes-256 file key").digest()
        u_salt, u_key_salt = b"usaltvl1", b"ukeysal1"
        o_salt, o_key_salt = b"osaltvl1", b"okeysal1"
        u_hash = hash_2b(b"", u_salt, b"")
        U = u_hash + u_salt + u_key_salt
        ue = aes_cbc(hash_2b(b"", u_key_salt, b""), bytes(16), file_key, pad=False)
        o_hash = hash_2b(owner_pw, o_salt, U)
        O = o_hash + o_salt + o_key_salt
        oe = aes_cbc(hash_2b(owner_pw, o_key_salt, U), bytes(16), file_key, pad=False)
        p = struct.pack("<i", perms) + bytes([0xFF] * 4) + b"Tadb" + b"rand"
        pe = aes_encrypt_block(file_key, p)
        d = ("<< /Filter /Standard /V 5 /R 6 /Length 256 /CF << /StdCF << /CFM /AESV3 /AuthEvent /DocOpen /Length 32 >> >> "
             "/StmF /StdCF /StrF /StdCF /O <%s> /U <%s> /OE <%s> /UE <%s> /Perms <%s> /P %d >>"
             % (O.hex(), U.hex(), oe.hex(), ue.hex(), pe.hex(), perms))
        return d, file_key, "aesv3"
    length = 5 if rev == 2 else 16
    # Owner entry (algorithm 3).
    h = hashlib.md5(owner_pw + PAD[:32 - len(owner_pw)]).digest()
    if rev >= 3:
        for _ in range(50):
            h = hashlib.md5(h[:length]).digest()
    okey = h[:length]
    O = rc4(okey, PAD)
    if rev >= 3:
        for i in range(1, 20):
            O = rc4(bytes(b ^ i for b in okey), O)
    # File key (algorithm 2) with the empty user password.
    h = hashlib.md5(PAD + O + struct.pack("<i", perms) + FILE_ID).digest()
    if rev >= 3:
        for _ in range(50):
            h = hashlib.md5(h[:length]).digest()
    key = h[:length]
    if rev == 2:
        U = rc4(key, PAD)
    else:
        u = rc4(key, hashlib.md5(PAD + FILE_ID).digest())
        for i in range(1, 20):
            u = rc4(bytes(b ^ i for b in key), u)
        U = u + bytes(16)
    if rev == 2:
        d = "<< /Filter /Standard /V 1 /R 2 /O <%s> /U <%s> /P %d >>" % (O.hex(), U.hex(), perms)
        return d, key, "rc4"
    if rev == 3:
        d = "<< /Filter /Standard /V 2 /R 3 /Length 128 /O <%s> /U <%s> /P %d >>" % (O.hex(), U.hex(), perms)
        return d, key, "rc4"
    d = ("<< /Filter /Standard /V 4 /R 4 /Length 128 /CF << /StdCF << /CFM /AESV2 /AuthEvent /DocOpen /Length 16 >> >> "
         "/StmF /StdCF /StrF /StdCF /O <%s> /U <%s> /P %d >>" % (O.hex(), U.hex(), perms))
    return d, key, "aesv2"


def hash_2b(pw, salt, udata):
    """The revision 6 password hash (ISO 32000-2, algorithm 2.B)."""
    k = hashlib.sha256(pw + salt + udata).digest()
    i = 0
    while True:
        k1 = (pw + k + udata) * 64
        e = aes_cbc(k[:16], k[16:32], k1, pad=False)
        r = sum(e[:16]) % 3
        k = [hashlib.sha256, hashlib.sha384, hashlib.sha512][r](e).digest()
        i += 1
        if i >= 64 and e[-1] <= i - 32:
            break
    return k[:32]


def obj_key(key, method, num, gen):
    if method == "aesv3":
        return key
    k = key + struct.pack("<I", num)[:3] + struct.pack("<H", gen)
    if method == "aesv2":
        k += b"sAlT"
    return hashlib.md5(k).digest()[:min(len(key) + 5, 16)]


def enc(key, method, num, gen, data):
    k = obj_key(key, method, num, gen)
    if method == "rc4":
        return rc4(k, data)
    iv = iv_for(num * 7 + len(data))
    return iv + aes_cbc(k, iv, data)


# A small tokenizer for the bodies of objects of a printed PDF: enough to find strings outside
# streams and to rewrite them.
def rewrite_strings(body, f):
    out = bytearray()
    i = 0
    n = len(body)
    while i < n:
        c = body[i:i + 1]
        if c == b"(":
            depth, j, s = 1, i + 1, bytearray()
            while j < n and depth:
                ch = body[j:j + 1]
                if ch == b"\\":
                    nx = body[j + 1:j + 2]
                    esc = {b"n": b"\n", b"r": b"\r", b"t": b"\t", b"b": b"\b", b"f": b"\f", b"(": b"(", b")": b")", b"\\": b"\\"}
                    if nx in esc:
                        s += esc[nx]
                        j += 2
                    elif nx.isdigit():
                        m = re.match(rb"[0-7]{1,3}", body[j + 1:j + 4])
                        s.append(int(m.group(0), 8) & 255)
                        j += 1 + len(m.group(0))
                    elif nx in (b"\r", b"\n"):
                        j += 2
                    else:
                        j += 2
                    continue
                if ch == b"(":
                    depth += 1
                elif ch == b")":
                    depth -= 1
                    if depth == 0:
                        break
                s += ch
                j += 1
            out += b"<" + f(bytes(s)).hex().encode() + b">"
            i = j + 1
            continue
        if c == b"<" and body[i + 1:i + 2] != b"<":
            j = body.index(b">", i)
            hx = re.sub(rb"\s", b"", body[i + 1:j])
            if len(hx) % 2:
                hx += b"0"
            out += b"<" + f(bytes.fromhex(hx.decode())).hex().encode() + b">"
            i = j + 1
            continue
        if c == b"<":
            out += b"<<"
            i += 2
            continue
        out += c
        i += 1
    return bytes(out)


def objects(pdf):
    """Every `n g obj ... endobj` of a file with a classic cross-reference table, in file order:
    (num, gen, dict text, stream bytes or None). Scans from the end of one object to the next, so
    bytes inside a stream are never taken for an object."""
    out = []
    pos = 0
    head_re = re.compile(rb"(\d+) (\d+) obj\s*")
    while True:
        m = head_re.search(pdf, pos)
        if not m:
            return out
        start = m.end()
        st = re.compile(rb"stream\r?\n").search(pdf, start)
        end = pdf.index(b"endobj", start)
        if st and st.start() < end:
            head = pdf[start:st.start()]
            lm = re.search(rb"/Length (\d+)( (\d+) 0 R)?", head)
            if lm.group(2):
                rm = re.search(rb"\n%d 0 obj\s*(\d+)" % int(lm.group(3)), pdf)
                ln = int(rm.group(1))
            else:
                ln = int(lm.group(1))
            data = pdf[st.end():st.end() + ln]
            end = pdf.index(b"endobj", st.end() + ln)
            out.append((int(m.group(1)), int(m.group(2)), head.rstrip(), data))
        else:
            out.append((int(m.group(1)), int(m.group(2)), pdf[start:end].rstrip(), None))
        pos = end + 6


def self_check():
    """Known answers for the ciphers written here: FIPS-197 appendix C and RFC 6229."""
    k = bytes(range(16))
    if aes_encrypt_block(k, bytes.fromhex("00112233445566778899aabbccddeeff")).hex() != "69c4e0d86a7b0430d8cdb78070b4c55a":
        sys.exit("aes-128 known answer failed")
    if aes_encrypt_block(bytes(range(32)), bytes.fromhex("00112233445566778899aabbccddeeff")).hex() != "8ea2b7ca516745bfeafc49904b496089":
        sys.exit("aes-256 known answer failed")
    if rc4(bytes([1, 2, 3, 4, 5]), bytes(16)).hex() != "b2396305f03dc027ccc3524a0a1118a8":
        sys.exit("rc4 known answer failed")


def encrypt_pdf(src, dst, rev):
    pdf = open(src, "rb").read()
    if b"/ObjStm" in pdf or b"/XRef" in pdf:
        sys.exit("encrypt: %s uses object or cross-reference streams; this crafter handles classic tables only" % src)
    tm = re.search(rb"trailer\s*<<(.*?)>>\s*startxref", pdf, re.S)
    trailer = tm.group(1)
    root = re.search(rb"/Root (\d+ \d+ R)", trailer).group(1)
    info = re.search(rb"/Info (\d+ \d+ R)", trailer)
    d, key, method = handler(rev)
    objs = objects(pdf)
    out = bytearray(b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n")
    offs = {}
    maxn = max(o[0] for o in objs)
    enc_num = maxn + 1
    for num, gen, head, data in objs:
        offs[num] = len(out)
        f = lambda s, num=num, gen=gen: enc(key, method, num, gen, s)
        if data is None:
            out += b"%d %d obj\n" % (num, gen) + rewrite_strings(head, f) + b"\nendobj\n"
        else:
            e = f(data)
            head = re.sub(rb"/Length (\d+ \d+ R|\d+)", b"/Length %d" % len(e), head)
            out += b"%d %d obj\n" % (num, gen) + rewrite_strings(head, f) + b"\nstream\n" + e + b"\nendstream\nendobj\n"
    offs[enc_num] = len(out)
    out += b"%d 0 obj\n" % enc_num + d.encode() + b"\nendobj\n"
    xref = len(out)
    size = enc_num + 1
    out += b"xref\n0 %d\n0000000000 65535 f \n" % size
    for n in range(1, size):
        if n in offs:
            out += b"%010d 00000 n \n" % offs[n]
        else:
            out += b"0000000000 65535 f \n"
    out += b"trailer\n<< /Size %d /Root %s " % (size, root)
    if info:
        out += b"/Info %s " % info.group(1)
    out += b"/Encrypt %d 0 R /ID [<%s> <%s>] >>\nstartxref\n%d\n%%%%EOF\n" % (enc_num, FILE_ID.hex().encode(), FILE_ID.hex().encode(), xref)
    write(dst, bytes(out))


# The older filters, written here because the printers above use Flate only.
def a85(data):
    out = bytearray()
    for i in range(0, len(data), 4):
        chunk = data[i:i + 4]
        n = len(chunk)
        v = struct.unpack(">I", chunk + bytes(4 - n))[0]
        if v == 0 and n == 4:
            out += b"z"
            continue
        digits = []
        for _ in range(5):
            digits.append(v % 85)
            v //= 85
        out += bytes(33 + d for d in reversed(digits))[:n + 1]
    return bytes(out) + b"~>"


def lzw(data):
    """LZW with the PDF defaults (early change 1), codes from 9 to 12 bits."""
    table = {bytes([i]): i for i in range(256)}
    nxt = 258
    width = 9
    bits = 0
    nbits = 0
    out = bytearray()

    def emit(code):
        nonlocal bits, nbits
        bits = (bits << width) | code
        nbits += width
        while nbits >= 8:
            nbits -= 8
            out.append((bits >> nbits) & 255)

    emit(256)
    w = b""
    for b in data:
        wc = w + bytes([b])
        if wc in table:
            w = wc
            continue
        emit(table[w])
        table[wc] = nxt
        nxt += 1
        if nxt + 1 > (1 << width) and width < 12:
            width += 1
        if nxt >= 4094:
            emit(256)
            table = {bytes([i]): i for i in range(256)}
            nxt = 258
            width = 9
        w = bytes([b])
    if w:
        emit(table[w])
        nxt += 1
        if nxt + 1 > (1 << width) and width < 12:
            width += 1
    emit(257)
    if nbits:
        out.append((bits << (8 - nbits)) & 255)
    return bytes(out)


def runlength(data):
    out = bytearray()
    i = 0
    while i < len(data):
        j = i
        while j + 1 < len(data) and data[j + 1] == data[i] and j - i < 127:
            j += 1
        if j > i:
            out += bytes([257 - (j - i + 1), data[i]])
            i = j + 1
            continue
        k = i
        while k + 1 < len(data) and data[k + 1] != data[k] and k - i < 127:
            k += 1
        out += bytes([k - i]) + data[i:k + 1]
        i = k + 1
    return bytes(out) + b"\x80"


def png_up(data, cols):
    """PNG Up predictor rows for a cross-reference stream."""
    out = bytearray()
    prev = bytes(cols)
    for i in range(0, len(data), cols):
        row = data[i:i + cols]
        out += b"\x02" + bytes((a - b) & 255 for a, b in zip(row, prev))
        prev = row
    return bytes(out)


def build(objs, root, xref_stream=False, objstm=None):
    """A PDF from {num: bytes body}; with `xref_stream`, objects listed in `objstm` go into one
    object stream and the cross-reference is a stream with a PNG Up predictor."""
    out = bytearray(b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n")
    offs = {}
    objstm = objstm or []
    nums = sorted(objs)
    size = max(nums) + 3
    stm_num = size - 2
    xref_num = size - 1
    for n in nums:
        if n in objstm:
            continue
        offs[n] = len(out)
        out += b"%d 0 obj\n" % n + objs[n] + b"\nendobj\n"
    if objstm:
        heads, body = [], bytearray()
        for n in objstm:
            heads.append(b"%d %d" % (n, len(body)))
            body += objs[n] + b"\n"
        head = b" ".join(heads) + b"\n"
        data = zlib.compress(head + bytes(body))
        offs[stm_num] = len(out)
        out += (b"%d 0 obj\n<< /Type /ObjStm /N %d /First %d /Filter /FlateDecode /Length %d >>\nstream\n" % (stm_num, len(objstm), len(head), len(data))
                + data + b"\nendstream\nendobj\n")
    if not xref_stream:
        xref = len(out)
        out += b"xref\n0 %d\n0000000000 65535 f \n" % size
        for n in range(1, size):
            out += (b"%010d 00000 n \n" % offs[n]) if n in offs else b"0000000000 65535 f \n"
        out += b"trailer\n<< /Size %d /Root %d 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (size, root, xref)
        return bytes(out)
    xref = len(out)
    offs[xref_num] = xref
    rows = bytearray()
    for n in range(size):
        if n == 0:
            rows += bytes([0, 0, 0, 0, 255, 255])
        elif n in objstm:
            rows += bytes([2]) + struct.pack(">I", stm_num)[1:] + struct.pack(">H", objstm.index(n))
        elif n in offs:
            rows += bytes([1]) + struct.pack(">I", offs[n])[1:] + b"\x00\x00"
        else:
            rows += bytes([0, 0, 0, 0, 0, 0])
    data = zlib.compress(png_up(bytes(rows), 6))
    out += (b"%d 0 obj\n<< /Type /XRef /Size %d /W [1 3 2] /Root %d 0 R /Filter /FlateDecode /DecodeParms << /Predictor 12 /Columns 6 >> /Length %d >>\nstream\n"
            % (xref_num, size, root, len(data)) + data + b"\nendstream\nendobj\nstartxref\n%d\n%%%%EOF\n" % xref)
    return bytes(out)


FEATURES_TEXT = [
    "Older filters and fonts",
    "This page is written with a simple font whose encoding replaces some codes by glyph names.",
    "Its content stream is packed with LZW and then written as ASCII85 text.",
    "A composite font with a ToUnicode map",
    "This line is drawn through two byte codes that only the map turns into letters.",
    "The second page is packed with run length coding and written as hexadecimal digits.",
]


def craft_features():
    """Two pages: page 1 through a Type1 font (no widths, one of the standard fonts) with a
    /Differences encoding that moves letters to other codes, its content LZW then ASCII85 coded;
    page 2 through a Type0 font with Identity-H codes and a ToUnicode map, its content run length
    then ASCII hex coded. Objects 3 to 8 sit in an object stream; the cross-reference is a stream."""
    # Simple font: codes 1 to 26 hold the lowercase letters by glyph name.
    diffs = "/Differences [1 " + " ".join("/" + chr(97 + i) for i in range(26)) + "]"

    def simple(s):
        return bytes(ord(c) - 96 if "a" <= c <= "z" else ord(c) for c in s)

    def lit(b):
        return b"(" + b"".join(b"\\%03o" % x if x < 32 or x in b"()\\" else bytes([x]) for x in b) + b")"

    c1 = (b"BT /F1 16 Tf 72 770 Td " + lit(simple(FEATURES_TEXT[0])) + b" Tj ET\n"
          b"BT /F1 11 Tf 72 740 Td 14 TL " + lit(simple(FEATURES_TEXT[1])) + b" Tj T* " + lit(simple(FEATURES_TEXT[2])) + b" Tj ET\n")
    # Composite font: each character is a two byte code (its position in the alphabet below plus 0x0100).
    alpha = sorted(set("".join(FEATURES_TEXT[3:])))
    code = {ch: 0x0100 + i for i, ch in enumerate(alpha)}

    def cid(s):
        return b"<" + "".join("%04X" % code[c] for c in s).encode() + b">"

    c2 = (b"BT /F2 16 Tf 72 770 Td " + cid(FEATURES_TEXT[3]) + b" Tj ET\n"
          b"BT /F2 11 Tf 72 740 Td 14 TL " + cid(FEATURES_TEXT[4]) + b" Tj T* " + cid(FEATURES_TEXT[5]) + b" Tj ET\n")
    cmap = ["/CIDInit /ProcSet findresource begin 12 dict begin begincmap /CMapName /Crafted def",
            "1 begincodespacerange <0000> <FFFF> endcodespacerange", "%d beginbfchar" % len(alpha)]
    cmap += ["<%04X> <%04X>" % (code[c], ord(c)) for c in alpha]
    cmap += ["endbfchar", "endcmap CMapName currentdict /CMap defineresource pop end end"]
    cmap = "\n".join(cmap).encode()
    s1 = a85(lzw(c1))
    s2 = runlength(c2).hex().upper().encode() + b">"
    objs = {
        1: b"<< /Type /Catalog /Pages 2 0 R >>",
        2: b"<< /Type /Pages /Kids [9 0 R 10 0 R] /Count 2 /MediaBox [0 0 612 792] /Resources << /Font << /F1 3 0 R /F2 4 0 R >> >> >>",
        3: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding << /Type /Encoding /BaseEncoding /WinAnsiEncoding " + diffs.encode() + b" >> >>",
        4: b"<< /Type /Font /Subtype /Type0 /BaseFont /Crafted /Encoding /Identity-H /DescendantFonts [5 0 R] /ToUnicode 11 0 R >>",
        5: b"<< /Type /Font /Subtype /CIDFontType2 /BaseFont /Crafted /CIDSystemInfo << /Registry (Adobe) /Ordering (Identity) /Supplement 0 >> /DW 500 /W [256 [600 600 600]] /FontDescriptor 6 0 R >>",
        6: b"<< /Type /FontDescriptor /FontName /Crafted /Flags 32 /FontBBox [0 -200 1000 900] /ItalicAngle 0 /Ascent 900 /Descent -200 /CapHeight 700 /StemV 80 >>",
        7: b"<< /Producer (crafted by make_pdf.py) >>",
        8: b"<< /Title (Crafted features) >>",
        9: b"<< /Type /Page /Parent 2 0 R /Contents 12 0 R >>",
        10: b"<< /Type /Page /Parent 2 0 R /Contents 13 0 R >>",
        11: b"<< /Length %d >>\nstream\n" % len(cmap) + cmap + b"\nendstream",
        12: b"<< /Filter [/ASCII85Decode /LZWDecode] /Length %d >>\nstream\n" % len(s1) + s1 + b"\nendstream",
        13: b"<< /Filter [/ASCIIHexDecode /RunLengthDecode] /Length %d >>\nstream\n" % len(s2) + s2 + b"\nendstream",
    }
    pdf = build(objs, 1, xref_stream=True, objstm=[3, 4, 5, 6, 7, 8])
    write(os.path.join(OUT, "crafted-features.pdf"), pdf)
    write(os.path.join(EXP, "crafted-features.txt"), expected_file(["\n".join(FEATURES_TEXT[:3]), "\n".join(FEATURES_TEXT[3:])]))


def craft_nounicode():
    """One page through a composite font with Identity-H codes, no ToUnicode map and no embedded
    font program: nothing maps its glyphs to characters, so the page must be reported unreadable."""
    text = "Glyph codes without any map to characters"
    codes = b"<" + "".join("%04X" % (0x0300 + (ord(c) * 7) % 200) for c in text).encode() + b">"
    c = b"BT /F1 12 Tf 72 720 Td " + codes + b" Tj ET\n"
    objs = {
        1: b"<< /Type /Catalog /Pages 2 0 R >>",
        2: b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        3: b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 6 0 R >>",
        4: b"<< /Type /Font /Subtype /Type0 /BaseFont /NoMap /Encoding /Identity-H /DescendantFonts [5 0 R] >>",
        5: b"<< /Type /Font /Subtype /CIDFontType2 /BaseFont /NoMap /CIDSystemInfo << /Registry (Adobe) /Ordering (Identity) /Supplement 0 >> /DW 600 >>",
        6: b"<< /Length %d >>\nstream\n" % len(c) + c + b"\nendstream",
    }
    write(os.path.join(OUT, "crafted-nomap.pdf"), build(objs, 1))


def step_craft():
    self_check()
    src = os.path.join(OUT, "manual-1col.pdf")
    for rev, name in ((2, "rc4-40"), (3, "rc4-128"), (4, "aes-128"), (6, "aes-256")):
        encrypt_pdf(src, os.path.join(OUT, "manual-1col-locked-%s.pdf" % name), rev)
    craft_features()
    craft_nounicode()
    print("crafted: four encrypted copies, crafted-features.pdf, crafted-nomap.pdf")


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else ""
    if step == "html":
        step_html()
    elif step == "craft":
        step_craft()
    else:
        sys.exit("usage: make_pdf.py html|craft")
