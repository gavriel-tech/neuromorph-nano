"""Writes formats/encodings/: one short note in every text encoding the reader decodes, plus files
the format table must skip, each with its expected outcome in expected.json. Standard library only;
running it again gives byte-identical files."""
import json, os, random

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "formats", "encodings")
os.makedirs(OUT, exist_ok=True)

TEXT = "Café notes\r\n\r\nThe boiler ran at 25°C for £4 a day, über quiet.\r\nLine three.\r\n"

def put(name, data):
    with open(os.path.join(OUT, name), "wb") as f:
        f.write(data)

put("utf8.txt", TEXT.encode("utf-8"))
put("utf8-bom.txt", b"\xef\xbb\xbf" + TEXT.encode("utf-8"))
put("utf16le.txt", b"\xff\xfe" + TEXT.encode("utf-16-le"))
put("utf16be.txt", b"\xfe\xff" + TEXT.encode("utf-16-be"))
put("utf16le-nobom.txt", TEXT.encode("utf-16-le"))
put("cp1252.txt", TEXT.encode("cp1252"))
put("notes", TEXT.encode("utf-8"))
rng = random.Random(7)
put("blob.dat", bytes([0, 1, 2, 3, 0, 0, 255, 254, 7]) + bytes(rng.randrange(256) for _ in range(500)))
put("picture.png", b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR" + bytes(16))
put("report.pdf", b"%PDF-1.4\n% made up\n1 0 obj << >> endobj\n%%EOF\n")
put("disguised.txt", b"%PDF-1.4\n% a pdf with a text name\n%%EOF\n")
put("letter.docx", b"PK\x03\x04" + bytes(30))
put("archive.txt", b"PK\x03\x04" + bytes(30))
put("tool.exe", b"MZ\x90\x00" + bytes(60))
put("Cargo.lock", b"# lock file\nversion = 3\n")
put("app.min.js", (b"var a=1;" * 300) + b"\n")
put(".env", b"TOKEN=not-a-real-value\n")
put("big.txt", (b"filler line of the large file\n") * 20000)

expected = {
    "note": "What the reader must do with each file of formats/encodings/: read it (decoder, encoding, decoded text equal to text.txt) or skip it with the reason.",
    "text": "text.txt",
    "files": [
        {"path": "utf8.txt", "decoder": "text", "encoding": "utf-8"},
        {"path": "utf8-bom.txt", "decoder": "text", "encoding": "utf-8-bom"},
        {"path": "utf16le.txt", "decoder": "text", "encoding": "utf-16le"},
        {"path": "utf16be.txt", "decoder": "text", "encoding": "utf-16be"},
        {"path": "utf16le-nobom.txt", "decoder": "text", "encoding": "utf-16le"},
        {"path": "cp1252.txt", "decoder": "text", "encoding": "windows-1252"},
        {"path": "notes", "decoder": "text", "encoding": "utf-8"},
        {"path": "README.md", "decoder": "markdown", "encoding": "utf-8", "own_text": True},
        {"path": "blob.dat", "decoder": "raw", "encoding": "binary", "general": "raw", "own_text": True},
        {"path": "picture.png", "decoder": "raw", "encoding": "binary", "general": "raw", "own_text": True},
        {"path": "report.pdf", "decoder": "raw", "encoding": "utf-8", "general": "raw", "own_text": True},
        {"path": "disguised.txt", "decoder": "raw", "encoding": "utf-8", "general": "raw", "own_text": True},
        {"path": "letter.docx", "decoder": "raw", "encoding": "binary", "general": "raw", "own_text": True},
        {"path": "archive.txt", "decoder": "raw", "encoding": "binary", "general": "raw", "own_text": True},
        {"path": "tool.exe", "decoder": "raw", "encoding": "binary", "general": "raw", "own_text": True},
        {"path": "Cargo.lock", "skip": "lock file (generated): not read"},
        {"path": "app.min.js", "skip": "minified file (generated): not read"},
        {"path": ".env", "skip": "secret store (.env): never read"},
        {"path": "big.txt", "skip": "too large: over 512 KB"},
        {"path": "expected.json", "decoder": "json", "encoding": "utf-8", "own_text": True},
        {"path": "text.txt", "decoder": "text", "encoding": "utf-8", "own_text": True},
    ],
}
with open(os.path.join(OUT, "expected.json"), "w", encoding="utf-8", newline="\n") as f:
    json.dump(expected, f, indent=1)
    f.write("\n")
with open(os.path.join(OUT, "text.txt"), "w", encoding="utf-8", newline="") as f:
    f.write(TEXT)
