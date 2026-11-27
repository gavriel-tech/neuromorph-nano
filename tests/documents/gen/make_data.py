"""Fixed-seed generator for corpus/data/ and the two special-encoding notes.

Run from anywhere:  python make_data.py
Writes:
  ../corpus/data/parts.csv          UTF-8, CRLF line ends (RFC 4180), header on line 1
  ../corpus/data/suppliers.tsv      UTF-8, LF
  ../corpus/data/orders.jsonl       UTF-8, LF
  ../corpus/data/pump-service.log   UTF-8, LF
  ../corpus/notes/2026-10-09-site-visit-report.txt   UTF-16 LE with byte order mark, CRLF
  ../corpus/notes/2026-10-22-supplier-visit.txt      Windows-1252, CRLF
Standard library only. Running it again gives byte-identical files.
"""
import csv
import io
import json
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.join(HERE, "..", "corpus")
DATA = os.path.join(CORPUS, "data")
NOTES = os.path.join(CORPUS, "notes")


def write_bytes(path, data):
    with open(path, "wb") as f:
        f.write(data)


# Section: suppliers
def make_suppliers(rng):
    first = ["Ashby", "Brindle", "Corvane", "Dunmore", "Elstow", "Farrow", "Gilden", "Harrow",
             "Islay", "Jarrow", "Kestle", "Lanark", "Merrow", "Northam", "Orwell", "Penhale",
             "Quarry", "Redcar", "Selby", "Tamar"]
    second = ["Seals", "Fasteners", "Electric", "Valves", "Fittings", "Bearings", "Supply",
              "Components", "Hydraulics", "Trade"]
    countries = ["UK", "Ireland", "France", "Germany", "Netherlands", "Poland", "Spain", "Italy"]
    rows = []
    used = set()
    for i in range(1, 41):
        sid = "S%03d" % i
        while True:
            name = rng.choice(first) + " " + rng.choice(second)
            if name not in used:
                used.add(name)
                break
        country = rng.choice(countries)
        lead = rng.randint(3, 30)
        if sid == "S014":
            name, country, lead = "Corvane Seals", "UK", 25
        if sid == "S022":
            name, country, lead = "Penhale Seals", "France", 6
        slug = name.lower().replace(" ", "-")
        rows.append([sid, name, country, str(lead), "orders@%s.example.com" % slug])
    out = io.StringIO()
    out.write("supplier_id\tname\tcountry\tlead_time_days\tcontact_email\n")
    for r in rows:
        out.write("\t".join(r) + "\n")
    write_bytes(os.path.join(DATA, "suppliers.tsv"), out.getvalue().encode("utf-8"))


# Section: parts
CATEGORIES = {
    "seal": ["O-ring", "lip seal", "gasket", "seal washer"],
    "fastener": ["hex bolt", "nut", "washer", "screw", "stud"],
    "electrical": ["cable gland", "terminal block", "fuse", "relay", "cable"],
    "valve": ["ball valve", "gate valve", "non-return valve", "float valve"],
    "fitting": ["elbow", "tee", "union", "reducer", "nipple"],
    "bearing": ["ball bearing", "bush", "thrust washer"],
    "hose": ["hose", "hose tail", "hose clamp"],
}
SIZES = ["6 mm", "8 mm", "10 mm", "12 mm", "15 mm", "22 mm", "28 mm", "1/2 inch", "3/4 inch",
         "1 inch"]
MATERIALS = ["brass", "steel", "stainless", "nylon", "nitrile", "EPDM", "zinc plated"]

# rows placed at fixed part numbers: part_no -> (name, category, stock, price, supplier, bin, date)
FIXED = {
    "P-10012": ("AQ-40 shaft seal kit, 12 mm", "seal", 15, "19.10", "S022", "A-03", "2026-10-29"),
    "P-10020": ("AQ-40 impeller, brass, 7 vane", "pump spare", 4, "46.00", "S031", "A-05",
                "2026-09-24"),
    "P-10040": ("Motor run capacitor 16 microfarad", "electrical", 5, "12.75", "S033", "C-02",
                "2026-10-27"),
    "P-10050": ("AQ-40 flow sensor", "pump spare", 2, "38.20", "S031", "A-06", "2026-09-24"),
    "P-10088": ("Pressure gauge 0-10 bar, 63 mm", "gauge", 0, "21.40", "S035", "B-11",
                "2026-09-24"),
    "P-10142": ("Inlet cap O-ring, 52 mm", "seal", 40, "0.85", "S014", "A-14", "2026-09-24"),
    "P-10175": ('Hose clamp 1" stainless', "hose", 120, "0.95", "S038", "D-09", "2026-09-24"),
    "P-10203": ("Citric acid descaler 1 kg", "consumable", 0, "4.40", "S036", "E-01",
                "2026-10-23"),
    "P-10219": ("Test gauge 0-6 bar, glycerine filled", "gauge", 3, "34.90", "S035", "B-12",
                "2026-09-24"),
    "P-10250": ("Rubber feet, set of four", "pump spare", 0, "7.30", "S031", "A-07",
                "2026-09-24"),
    "P-10266": ("Lithium bearing grease 400 g", "consumable", 9, "6.80", "S036", "E-02",
                "2026-09-24"),
    "P-10291": ("Contact thermometer, digital", "tool", 1, "189.00", "S040", "F-01",
                "2026-09-24"),
}


def make_parts(rng):
    rows = []
    for i in range(1, 301):
        pno = "P-%05d" % (10000 + i)
        if pno in FIXED:
            name, cat, stock, price, sup, bin_, date = FIXED[pno]
            rows.append([pno, name, cat, str(stock), price, sup, bin_, date])
            continue
        cat = rng.choice(sorted(CATEGORIES))
        kind = rng.choice(CATEGORIES[cat])
        name = "%s %s %s" % (kind.capitalize(), rng.choice(SIZES), rng.choice(MATERIALS))
        if rng.random() < 0.03:
            name = "%s, %s, %s" % (kind.capitalize(), rng.choice(SIZES), rng.choice(MATERIALS))
        stock = rng.randint(1, 250)
        price = "%.2f" % (rng.randint(15, 4000) / 100.0)
        sup = "S%03d" % rng.randint(1, 30)
        bin_ = "%s-%02d" % (rng.choice("ABCD"), rng.randint(1, 20))
        date = "2026-%02d-%02d" % (rng.choice([8, 9]), rng.randint(1, 28))
        rows.append([pno, name, cat, str(stock), price, sup, bin_, date])
    out = io.StringIO()
    w = csv.writer(out, lineterminator="\r\n", quoting=csv.QUOTE_MINIMAL)
    w.writerow(["part_no", "name", "category", "stock", "unit_price", "supplier_id", "bin",
                "last_counted"])
    w.writerows(rows)
    write_bytes(os.path.join(DATA, "parts.csv"), out.getvalue().encode("utf-8"))
    return rows


# Section: orders
def make_orders(rng, parts):
    fixed = [
        ("2026-09-02", "P-10012", 10, "S014", "delivered"),
        ("2026-09-24", "P-10088", 2, "S035", "open"),
        ("2026-10-06", "P-10203", 5, "S036", "cancelled"),
        ("2026-10-20", "P-10012", 15, "S022", "delivered"),
        ("2026-10-27", "P-10040", 2, "S033", "delivered"),
        ("2026-11-04", "P-10012", 50, "S022", "open"),
    ]
    items = []
    general = [p for p in parts if p[0] not in FIXED]
    for _ in range(94):
        p = rng.choice(general)
        date = "2026-%02d-%02d" % (rng.choice([8, 9, 10]), rng.randint(1, 28))
        status = rng.choice(["delivered", "delivered", "delivered", "shipped", "open"])
        items.append((date, p[0], rng.randint(1, 60), p[5], status))
    items[17] = (items[17][0], items[17][1], items[17][2], items[17][3], "cancelled")
    items[58] = (items[58][0], items[58][1], items[58][2], items[58][3], "cancelled")
    items.extend(fixed)
    items.sort(key=lambda t: (t[0], t[1]))
    out = io.StringIO()
    for n, (date, pno, qty, sup, status) in enumerate(items, 1):
        obj = {"order_id": "O-2026-%04d" % n, "date": date, "part_no": pno, "qty": qty,
               "supplier_id": sup, "status": status}
        out.write(json.dumps(obj) + "\n")
    write_bytes(os.path.join(DATA, "orders.jsonl"), out.getvalue().encode("utf-8"))


# Section: log
def make_log(rng):
    lines = []
    t = 6 * 3600  # seconds since midnight, 06:00:00

    def stamp(sec):
        return "2026-10-03T%02d:%02d:%02dZ" % (sec // 3600, (sec // 60) % 60, sec % 60)

    def bench(p):
        return "bench1" if p <= 3 else "bench2"

    special = {
        9 * 3600 + 12 * 60: "stack",
        11 * 3600 + 47 * 60: "strainer",
        14 * 3600 + 30 * 60: "burst",
        15 * 3600 + 20 * 60: "starts",
    }
    done = set()
    lines.append(stamp(t) + " INFO logsvc service started, version 2.4.1, benches=2")
    while len(lines) < 398:
        t += rng.randint(60, 140)
        for at, what in special.items():
            if what in done or t < at:
                continue
            done.add(what)
            if what == "stack":
                s = at + 7
                lines.append(stamp(s) + " ERROR logsvc failed to rotate log file pump-service.log")
                lines.append("    at logsvc::rotate::rename_current (rotate.rs:88)")
                lines.append("    at logsvc::rotate::run (rotate.rs:41)")
                lines.append("    at logsvc::main_loop (main.rs:120)")
                lines.append("    caused by: access denied, file is open in another program")
                lines.append(stamp(s + 3) + " WARN logsvc rotation skipped, will retry at next hour")
            elif what == "strainer":
                lines.append(stamp(at + 30) + " WARN bench2 pump=5 strainer pressure drop high,"
                             " clean strainer")
            elif what == "burst":
                s = at + 5
                for k in range(24):
                    lines.append(stamp(s + 2 * k) + " ERROR bench1 pump=2 pressure sensor read"
                                 " timeout after 500 ms")
                lines.append(stamp(at + 62) + " ERROR bench1 pump=2 stopped, fault code 5"
                             " (pressure switch fault)")
                lines.append(stamp(at + 6 * 60 + 14) + " INFO bench1 pump=2 fault cleared by"
                             " operator")
                lines.append(stamp(at + 6 * 60 + 20) + " INFO bench1 pump=2 started")
                t = at + 7 * 60
            elif what == "starts":
                lines.append(stamp(at + 12) + " WARN bench2 pump=6 starts per hour 14,"
                             " above limit 12")
        p = rng.randint(1, 6)
        r = rng.random()
        if r < 0.80:
            flow = 38.0 + rng.randint(0, 40) / 10.0
            pres = 3.6 + rng.randint(0, 4) / 10.0
            lines.append(stamp(t) + " INFO %s pump=%d flow=%.1f lpm pressure=%.1f bar"
                         % (bench(p), p, flow, pres))
        elif r < 0.93:
            lines.append(stamp(t) + " INFO %s pump=%d %s" % (bench(p), p,
                         rng.choice(["started", "stopped, demand met"])))
        else:
            temp = 50 + rng.randint(0, 9)
            lines.append(stamp(t) + " INFO %s pump=%d housing temperature %d C"
                         % (bench(p), p, temp))
    lines.append(stamp(t + 60) + " INFO logsvc daily summary: 6 pumps, 1 fault stop")
    lines.append(stamp(t + 61) + " INFO logsvc service stopping")
    write_bytes(os.path.join(DATA, "pump-service.log"), ("\n".join(lines) + "\n").encode("utf-8"))


# Section: special notes
SITE_VISIT = """Site visit report: site B
=========================

Date: 9 October 2026. Visit to site B to check pumps 3 and 4 after four weeks of the trial.

Findings
--------

Pump 3 flow has dropped from 39 to 36 litres per minute at the site pressure.
White scale is visible inside the inlet cap and on the strainer mesh.
Pump 4 runs dry and quiet since its seal change.
The pump room is unheated. The site expects frost from late November.

Actions
-------

Send a water sample from site B to the lab for a hardness test.
Bring citric acid and a bucket on the next visit in case pump 3 needs descaling.
Remind the site to drain both pumps if frost is forecast before we collect them.
"""

SUPPLIER_VISIT = """Supplier visit: Penhale Seals
============================

Date: 22 October 2026. Visit to the works of supplier S022 after the switch of seal kits to them.

What we saw
-----------

The seal faces are lapped on site and checked on an optical flat; the works calls the finish
über flat, which they say beats the old kits.
Finished kits are stored at 18°C in a dry room, each bag dated.
We talked over the order terms at the café next to the works.

Terms
-----

Price per kit £19.10 for up to 49 kits, £17.95 for 50 kits or more.
Lead time 6 working days, 3 for urgent orders at extra cost.
They will add a spare volute O-ring to every kit from November.
"""


def make_special_notes():
    text16 = SITE_VISIT.replace("\n", "\r\n")
    write_bytes(os.path.join(NOTES, "2026-10-09-site-visit-report.txt"),
                b"\xff\xfe" + text16.encode("utf-16-le"))
    text1252 = SUPPLIER_VISIT.replace("\n", "\r\n")
    write_bytes(os.path.join(NOTES, "2026-10-22-supplier-visit.txt"), text1252.encode("cp1252"))


def main():
    os.makedirs(DATA, exist_ok=True)
    os.makedirs(NOTES, exist_ok=True)
    make_suppliers(random.Random(4012))
    parts = make_parts(random.Random(4040))
    make_orders(random.Random(4050), parts)
    make_log(random.Random(4075))
    make_special_notes()
    print("data and special notes written")


if __name__ == "__main__":
    main()
