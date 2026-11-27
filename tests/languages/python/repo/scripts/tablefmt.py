"""Plain text tables with columns padded to their widest cell."""

SEPARATOR = " | "


def render(header, rows):
    table = [list(map(str, header))] + [list(map(str, r)) for r in rows]
    widths = [max(len(row[i]) for row in table) for i in range(len(header))]
    lines = [SEPARATOR.join(cell.ljust(w) for cell, w in zip(row, widths)) for row in table]
    lines.insert(1, "-+-".join("-" * w for w in widths))
    return "\n".join(lines)
