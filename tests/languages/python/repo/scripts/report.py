"""Prints a stock report for every site found in the data folder."""
from pathlib import Path

import tablefmt
from inventory.storage.csvstore import CsvStore


def site_rows(folder: Path):
    store = CsvStore(folder)
    for csv_path in sorted(folder.glob("*.csv")):
        wh = store.load(csv_path.stem)
        yield [wh.site, len(wh._items), wh.total_units]


if __name__ == "__main__":
    print(tablefmt.render(["site", "items", "units"], list(site_rows(Path("data")))))
