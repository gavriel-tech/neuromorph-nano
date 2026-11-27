import csv
from pathlib import Path

from ..models import Item, Warehouse

FIELDS = ("sku", "name", "units")


class CsvStore:
    """A warehouse saved as one CSV file per site."""

    def __init__(self, folder: Path) -> None:
        self.folder = folder

    def path_for(self, site: str) -> Path:
        return self.folder / f"{site}.csv"

    def save(self, wh: Warehouse) -> Path:
        p = self.path_for(wh.site)
        with p.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(FIELDS)
            for item in wh._items.values():
                w.writerow([item.sku, item.name, item.units])
        return p

    def load(self, site: str) -> Warehouse:
        wh = Warehouse(site)
        with self.path_for(site).open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                wh.receive(row["sku"], row["name"], int(row["units"]))
        return wh
