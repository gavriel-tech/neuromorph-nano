"""Items and the warehouse that holds them.

def not_a_function(): this line is inside the module docstring.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Shelves hold at most this many units of one item.
SHELF_CAPACITY = 240
REORDER_FRACTION = 0.25


@dataclass
class Item:
    sku: str
    name: str
    units: int = 0
    tags: list[str] = field(default_factory=list)

    def shelves_needed(self) -> int:
        # Ceiling division: a partly filled shelf still takes a shelf.
        return -(-self.units // SHELF_CAPACITY)

    def label(self) -> str:
        return f"{self.sku}: {self.name} ({self.units} units, tags {', '.join(self.tags) or 'none'})"


class StockError(ValueError):
    """Raised when a removal would take stock below zero."""


class Warehouse:
    """All items of one site, by SKU."""

    def __init__(self, site: str) -> None:
        self.site = site
        self._items: dict[str, Item] = {}

    def receive(self, sku: str, name: str, units: int) -> Item:
        item = self._items.setdefault(sku, Item(sku, name))
        item.units += units
        return item

    def ship(self, sku: str, units: int) -> Item:
        item = self._items[sku]
        if units > item.units:
            raise StockError(f"cannot ship {units} of {sku}: only {item.units} left")
        item.units -= units
        return item

    def needs_reorder(self, target: dict[str, int]) -> list[str]:
        """SKUs whose stock fell below REORDER_FRACTION of their target level."""
        low = []
        for sku, want in target.items():
            have = self._items.get(sku, Item(sku, sku)).units
            if have < want * REORDER_FRACTION:
                low.append(sku)
        return sorted(low)

    @property
    def total_units(self) -> int:
        return sum(i.units for i in self._items.values())
