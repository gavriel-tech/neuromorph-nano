from ..models import Warehouse


class MemoryStore:
    """Warehouses kept in a dictionary; for tests and dry runs."""

    def __init__(self) -> None:
        self._sites: dict[str, Warehouse] = {}

    def save(self, wh: Warehouse) -> str:
        self._sites[wh.site] = wh
        return wh.site

    def load(self, site: str) -> Warehouse:
        return self._sites.setdefault(site, Warehouse(site))
