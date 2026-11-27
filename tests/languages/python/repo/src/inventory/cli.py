"""Command line: `inventory ship <site> <sku> <units>` and friends."""
import argparse
import sys
from pathlib import Path

from inventory.storage import CsvStore
from . import pricing

DATA_DIR = Path("data")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="inventory")
    sub = p.add_subparsers(dest="cmd", required=True)
    ship = sub.add_parser("ship")
    ship.add_argument("site")
    ship.add_argument("sku")
    ship.add_argument("units", type=int)
    quote = sub.add_parser("quote")
    quote.add_argument("price")
    quote.add_argument("units", type=int)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "ship":
        store = CsvStore(DATA_DIR)
        wh = store.load(args.site)
        item = wh.ship(args.sku, args.units)
        store.save(wh)
        print(item.label())
    elif args.cmd == "quote":
        from decimal import Decimal
        print(pricing.unit_price(Decimal(args.price), args.units))
    return 0


if __name__ == "__main__":
    sys.exit(main())
