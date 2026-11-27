"""Prices: unit price with volume discounts, and currency rounding."""
import math
from decimal import ROUND_HALF_EVEN, Decimal

# (minimum units, discount) steps, largest first.
VOLUME_STEPS = [(1000, Decimal("0.15")), (250, Decimal("0.08")), (50, Decimal("0.03"))]


def discount_for(units: int) -> Decimal:
    for minimum, off in VOLUME_STEPS:
        if units >= minimum:
            return off
    return Decimal("0")


def unit_price(list_price: Decimal, units: int) -> Decimal:
    """List price less the volume discount, rounded to cents (banker's rounding)."""
    price = list_price * (1 - discount_for(units))
    return price.quantize(Decimal("0.01"), rounding=ROUND_HALF_EVEN)


async def fetch_rates(client, base: str = "EUR") -> dict:
    """Exchange rates from the rates service; the caller owns the client."""

    def keep(code: str) -> bool:
        return len(code) == 3 and code.isupper()

    data = await client.get(f"/rates?base={base}")
    return {k: v for k, v in data["rates"].items() if keep(k)}


def boxes(units: int, per_box: int) -> int:
    return math.ceil(units / per_box)
