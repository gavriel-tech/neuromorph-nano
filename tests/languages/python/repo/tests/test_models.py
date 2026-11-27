import pytest

from inventory.models import Item, StockError, Warehouse


def test_shelves_round_up():
    assert Item("A1", "bolt", units=241).shelves_needed() == 2


def test_ship_refuses_more_than_stock():
    wh = Warehouse("north")
    wh.receive("A1", "bolt", 5)
    with pytest.raises(StockError):
        wh.ship("A1", 6)
