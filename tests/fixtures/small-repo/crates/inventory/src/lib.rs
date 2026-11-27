//! Stock keeping for the shop.

/// One line of stock: a product code and how many are on the shelf.
pub struct StockItem {
    pub code: String,
    pub count: u32,
}

/// Remove `n` units of a product; returns false when there are not enough on the shelf.
pub fn take_from_shelf(item: &mut StockItem, n: u32) -> bool {
    if item.count < n {
        return false;
    }
    item.count -= n;
    true
}

/// Total units across every line of stock.
pub fn total_units(items: &[StockItem]) -> u64 {
    items.iter().map(|i| u64::from(i.count)).sum()
}
