//! The checkout of the shop.
use inventory::{take_from_shelf, StockItem};

/// Price of one unit in cents, by product code.
pub fn unit_price_cents(code: &str) -> u64 {
    match code {
        "APPLE" => 40,
        "PEAR" => 55,
        _ => 100,
    }
}

/// Check out `n` units: takes them from the shelf and returns the price in cents, or None.
pub fn checkout(item: &mut StockItem, n: u32) -> Option<u64> {
    if !take_from_shelf(item, n) {
        return None;
    }
    Some(unit_price_cents(&item.code) * u64::from(n))
}
