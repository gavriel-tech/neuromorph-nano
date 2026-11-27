# Setup guide: the shop workspace

This guide explains how to build and test the small shop program in this folder. The program is a
Rust workspace with two crates, `inventory` and `shop`, listed in the workspace `Cargo.toml` at the
top of the folder.

## What you need

- A Rust toolchain with Cargo, edition 2021 or later.
- No outside crates. The `shop` crate depends only on `inventory`, through a path dependency
  (`inventory = { path = "../inventory" }` in `crates/shop/Cargo.toml`).

## Building

1. Open a terminal at the top of this folder, where the workspace `Cargo.toml` is.
2. Run `cargo build`. Cargo builds `inventory` first, because `shop` depends on it.
3. Check that both crates compile without warnings.

## Running the tests

Run `cargo test` from the top of the folder. There is one integration test today, in
`crates/shop/tests/checkout_refund.rs`. Its single test, `checkout_refuses_when_shelf_is_short`,
puts one pear on the shelf, asks `checkout` for two, and checks that the answer is `None` and that
the shelf still holds one pear.

## Where things live

- Stock lines: the `StockItem` struct in `crates/inventory/src/lib.rs`, with a product `code` and a
  `count` of units on the shelf.
- Taking stock off the shelf: `take_from_shelf` in the same file. It returns `false` and changes
  nothing when the shelf holds fewer units than asked for.
- Counting all stock: `total_units`, also in the inventory crate.
- Prices and checkout: `unit_price_cents` and `checkout` in `crates/shop/src/lib.rs`.
