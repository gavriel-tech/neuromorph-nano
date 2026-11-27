# Design note: splitting stock and checkout

## Why two crates

Stock keeping and selling change for different reasons. The `inventory` crate knows only about
shelves: what a `StockItem` is and how units come off the shelf. The `shop` crate knows about
prices and sales. Keeping them apart means a price change never touches the stock code.

## The checkout rule

`checkout` must never charge for units it could not take. It calls `take_from_shelf` first and
returns `None` at once if that fails, so the price is worked out only after the units have left the
shelf. Because `take_from_shelf` checks the count before subtracting, a refused checkout leaves the
shelf exactly as it was. The test `checkout_refuses_when_shelf_is_short` guards this rule.

## Money in cents

Prices are whole numbers of cents in a `u64`, never floating point. `checkout` multiplies the unit
price by the number of units, converted with `u64::from`, so a large order cannot overflow a `u32`.

## Unknown products

`unit_price_cents` gives 100 cents for any code it does not know. This keeps checkout working when a
new product reaches the shelf before its price is added, at the risk of a wrong price. A later
version should refuse unknown codes instead.

## Not done yet

- A refund function in `shop` (see the refund policy for the manual process).
- A way to add units back to a shelf; today code must change `count` directly.
