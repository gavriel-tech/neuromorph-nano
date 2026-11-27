# Code review: the small shop program

We looked at the little shop program that stores uses to test its stock tool (folder mixed/).

- The inventory crate keeps StockItem lines and has take_from_shelf and total_units.
- The shop crate has checkout, which calls take_from_shelf and prices the units with unit_price_cents.
- There is no refund function in the code yet, even though the readme says the shop refunds.
  The test file is called checkout_refund.rs but its only test checks that checkout refuses when the
  shelf is short.

Decision: refunds stay a manual process for now; see the refund policy in mixed/docs.
