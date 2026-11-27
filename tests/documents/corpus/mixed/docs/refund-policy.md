# Refund policy

This policy says how the shop refunds a customer. It applies to every product the shop sells.

## Who can get a refund

A customer can return unopened goods within 14 days of purchase with the receipt. Opened goods are
refunded only when they are faulty.

## How much is refunded

The refund is the price the customer paid: the number of units returned times the unit price in
cents given by `unit_price_cents` in the `shop` crate. Today that function prices an apple at 40
cents, a pear at 55 cents and every other product code at 100 cents. A refund never uses a newer or
older price than the one on the receipt.

## How a refund is processed

The code has no refund function yet. The test file `checkout_refund.rs` is named for refunds but
only tests checkout. Until a refund function is written, refunds are handled by hand:

1. Check the receipt and the returned goods.
2. Work out the amount as described above.
3. Pay the customer back by the same method they used to pay.
4. Put the returned units back on the shelf by raising the `count` of the matching `StockItem`.
5. Write the refund in the refunds book with the date, product code, units and amount.

## Faulty goods

Faulty goods are refunded in full but are not put back on the shelf. Leave the `count` unchanged and
put the goods in the returns crate for the supplier.
