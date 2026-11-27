use inventory::StockItem;
use shop::checkout;

/// Checking out more than the shelf holds is refused and leaves the shelf untouched.
#[test]
fn checkout_refuses_when_shelf_is_short() {
    let mut item = StockItem { code: "PEAR".into(), count: 1 };
    assert_eq!(checkout(&mut item, 2), None);
    assert_eq!(item.count, 1);
}
