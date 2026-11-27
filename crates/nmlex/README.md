# nmlex

Token and term helpers shared by the substrate scanner, the drone tip readers and the code tools.
`src/lex.rs` (comment and literal aware tokens) is in-house engine code of Gavriel Technologies
(see `THIRD_PARTY_NOTICES.md`); `src/lib.rs` wraps it and adds `words`, `key_terms`, `names` and
the stop list. Bottom layer beside `nmjson` and `nmhash`.

Section checks: `neuromorph-nano check nmlex`.
