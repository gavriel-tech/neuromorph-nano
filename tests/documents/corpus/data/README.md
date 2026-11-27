# data

Written by `gen/make_data.py` with fixed seeds; do not edit by hand, rerun the generator.

- `parts.csv`: parts inventory, header on line 1 and 300 data rows (data row k is on line k+1).
  RFC 4180: CRLF line ends, fields with commas are quoted, one name holds a doubled quote.
- `suppliers.tsv`: 40 suppliers, tab separated, header row; email addresses use example.com.
- `orders.jsonl`: 100 orders, one JSON object per line, sorted by date.
- `pump-service.log`: 400 lines of a bench service log for one day, ISO 8601 UTC timestamps,
  levels INFO, WARN and ERROR, one indented stack trace and one burst of repeated errors.
