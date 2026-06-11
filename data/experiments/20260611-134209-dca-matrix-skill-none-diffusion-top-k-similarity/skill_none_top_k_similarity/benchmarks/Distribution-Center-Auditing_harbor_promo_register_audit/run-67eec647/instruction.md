# Task Instruction

## Task: Build Promo Register Audit Files

You must create two deliverable files from the source workbook `/root/Promo_Price_Check_Source.xlsx`.

### Step 1: Inspect the source
```bash
cd /root
python3 -c "
import openpyxl
wb = openpyxl.load_workbook('Promo_Price_Check_Source.xlsx')
for s in wb.sheetnames:
    ws = wb[s]
    print(f'Sheet: {s}, rows={ws.max_row}, cols={ws.max_column}')
    for r in ws.iter_rows(min_row=1, max_row=min(5, ws.max_row), values_only=True):
        print(r)
"
```
Examine the column headers and a few data rows. Identify which columns map to: Promo ID, SKU, Promo Price, Register Price, Promo Start Date, Sale Date, Promo End Date, Store ID. Note the exact column order and data types (especially whether dates are datetime objects or strings).

### Step 2: Build the audit workbook and Word doc

Write and run a single Python script (`/root/build_audit.py`) that does ALL of the following:

```python
import openpyxl
from openpyxl import Workbook
from datetime import datetime, date
from docx import Document
from copy import copy

# --- Load source ---
src = openpyxl.load_workbook('Promo_Price_Check_Source.xlsx')
# Identify the sheet with data (likely the first or only sheet)
src_ws = src[src.sheetnames[0]]

# Read all rows (header + data)
all_rows = list(src_ws.iter_rows(min_row=1, values_only=True))
header = list(all_rows[0])
data = [list(r) for r in all_rows[1:]]

# Map columns by header name (case-insensitive, stripped)
def col_index(name, header):
    name_l = name.strip().lower()
    for i, h in enumerate(header):
        if h and h.strip().lower() == name_l:
            return i
    raise ValueError(f"Column '{name}' not found in {header}")

idx_promo_price = col_index('Promo Price', header)
idx_reg_price = col_index('Register Price', header)
idx_promo_start = col_index('Promo Start Date', header)
idx_sale_date = col_index('Sale Date', header)
idx_promo_end = col_index('Promo End Date', header)
idx_sku = col_index('SKU', header)
idx_store = col_index('Store ID', header)

# Helper: normalize to date for comparison
def to_date(v):
    if isinstance(v, datetime):
        return v.date()
    if isinstance(v, date):
        return v
    if isinstance(v, str):
        for fmt in ('%Y-%m-%d', '%m/%d/%Y', '%m-%d-%Y', '%d/%m/%Y'):
            try:
                return datetime.strptime(v.strip(), fmt).date()
            except ValueError:
                continue
    raise ValueError(f"Cannot parse date: {v}")

# --- Create output workbook ---
wb = Workbook()

# ---- RawData sheet ----
ws_raw = wb.active
ws_raw.title = 'RawData'
ws_raw.append(header)
for row in data:
    ws_raw.append(row)

# ---- Formatted Data sheet ----
ws_fmt = wb.create_sheet('Formatted Data')
fmt_header = ['Promo ID', 'SKU', 'Promo Price', 'Register Price',
              'Promo Start Date', 'Sale Date', 'Promo End Date', 'Store ID',
              'Price Error', 'Window Error', 'Total Errors', 'Error Summary']
ws_fmt.append(fmt_header)

# Build the 8-column index mapping from source header to the required order
required_8 = ['Promo ID', 'SKU', 'Promo Price', 'Register Price',
              'Promo Start Date', 'Sale Date', 'Promo End Date', 'Store ID']
idx_map = [col_index(name, header) for name in required_8]

formatted_rows = []  # store dicts for summary
for row in data:
    base = [row[i] for i in idx_map]
    promo_price = row[idx_promo_price]
    reg_price = row[idx_reg_price]
    price_error = 1 if reg_price != promo_price else 0

    sale_d = to_date(row[idx_sale_date])
    start_d = to_date(row[idx_promo_start])
    end_d = to_date(row[idx_promo_end])
    window_error = 1 if (sale_d < start_d or sale_d > end_d) else 0

    total_errors = price_error + window_error

    parts = []
    if price_error:
        parts.append('Price Error')
    if window_error:
        parts.append('Window Error')
    error_summary = ', '.join(parts) if parts else 'None'

    out_row = base + [price_error, window_error, total_errors, error_summary]
    ws_fmt.append(out_row)
    formatted_rows.append({
        'sku': row[idx_sku],
        'store': row[idx_store],
        'price_error': price_error,
        'window_error': window_error,
        'total_errors': total_errors
    })

# ---- Summary sheet ----
ws_sum = wb.create_sheet('Summary')
sum_header = ['SKU', 'Store ID', 'Price Errors', 'Window Errors', 'Total Errors']
ws_sum.append(sum_header)

from collections import defaultdict
agg = defaultdict(lambda: [0, 0, 0])
for r in formatted_rows:
    key = (r['sku'], r['store'])
    agg[key][0] += r['price_error']
    agg[key][1] += r['window_error']
    agg[key][2] += r['total_errors']

# Filter and sort
filtered = {k: v for k, v in agg.items() if v[2] > 0}
sorted_keys = sorted(filtered.keys(), key=lambda x: (str(x[0]), str(x[1])))

grand_pe = 0
grand_we = 0
grand_te = 0
for k in sorted_keys:
    v = filtered[k]
    ws_sum.append([k[0], k[1], v[0], v[1], v[2]])
    grand_pe += v[0]
    grand_we += v[1]
    grand_te += v[2]

ws_sum.append(['Grand Total', '-', grand_pe, grand_we, grand_te])

wb.save('Promo_Register_Audit.xlsx')
print(f'Audit workbook saved. Grand totals: PE={grand_pe}, WE={grand_we}, TE={grand_te}')
print(f'Formatted rows: {len(formatted_rows)}, Summary groups: {len(sorted_keys)}')

# --- Identify top SKUs by total errors ---
sku_errors = defaultdict(int)
for r in formatted_rows:
    sku_errors[r['sku']] += r['total_errors']
top_skus = sorted(sku_errors.items(), key=lambda x: -x[1])[:5]
print('Top SKUs by errors:', top_skus)

# --- Word document ---
doc = Document()
doc.add_heading('Promo Register Audit – Executive Summary', level=1)

top2 = [str(s[0]) for s in top_skus[:2]]

paragraph = (
    f'This audit reviewed {len(formatted_rows)} promotional register transactions '
    f'across the dataset. '
    f'A Price Error is flagged when the register price charged to the customer '
    f'does not match the authorized promotional price. '
    f'A Window Error is flagged when a sale is recorded outside the valid '
    f'promotional window (before the start date or after the end date). '
    f'Across all transactions, the audit identified {grand_pe} Price Errors, '
    f'{grand_we} Window Errors, and {grand_te} Total Errors. '
    f'SKUs {top2[0]} and {top2[1]} were among the highest-priority items with '
    f'the most frequent exceptions and should be investigated first. '
    f'We recommend conducting a root-cause analysis on register-price synchronization '
    f'and promotional calendar alignment for these SKUs, and implementing automated '
    f'price-verification checks at the point of sale to prevent future discrepancies.'
)
doc.add_paragraph(paragraph)
doc.save('Promo_Register_Brief.docx')
print('Word brief saved.')
```

Adjust the script if the source column names differ from what you find in Step 1. The key contract points:
- Column names in source may vary slightly; match them case-insensitively.
- Date comparison must use date objects (not datetimes with time components that could cause spurious mismatches).
- Price comparison: use `!=` directly on the values as read from openpyxl (they should be numeric).

### Step 3: Run and validate
```bash
python3 /root/build_audit.py
```

Then verify the output:
```python
import openpyxl

# Verify audit workbook
wb = openpyxl.load_workbook('Promo_Register_Audit.xlsx')
print('Sheets:', wb.sheetnames)  # Must be exactly ['RawData', 'Formatted Data', 'Summary']

# Check RawData row count matches source
ws = wb['RawData']
print(f'RawData rows (including header): {ws.max_row}')

# Check Formatted Data headers and sample
ws2 = wb['Formatted Data']
headers = [c.value for c in ws2[1]]
print('Formatted Data headers:', headers)
# Verify first few rows have computed values
for row in ws2.iter_rows(min_row=2, max_row=min(5, ws2.max_row), values_only=True):
    print(row)

# Check Summary
ws3 = wb['Summary']
print('Summary headers:', [c.value for c in ws3[1]])
last_row = [c.value for c in ws3[ws3.max_row]]
print('Last row (Grand Total):', last_row)
assert last_row[0] == 'Grand Total'
assert last_row[1] == '-'

# Verify Word doc exists and has content
from docx import Document
d = Document('Promo_Register_Brief.docx')
for p in d.paragraphs:
    print(p.text[:200] if p.text else '')
print('Validation complete.')
```

### Step 4: Fix any issues
- If column names don't match, update the script with the actual names found in Step 1.
- If dates are strings, ensure the parsing handles the actual format.
- If the Word doc library is missing, install it: `pip install python-docx`
- If openpyxl is missing: `pip install openpyxl`
- Re-run until both files validate correctly.

### Critical Contracts to Satisfy
1. **File paths**: `/root/Promo_Register_Audit.xlsx` and `/root/Promo_Register_Brief.docx` exactly.
2. **Sheet names**: `RawData`, `Formatted Data`, `Summary` — exact case and spacing.
3. **Formatted Data columns 9-12**: Must contain concrete numeric/string values, NOT Excel formulas.
4. **Error Summary values**: Must be exactly one of: `None`, `Price Error`, `Window Error`, `Price Error, Window Error`.
5. **Summary sheet**: Only groups with Total Errors > 0; sorted by SKU asc then Store ID asc; last row is Grand Total with `-` for Store ID.
6. **Word doc**: 3-6 sentences, mentions both error definitions, computed totals, two high-priority SKUs, and a recommendation.

# Executor Policy

---
name: executor
description: Portable executor policy for workflow, verification, resource use, and failure handling across task runtimes.
---

## Executor Policy

Use this skill as execution policy, not as domain-specific task knowledge. When
task-local curated skills or resources are available, prefer them for domain
details and use this policy for workflow control.

## Task Execution

1. Read the task instruction, task resources, and verifier contract before editing.
2. Identify the scoring mechanism and the smallest command that can reproduce the
   failure or verify the expected behavior.
3. Inspect existing files and task-local resources before making changes.
4. Make the smallest source change that satisfies the task and verifier contract.
5. Keep a compact record of the concrete evidence behind the change: observed
   failure, files inspected, edit made, and verifier result.
6. Run targeted verification before broad verification when practical.

## File Editing

1. Read the actual current file contents immediately before making any edit.
   Never rely on memory, prior snapshots, or assumed content.
2. Prefer direct in-place edits over patch or diff application when the exact
   current context is uncertain.
3. If using a patch or diff, confirm that every context line exists verbatim in
   the file before applying it.
4. If a patch hunk fails to apply, re-read the affected file region and perform
   the edit directly instead of retrying the same patch.
5. After any edit, re-read the affected region to confirm the change landed.

## Build and Test Fixes

When a task requires fixing a broken build, failing test, or generated artifact:

1. Run the relevant build, test, or verifier command first to capture the
   baseline failure.
2. Identify the specific error message, file, line, or expected output before
   editing.
3. Apply the smallest fix, then re-run the same targeted command.
4. Treat newly introduced failures as separate sub-tasks and resolve them in
   order.
5. Do not mark the task complete until the verifier-relevant command succeeds or
   the remaining failure is clearly outside the task boundary.

## Artifact-Contract Handling

Do not treat artifacts as ordinary text files. Treat them as contract-bearing
interfaces between input data, generated output, verifier checks, and downstream
consumers.

When a task requires reading, modifying, or generating an artifact such as JSON,
DOT, reports, configs, generated source, schemas, datasets, or parsed outputs:

1. Identify the artifact contract first: format, schema, required fields,
   identifiers, references, ordering, examples, verifier assertions, and
   consuming code.
2. Inspect representative source artifacts directly before deciding how to
   transform or preserve them.
3. Determine whether the task calls for preservation, transformation, repair,
   generation, or validation.
4. Preserve required literals, identifiers, references, ordering, and
   representative content unless the contract explicitly requires a change.
5. Do not invent, drop, rename, normalize, collapse, expand, or repair artifact
   elements unless the verifier or consumer contract requires that behavior.
6. Prefer structured parsers, serializers, validators, or existing consumer code
   over ad hoc string manipulation when they are available.
7. After producing the artifact, run targeted checks for parseability, required
   keys or IDs, reference consistency, expected counts, preserved content, and
   format-specific validity.
8. If targeted checks regress or become unusable after a change, stop expanding
   the solution. Re-inspect the source contract and narrow the edit before trying
   a broader repair.

A plausible-looking artifact is not sufficient evidence. The artifact is only
correct when it satisfies the task contract under the verifier or consuming
code.

## Constraints

- Do not bypass, remove, or weaken tests, verifier scripts, fixtures, or expected
  output checks.
- Do not treat this policy as overriding task-specific instructions or verifier
  requirements.
- On tool or environment errors, retry once when the retry is safe, then report
  the failure with the command and error output.
- On ambiguous instructions, make a conservative assumption and continue.

# Task Resources

Inspect the task files, environment, tests, and expected outputs directly.

# Verifier Contract

Success is judged by the SkillFlow verifier for this task.
Do not bypass, remove, or weaken verifier scripts, tests, fixtures, or expected-output checks.
Run the provided tests or verifier command when practical before finalizing.
Task metadata: author_email=catpaw@meituan.com, author_name=CatPaw Benchmark Builder, category=spreadsheet-audit, difficulty=medium, tags=[excel, openpyxl, docx, audit, pricing].
Verifier config: timeout_sec=900.0.