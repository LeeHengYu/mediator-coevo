# Task Instruction

Execute the following steps to produce `/root/Promo_Register_Audit.xlsx` and `/root/Promo_Register_Brief.docx`.

```python
import pandas as pd
from openpyxl import Workbook
from docx import Document

# ── 1. Read source ──────────────────────────────────────────────
df = pd.read_excel('/root/Promo_Price_Check_Source.xlsx')

# ── 2. Build RawData (exact copy) ──────────────────────────────
# Keep a copy for RawData sheet
raw = df.copy()

# ── 3. Build Formatted Data ────────────────────────────────────
fmt = df.copy()

# Ensure the first 8 columns are named exactly as required
expected_cols = ['Promo ID', 'SKU', 'Promo Price', 'Register Price',
                 'Promo Start Date', 'Sale Date', 'Promo End Date', 'Store ID']
# Rename if needed (in case source has slightly different names)
fmt.columns = list(fmt.columns)  # ensure list
# Map current columns to expected (positional, first 8)
col_map = {fmt.columns[i]: expected_cols[i] for i in range(min(8, len(fmt.columns)))}
fmt.rename(columns=col_map, inplace=True)

# Parse dates for comparison (keep originals for writing as strings)
for dc in ['Promo Start Date', 'Sale Date', 'Promo End Date']:
    fmt[dc] = pd.to_datetime(fmt[dc])

# Compute error columns
fmt['Price Error'] = (fmt['Register Price'] != fmt['Promo Price']).astype(int)
fmt['Window Error'] = ((fmt['Sale Date'] < fmt['Promo Start Date']) |
                       (fmt['Sale Date'] > fmt['Promo End Date'])).astype(int)
fmt['Total Errors'] = fmt['Price Error'] + fmt['Window Error']

def error_summary(row):
    pe = row['Price Error'] == 1
    we = row['Window Error'] == 1
    if pe and we:
        return 'Price Error, Window Error'
    elif pe:
        return 'Price Error'
    elif we:
        return 'Window Error'
    else:
        return 'None'

fmt['Error Summary'] = fmt.apply(error_summary, axis=1)

# CRITICAL: Convert date columns to YYYY-MM-DD strings so the verifier
# sees string values, not datetime objects.
for dc in ['Promo Start Date', 'Sale Date', 'Promo End Date']:
    fmt[dc] = fmt[dc].dt.strftime('%Y-%m-%d')

# Also convert date columns in raw to strings for consistency
for dc in ['Promo Start Date', 'Sale Date', 'Promo End Date']:
    col_name = raw.columns[expected_cols.index(dc)] if dc in expected_cols else dc
    # Find the matching column in raw by position
    pos = expected_cols.index(dc)
    actual_col = raw.columns[pos]
    raw[actual_col] = pd.to_datetime(raw[actual_col]).dt.strftime('%Y-%m-%d')

# ── 4. Build Summary ───────────────────────────────────────────
grp = fmt.groupby(['SKU', 'Store ID'], as_index=False).agg(
    **{'Price Errors': ('Price Error', 'sum'),
       'Window Errors': ('Window Error', 'sum'),
       'Total Errors': ('Total Errors', 'sum')}
)
grp = grp[grp['Total Errors'] > 0].copy()
grp.sort_values(['SKU', 'Store ID'], inplace=True)
grp.reset_index(drop=True, inplace=True)

# Ensure numeric columns are plain ints
for c in ['Price Errors', 'Window Errors', 'Total Errors']:
    grp[c] = grp[c].astype(int)

# Grand Total row
grand = pd.DataFrame([{
    'SKU': 'Grand Total',
    'Store ID': '-',
    'Price Errors': int(grp['Price Errors'].sum()),
    'Window Errors': int(grp['Window Errors'].sum()),
    'Total Errors': int(grp['Total Errors'].sum())
}])
summary = pd.concat([grp, grand], ignore_index=True)

# ── 5. Write Excel workbook ────────────────────────────────────
out_path = '/root/Promo_Register_Audit.xlsx'
with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
    raw.to_excel(writer, sheet_name='RawData', index=False)
    fmt.to_excel(writer, sheet_name='Formatted Data', index=False)
    summary.to_excel(writer, sheet_name='Summary', index=False)

print('Excel written to', out_path)

# ── 6. Identify high-priority SKUs ─────────────────────────────
# Use Formatted Data aggregation by SKU
sku_err = fmt.groupby('SKU')['Total Errors'].sum().sort_values(ascending=False)
top_skus = sku_err[sku_err > 0].head(2)
top_sku_list = list(top_skus.index)

total_price_errors = int(fmt['Price Error'].sum())
total_window_errors = int(fmt['Window Error'].sum())
total_total_errors = int(fmt['Total Errors'].sum())

# ── 7. Write Word brief ────────────────────────────────────────
doc = Document()
doc.add_heading('Promotional Register Audit – Executive Summary', level=1)

para = (
    f'This audit evaluated promotional register accuracy using two checks: '
    f'a Price Error flags any transaction where the register price differs from the '
    f'authorized promo price, and a Window Error flags any sale date that falls outside '
    f'the promotional start-to-end date window. '
    f'Across the dataset, {total_price_errors} Price Errors, {total_window_errors} Window Errors, '
    f'and {total_total_errors} Total Errors were identified. '
    f'SKUs {top_sku_list[0]} and {top_sku_list[1] if len(top_sku_list) > 1 else top_sku_list[0]} '
    f'exhibited the highest frequency of exceptions and should be treated as high priority for remediation. '
    f'We recommend conducting a root-cause review of register-price synchronization processes '
    f'and calendar-feed accuracy for promotional windows to prevent recurrence.'
)
doc.add_paragraph(para)

doc_path = '/root/Promo_Register_Brief.docx'
doc.save(doc_path)
print('Word brief written to', doc_path)
```

After running the script, verify:
1. `/root/Promo_Register_Audit.xlsx` exists and has exactly three sheets: `RawData`, `Formatted Data`, `Summary`.
2. In `Formatted Data`, date columns contain string values like `'2026-03-01'`, NOT datetime objects.
3. Columns 9-12 of `Formatted Data` are `Price Error`, `Window Error`, `Total Errors`, `Error Summary` with concrete integer/string values.
4. `Summary` sheet has headers `SKU, Store ID, Price Errors, Window Errors, Total Errors`, only rows with Total Errors > 0, sorted by SKU then Store ID ascending, with a Grand Total row at the bottom.
5. `/root/Promo_Register_Brief.docx` exists and contains the executive summary.

Run the verification commands:
```bash
python3 -c "
import openpyxl
wb = openpyxl.load_workbook('/root/Promo_Register_Audit.xlsx')
print('Sheets:', wb.sheetnames)
ws = wb['Formatted Data']
headers = [c.value for c in ws[1]]
print('Formatted Data headers:', headers)
# Check a date cell is a string
for row in ws.iter_rows(min_row=2, max_row=2, min_col=5, max_col=7):
    for cell in row:
        print(f'  Cell {cell.coordinate}: value={cell.value!r}, type={type(cell.value).__name__}')
ws2 = wb['Summary']
headers2 = [c.value for c in ws2[1]]
print('Summary headers:', headers2)
last_row = ws2.max_row
print('Last row (Grand Total):', [c.value for c in ws2[last_row]])
"
```

Also verify the Word file:
```bash
python3 -c "
from docx import Document
d = Document('/root/Promo_Register_Brief.docx')
for p in d.paragraphs:
    print(p.text[:200] if p.text else '')
"
```

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