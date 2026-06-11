# Task Instruction

Execute the following steps in order to produce `/root/Promo_Register_Audit.xlsx` and `/root/Promo_Register_Brief.docx`.

## Step 0 — Inspect the source workbook
```python
import pandas as pd
df = pd.read_excel('/root/Promo_Price_Check_Source.xlsx')
print(df.shape)
print(df.columns.tolist())
print(df.dtypes)
print(df.head(5))
```
Confirm column names, date formats, and numeric types before proceeding.

## Step 1 — Load and clean data
```python
import pandas as pd
import numpy as np
from copy import deepcopy

df_raw = pd.read_excel('/root/Promo_Price_Check_Source.xlsx')

# Ensure the first 8 columns match the required names; rename if needed:
required_cols = ['Promo ID','SKU','Promo Price','Register Price',
                 'Promo Start Date','Sale Date','Promo End Date','Store ID']
# If source column names differ only by whitespace/case, map them.
# Otherwise keep as-is (inspect Step 0 output first).

# Parse date columns robustly
for dc in ['Promo Start Date','Sale Date','Promo End Date']:
    df_raw[dc] = pd.to_datetime(df_raw[dc])

# Round prices to 2 decimals to avoid float comparison artifacts
df_raw['Promo Price'] = df_raw['Promo Price'].round(2)
df_raw['Register Price'] = df_raw['Register Price'].round(2)
```

## Step 2 — Build Formatted Data
```python
df_fmt = df_raw.copy()

# Price Error: 1 if Register Price != Promo Price
df_fmt['Price Error'] = (df_fmt['Register Price'] != df_fmt['Promo Price']).astype(int)

# Window Error: 1 if Sale Date < Promo Start Date OR Sale Date > Promo End Date
df_fmt['Window Error'] = ((df_fmt['Sale Date'] < df_fmt['Promo Start Date']) |
                          (df_fmt['Sale Date'] > df_fmt['Promo End Date'])).astype(int)

# Total Errors
df_fmt['Total Errors'] = df_fmt['Price Error'] + df_fmt['Window Error']

# Error Summary
def error_summary(row):
    parts = []
    if row['Price Error'] == 1:
        parts.append('Price Error')
    if row['Window Error'] == 1:
        parts.append('Window Error')
    return ', '.join(parts) if parts else 'None'

df_fmt['Error Summary'] = df_fmt.apply(error_summary, axis=1)
```

## Step 3 — Build Summary sheet
```python
grp = df_fmt.groupby(['SKU','Store ID'], as_index=False).agg(
    **{'Price Errors': ('Price Error','sum'),
       'Window Errors': ('Window Error','sum'),
       'Total Errors': ('Total Errors','sum')})

# Keep only groups with Total Errors > 0
grp = grp[grp['Total Errors'] > 0].copy()

# Sort by SKU asc, Store ID asc
grp = grp.sort_values(['SKU','Store ID']).reset_index(drop=True)

# Grand Total row
grand = pd.DataFrame([{
    'SKU': 'Grand Total',
    'Store ID': '-',
    'Price Errors': grp['Price Errors'].sum(),
    'Window Errors': grp['Window Errors'].sum(),
    'Total Errors': grp['Total Errors'].sum()
}])
df_summary = pd.concat([grp, grand], ignore_index=True)
```

## Step 4 — Write the Excel workbook
```python
from openpyxl import Workbook

with pd.ExcelWriter('/root/Promo_Register_Audit.xlsx', engine='openpyxl') as writer:
    df_raw.to_excel(writer, sheet_name='RawData', index=False)
    df_fmt.to_excel(writer, sheet_name='Formatted Data', index=False)
    df_summary.to_excel(writer, sheet_name='Summary', index=False)
```
After writing, re-read and verify:
- Sheet names are exactly `RawData`, `Formatted Data`, `Summary`.
- `Formatted Data` has 12 columns with exact header names.
- `Summary` last row has SKU == 'Grand Total'.
- No Excel formulas — all values are concrete.

## Step 5 — Build the Word brief
```python
from docx import Document

total_price_errors = int(df_fmt['Price Error'].sum())
total_window_errors = int(df_fmt['Window Error'].sum())
total_errors = int(df_fmt['Total Errors'].sum())

# Identify top SKUs by total errors
sku_errors = df_fmt.groupby('SKU')['Total Errors'].sum().sort_values(ascending=False)
top_skus = sku_errors.head(2).index.tolist()

doc = Document()
doc.add_heading('Promotional Register Audit – Executive Brief', level=1)

paragraph = (
    f'This audit evaluated promotional register accuracy using two checks: '
    f'a Price Error flags any transaction where the register price differs from the scheduled promo price, '
    f'and a Window Error flags any sale date that falls outside the promotional start and end dates. '
    f'Across the dataset, {total_price_errors} Price Errors, {total_window_errors} Window Errors, '
    f'and {total_errors} Total Errors were identified. '
    f'SKUs {top_skus[0]} and {top_skus[1]} exhibited the highest frequency of exceptions and should be '
    f'prioritized for root-cause investigation. '
    f'We recommend implementing automated register-price synchronization at the start of each promotional window '
    f'and adding date-range validation at the point of sale to prevent future discrepancies.'
)
doc.add_paragraph(paragraph)
doc.save('/root/Promo_Register_Brief.docx')
```

## Step 6 — Final validation
Re-open both output files and print:
- Sheet names of the Excel workbook.
- Column headers and row counts of each sheet.
- First and last rows of Summary.
- Paragraph text of the Word doc.
Confirm everything matches the specification before finishing.

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