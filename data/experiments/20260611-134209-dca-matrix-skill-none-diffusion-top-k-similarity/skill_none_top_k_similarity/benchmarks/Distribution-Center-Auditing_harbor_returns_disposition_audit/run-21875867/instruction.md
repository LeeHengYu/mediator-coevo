# Task Instruction

Execute the following steps in order to produce `/root/Returns_Disposition_Audit.xlsx` and `/root/Returns_Disposition_Brief.docx`.

## Step 0 — Inspect source files

```bash
cd /root
pip install openpyxl python-docx pandas xlrd 2>/dev/null
python3 -c "
import pandas as pd, json

rp = pd.read_excel('Return_Plan.xlsx')
print('=== Return_Plan columns:', list(rp.columns))
print('Return_Plan shape:', rp.shape)
print(rp.head(20).to_string(index=False))
print('...')
print(rp.tail(5).to_string(index=False))

el = pd.read_excel('Disposition_Event_Log.xlsx')
print('\n=== Event Log columns:', list(el.columns))
print('Event Log shape:', el.shape)
print('Unique Event Status values:', el['Event Status'].unique() if 'Event Status' in el.columns else 'COLUMN NOT FOUND')
print(el.head(20).to_string(index=False))

al = pd.read_excel('Disposition_Alias.xlsx')
print('\n=== Alias columns:', list(al.columns))
print('Alias shape:', al.shape)
print(al.to_string(index=False))
"
```

Read the output carefully. Note:
- The exact column names in each file (they may differ from the task description; use the actual names).
- The `Event Status` unique values (might be 'COMPLETED', 'Completed', etc.).
- The alias mapping direction: which column is the alias and which is the standard disposition.
- How many rows are in `Return_Plan` — every one must appear in `Formatted Data`.

## Step 1 — Write the processing script

After inspecting the data, write a Python script `/root/build_audit.py` that does the following. Adapt column names to what you actually saw in Step 0.

### 1a) Read all three files
```python
import pandas as pd
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
from docx import Document

rp = pd.read_excel('Return_Plan.xlsx')
el = pd.read_excel('Disposition_Event_Log.xlsx')
al = pd.read_excel('Disposition_Alias.xlsx')
```

### 1b) Build alias lookup
- Print the alias dataframe to confirm which column is the alias and which is the standard name.
- Build a dict: `alias_to_standard = {alias.strip().lower(): standard.strip() for alias, standard in zip(al[<alias_col>], al[<standard_col>])}`
- IMPORTANT: get the direction right. The alias column contains alternative names; the standard column contains the canonical name that should match `Planned Disposition`.

### 1c) Filter event log to COMPLETED only
- Use case-insensitive comparison: `el[el['Event Status'].str.strip().str.upper() == 'COMPLETED']`
- If there is a timestamp or sequence column, sort by it descending before dedup. If not, keep the last row per `(Return ID, Line ID)` as it appears in the file (i.e., use `drop_duplicates(subset=[return_id_col, line_id_col], keep='last')`).
- The result is `latest_events` — one row per (Return ID, Line ID) at most.

### 1d) Normalize the Final Disposition in latest_events
- For each row's `Final Disposition`, strip and lowercase it, look it up in `alias_to_standard`. If found, use the standard value; otherwise keep the original text.
- Store the normalized value in a new column `Normalized Disposition`.

### 1e) Merge with Return_Plan (LEFT join)
- `merged = rp.merge(latest_events[['Return ID','Line ID','Normalized Disposition']], on=['Return ID','Line ID'], how='left')`
- Confirm `len(merged) == len(rp)`. If not, investigate duplicates in latest_events and fix.

### 1f) Compute error columns
```python
merged['Missing Final Event'] = merged['Normalized Disposition'].isna().astype(int)
merged['Disposition Mismatch'] = 0
mask = merged['Normalized Disposition'].notna()
merged.loc[mask, 'Disposition Mismatch'] = (
    merged.loc[mask, 'Normalized Disposition'].str.strip().str.lower() !=
    merged.loc[mask, 'Planned Disposition'].str.strip().str.lower()
).astype(int)
merged['Total Errors'] = merged['Missing Final Event'] + merged['Disposition Mismatch']

def make_summary(row):
    parts = []
    if row['Missing Final Event'] == 1:
        parts.append('Missing Final Event')
    if row['Disposition Mismatch'] == 1:
        parts.append('Disposition Mismatch')
    return ', '.join(parts) if parts else 'None'

merged['Error Summary'] = merged.apply(make_summary, axis=1)
```

### 1g) Build Formatted Data with exactly 12 columns
Select the first 8 columns of `rp` (in order: Return ID, Line ID, Planned Disposition, Reason Code, Requested Qty, Warehouse, Carrier, Lane) plus the 4 new columns. Use the exact header names from the task.

### 1h) Build Summary sheet
```python
errors = formatted[formatted['Total Errors'] > 0]
summary = errors.groupby(['Warehouse','Carrier']).agg(
    **{'Missing Final Events': ('Missing Final Event','sum'),
       'Disposition Mismatches': ('Disposition Mismatch','sum'),
       'Total Errors': ('Total Errors','sum')}
).reset_index()
summary = summary.sort_values(['Warehouse','Carrier']).reset_index(drop=True)
grand = pd.DataFrame([{
    'Warehouse': 'Grand Total',
    'Carrier': '-',
    'Missing Final Events': summary['Missing Final Events'].sum(),
    'Disposition Mismatches': summary['Disposition Mismatches'].sum(),
    'Total Errors': summary['Total Errors'].sum()
}])
summary = pd.concat([summary, grand], ignore_index=True)
```

### 1i) Write Excel with openpyxl
Create a workbook with sheets named exactly `RawData`, `Formatted Data`, `Summary`. Write dataframes with headers. Ensure all values are concrete (no formulas). Save to `/root/Returns_Disposition_Audit.xlsx`.

### 1j) Write Word document
Create `/root/Returns_Disposition_Brief.docx` with an executive summary paragraph (3-6 sentences) that:
- Defines both checks in plain language.
- States the computed totals for Missing Final Events, Disposition Mismatches, and Total Errors (use the Grand Total row values).
- Gives at least one actionable recommendation.
- Mentions at least two Return IDs that have the most errors (find the top 2 Return IDs by Total Errors sum).

## Step 2 — Run the script
```bash
cd /root && python3 build_audit.py
```

## Step 3 — Validate outputs
```python
import pandas as pd

# Check sheets exist
xl = pd.ExcelFile('Returns_Disposition_Audit.xlsx')
print('Sheets:', xl.sheet_names)
assert 'RawData' in xl.sheet_names
assert 'Formatted Data' in xl.sheet_names
assert 'Summary' in xl.sheet_names

raw = pd.read_excel('Returns_Disposition_Audit.xlsx', sheet_name='RawData')
fmt = pd.read_excel('Returns_Disposition_Audit.xlsx', sheet_name='Formatted Data')
smry = pd.read_excel('Returns_Disposition_Audit.xlsx', sheet_name='Summary')

print('RawData shape:', raw.shape)
print('Formatted Data shape:', fmt.shape)
print('Formatted Data columns:', list(fmt.columns))
print('Summary shape:', smry.shape)
print('Summary:\n', smry.to_string(index=False))

# Verify row counts match
assert len(raw) == len(fmt), f'Row mismatch: {len(raw)} vs {len(fmt)}'

# Verify column count
assert len(fmt.columns) == 12, f'Expected 12 columns, got {len(fmt.columns)}'

# Print some error stats
print('Total Missing Final Events:', fmt['Missing Final Event'].sum())
print('Total Disposition Mismatches:', fmt['Disposition Mismatch'].sum())
print('Total Errors:', fmt['Total Errors'].sum())

# Verify Grand Total row
last_row = smry.iloc[-1]
assert last_row['Warehouse'] == 'Grand Total'
assert last_row['Total Errors'] == fmt['Total Errors'].sum()

print('\nAll validations passed.')
```

Also verify the Word doc exists and contains the expected totals:
```python
from docx import Document
doc = Document('Returns_Disposition_Brief.docx')
text = ' '.join([p.text for p in doc.paragraphs])
print('Word doc text:', text[:500])
assert str(int(fmt['Total Errors'].sum())) in text, 'Total Errors not found in Word doc'
print('Word doc validation passed.')
```

## Critical Reminders
- **Do NOT skip Step 0.** You must inspect the actual column names and data before writing the script. The previous attempt failed because of wrong column references and alias mapping direction.
- **Case-insensitive matching everywhere**: Event Status filtering, disposition comparison, alias lookup.
- **LEFT join** from Return_Plan to events — never lose a Return_Plan row.
- **Alias direction**: carefully determine which column is the alias (alternative name) and which is the standard (canonical) name by reading the column headers and data in the alias file.
- **Concrete values only** in the 4 new columns — no Excel formulas.
- **Sheet names must be exact**: `RawData`, `Formatted Data`, `Summary`.
- If any step fails, re-read the data, diagnose, fix, and re-run before moving on.

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
Task metadata: author_email=catpaw@meituan.com, author_name=CatPaw Benchmark Builder, category=spreadsheet-audit, difficulty=hard, tags=[excel, openpyxl, docx, audit, returns].
Verifier config: timeout_sec=900.0.