# Task Instruction

Execute the following steps in order to produce the two deliverables.

## 1. Inspect the source files

```bash
pip install openpyxl python-docx pandas
```

Then read and print the contents of all three input files:

```python
import pandas as pd

rp = pd.read_excel('/root/Return_Plan.xlsx')
print('=== Return_Plan ===')
print(rp.columns.tolist())
print(rp.head(20))
print(rp.shape)
print(rp.dtypes)

el = pd.read_excel('/root/Disposition_Event_Log.xlsx')
print('\n=== Disposition_Event_Log ===')
print(el.columns.tolist())
print(el.head(20))
print(el.shape)
print(el.dtypes)

da = pd.read_excel('/root/Disposition_Alias.xlsx')
print('\n=== Disposition_Alias ===')
print(da.columns.tolist())
print(da.head(20))
print(da.shape)
print(da.dtypes)
```

Carefully note the exact column names in each file (they may differ slightly from the task description, e.g., spaces, casing). Adapt the code below accordingly.

## 2. Build the audit workbook and Word brief

After inspecting, run a single Python script that does everything. Adapt column names to match what you observed. Here is the template logic:

```python
import pandas as pd
from openpyxl import Workbook
from docx import Document

# ---- Load data ----
rp = pd.read_excel('/root/Return_Plan.xlsx')
el = pd.read_excel('/root/Disposition_Event_Log.xlsx')
da = pd.read_excel('/root/Disposition_Alias.xlsx')

# ---- Identify exact column names from inspection above ----
# Adapt these variable assignments to the real column names you observed:
# e.g., RETURN_ID_COL = 'Return ID'  or 'ReturnID' etc.
# IMPORTANT: Use the EXACT column names from the files.

# ---- RawData: exact copy of Return_Plan ----
raw = rp.copy()

# ---- Build alias lookup (case-insensitive) ----
# da should have an alias column and a standard disposition column.
# Build a dict: lowercase alias -> standard disposition (preserve original casing of standard).
alias_map = {}
for _, row in da.iterrows():
    # Adapt column names from inspection
    alias_val = str(row['Alias']).strip()          # adjust col name
    standard_val = str(row['Standard Disposition']).strip()  # adjust col name
    alias_map[alias_val.lower()] = standard_val

# ---- Process event log: keep only COMPLETED, then latest per (Return ID, Line ID) ----
# Adapt column names from inspection
el_completed = el[el['Event Status'].str.strip().str.upper() == 'COMPLETED'].copy()

# Determine which column indicates ordering (timestamp, event ID, row order).
# If there is a timestamp or sequence column, sort by it. Otherwise use row index as proxy.
# After inspection, pick the right approach. Example with a timestamp column:
# el_completed = el_completed.sort_values('Event Timestamp')
# If no timestamp, use original row order (index):
el_completed = el_completed.reset_index().rename(columns={'index': '_orig_idx'})
el_completed = el_completed.sort_values('_orig_idx')
latest = el_completed.groupby(['Return ID', 'Line ID']).last().reset_index()

# ---- Build Formatted Data ----
fmt = rp.copy()
# Ensure first 8 columns are exactly as specified. Rename if needed.
# The first 8 columns should be: Return ID, Line ID, Planned Disposition, Reason Code,
# Requested Qty, Warehouse, Carrier, Lane
# Reorder/rename to match exactly.
fmt = fmt[['Return ID', 'Line ID', 'Planned Disposition', 'Reason Code',
           'Requested Qty', 'Warehouse', 'Carrier', 'Lane']].copy()

# Merge with latest completed events
fmt = fmt.merge(latest[['Return ID', 'Line ID', 'Final Disposition']],
                on=['Return ID', 'Line ID'], how='left')

# Normalize Final Disposition using alias map
def normalize(val):
    if pd.isna(val):
        return None
    val_str = str(val).strip()
    mapped = alias_map.get(val_str.lower(), val_str)
    return mapped

fmt['_norm_final'] = fmt['Final Disposition'].apply(normalize)

# Compute error columns
fmt['Missing Final Event'] = fmt['Final Disposition'].isna().astype(int)

def check_mismatch(row):
    if pd.isna(row['Final Disposition']):
        return 0
    planned = str(row['Planned Disposition']).strip().lower()
    norm = str(row['_norm_final']).strip().lower()
    return 0 if planned == norm else 1

fmt['Disposition Mismatch'] = fmt.apply(check_mismatch, axis=1)
fmt['Total Errors'] = fmt['Missing Final Event'] + fmt['Disposition Mismatch']

def error_summary(row):
    parts = []
    if row['Missing Final Event'] == 1:
        parts.append('Missing Final Event')
    if row['Disposition Mismatch'] == 1:
        parts.append('Disposition Mismatch')
    return ', '.join(parts) if parts else 'None'

fmt['Error Summary'] = fmt.apply(error_summary, axis=1)

# Drop helper columns
fmt = fmt.drop(columns=['Final Disposition', '_norm_final'], errors='ignore')

# ---- Summary sheet ----
fmt_errors = fmt[fmt['Total Errors'] > 0].copy()
summary = fmt_errors.groupby(['Warehouse', 'Carrier']).agg(
    **{'Missing Final Events': ('Missing Final Event', 'sum'),
       'Disposition Mismatches': ('Disposition Mismatch', 'sum'),
       'Total Errors': ('Total Errors', 'sum')}
).reset_index()
summary = summary.sort_values(['Warehouse', 'Carrier']).reset_index(drop=True)

# Grand Total row
grand = pd.DataFrame([{
    'Warehouse': 'Grand Total',
    'Carrier': '-',
    'Missing Final Events': fmt['Missing Final Event'].sum(),
    'Disposition Mismatches': fmt['Disposition Mismatch'].sum(),
    'Total Errors': fmt['Total Errors'].sum()
}])
summary = pd.concat([summary, grand], ignore_index=True)

# Convert numeric cols to int
for c in ['Missing Final Events', 'Disposition Mismatches', 'Total Errors']:
    summary[c] = summary[c].astype(int)

# ---- Write Excel ----
with pd.ExcelWriter('/root/Returns_Disposition_Audit.xlsx', engine='openpyxl') as writer:
    raw.to_excel(writer, sheet_name='RawData', index=False)
    fmt.to_excel(writer, sheet_name='Formatted Data', index=False)
    summary.to_excel(writer, sheet_name='Summary', index=False)

print('Excel written.')

# ---- Identify high-priority return IDs ----
error_counts = fmt[fmt['Total Errors'] > 0].groupby('Return ID')['Total Errors'].sum().sort_values(ascending=False)
top_ids = error_counts.head(2).index.tolist()
print('Top error Return IDs:', top_ids)

total_missing = int(fmt['Missing Final Event'].sum())
total_mismatch = int(fmt['Disposition Mismatch'].sum())
total_errors = int(fmt['Total Errors'].sum())

# ---- Write Word Brief ----
doc = Document()
doc.add_heading('Returns Disposition Audit – Executive Summary', level=1)

para = (
    f'This audit evaluated returns disposition accuracy using two checks: '
    f'"Missing Final Event" flags return lines with no completed disposition event recorded in the system, '
    f'while "Disposition Mismatch" flags lines where the final recorded disposition differs from the planned disposition after alias normalization. '
    f'Across the audited dataset, {total_missing} Missing Final Event(s), {total_mismatch} Disposition Mismatch(es), '
    f'and {total_errors} Total Error(s) were identified. '
    f'Return IDs {top_ids[0]} and {top_ids[1] if len(top_ids) > 1 else top_ids[0]} exhibited the highest frequency of exceptions and should be prioritized for root-cause investigation. '
    f'We recommend implementing automated disposition confirmation workflows and periodic alias-table reviews to reduce recurring mismatches and missing events.'
)
doc.add_paragraph(para)
doc.save('/root/Returns_Disposition_Brief.docx')
print('Word brief written.')
```

## 3. Validate outputs

After generating the files, verify:

```python
import pandas as pd
from docx import Document

# Check Excel
xl = pd.ExcelFile('/root/Returns_Disposition_Audit.xlsx')
print('Sheet names:', xl.sheet_names)
assert xl.sheet_names == ['RawData', 'Formatted Data', 'Summary'], f'Wrong sheets: {xl.sheet_names}'

raw = pd.read_excel(xl, 'RawData')
fmt = pd.read_excel(xl, 'Formatted Data')
smry = pd.read_excel(xl, 'Summary')

print('RawData columns:', raw.columns.tolist())
print('RawData shape:', raw.shape)
print('Formatted Data columns:', fmt.columns.tolist())
print('Formatted Data shape:', fmt.shape)
print('Summary columns:', smry.columns.tolist())
print('Summary shape:', smry.shape)
print('Summary:\n', smry)

# Check Formatted Data columns
expected_fmt_cols = ['Return ID', 'Line ID', 'Planned Disposition', 'Reason Code',
                     'Requested Qty', 'Warehouse', 'Carrier', 'Lane',
                     'Missing Final Event', 'Disposition Mismatch', 'Total Errors', 'Error Summary']
assert fmt.columns.tolist() == expected_fmt_cols, f'Formatted Data cols mismatch: {fmt.columns.tolist()}'

# Check Summary columns
expected_smry_cols = ['Warehouse', 'Carrier', 'Missing Final Events', 'Disposition Mismatches', 'Total Errors']
assert smry.columns.tolist() == expected_smry_cols, f'Summary cols mismatch: {smry.columns.tolist()}'

# Check Grand Total row
last_row = smry.iloc[-1]
assert last_row['Warehouse'] == 'Grand Total', f'Last row Warehouse: {last_row["Warehouse"]}'
assert last_row['Carrier'] == '-', f'Last row Carrier: {last_row["Carrier"]}'

# Check Word doc
doc = Document('/root/Returns_Disposition_Brief.docx')
text = '\n'.join([p.text for p in doc.paragraphs])
print('Word text preview:', text[:500])
assert 'Missing Final Event' in text
assert 'Disposition Mismatch' in text
print('All validations passed.')
```

## Critical Notes

- **Step 1 is essential.** You MUST inspect the actual column names in all three files before running the main script. The template above uses guessed column names (e.g., 'Alias', 'Standard Disposition', 'Event Status', 'Final Disposition'). You must adapt these to match reality.
- If the event log has a timestamp or sequence column, use it to determine the "latest" completed event. If not, use the original row order (last row = latest).
- All numeric error columns must contain concrete integer values (0 or 1), not formulas.
- Error Summary must be exactly one of the four specified strings.
- Summary sheet must only include (Warehouse, Carrier) groups with Total Errors > 0, plus the Grand Total row at the end.
- The Grand Total row aggregates from the FULL dataset (all rows in Formatted Data), not just the filtered summary rows.
- Keep worksheet names exactly: 'RawData', 'Formatted Data', 'Summary'.
- Keep output file paths exactly: '/root/Returns_Disposition_Audit.xlsx' and '/root/Returns_Disposition_Brief.docx'.

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