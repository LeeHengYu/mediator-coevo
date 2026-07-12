# Task Instruction

Execute the following steps in order.

## 1. Inspect source files and template

```python
import openpyxl, pprint

# Inspect Manifest_Plan.xlsx
wb_mp = openpyxl.load_workbook('/root/Manifest_Plan.xlsx')
for s in wb_mp.sheetnames:
    ws = wb_mp[s]
    print(f'--- Manifest_Plan / {s} ---')
    for r in ws.iter_rows(min_row=1, max_row=min(ws.max_row,10), values_only=False):
        print([c.value for c in r])
    print(f'Total rows: {ws.max_row}')
wb_mp.close()

# Inspect Dock_Scan_Log.xlsx
wb_ds = openpyxl.load_workbook('/root/Dock_Scan_Log.xlsx')
for s in wb_ds.sheetnames:
    ws = wb_ds[s]
    print(f'--- Dock_Scan_Log / {s} ---')
    for r in ws.iter_rows(min_row=1, max_row=min(ws.max_row,10), values_only=False):
        print([c.value for c in r])
    print(f'Total rows: {ws.max_row}')
wb_ds.close()

# Inspect template
wb_t = openpyxl.load_workbook('/root/Outbound_Audit_Template.xlsx')
print('Template sheets:', wb_t.sheetnames)
for s in wb_t.sheetnames:
    ws = wb_t[s]
    print(f'--- Template / {s} ---')
    for r in ws.iter_rows(min_row=1, max_row=min(ws.max_row,10), values_only=False):
        print([c.value for c in r])
    print(f'Total rows: {ws.max_row}')
wb_t.close()
```

Read the output carefully. Note:
- Exact column headers in Manifest_Plan (there should be 8 columns matching: Shipment ID, Carton ID, Planned Zone, Route, Expected Weight, Hazmat Flag, Carrier, Wave).
- Exact column headers in Dock_Scan_Log (look for Shipment ID, Carton ID, Scanned Zone, Status, and any timestamp/sequence column).
- Which sheets exist in the template and what is on the Overview sheet.

## 2. Build the output workbook and Word document

After inspecting, run one comprehensive Python script. Adapt column names to whatever you observed in Step 1. The script must:

```python
import pandas as pd
import openpyxl
from openpyxl.utils.dataframe import dataframe_to_rows
from copy import copy
from docx import Document

# ── Load data ──
manifest = pd.read_excel('/root/Manifest_Plan.xlsx')
dock = pd.read_excel('/root/Dock_Scan_Log.xlsx')

# Print columns to confirm
print('Manifest columns:', list(manifest.columns))
print('Dock columns:', list(dock.columns))
print('Manifest shape:', manifest.shape)
print('Dock shape:', dock.shape)

# ── Identify the correct column names from the dock scan log ──
# Adjust these variable assignments based on what you see:
# sid_col_dock, cid_col_dock, zone_col_dock, status_col_dock, time_col_dock
# For manifest: sid_col_man, cid_col_man, zone_col_man
# Use .str.strip() on string columns to avoid whitespace issues.

# Strip whitespace from all string columns in both dataframes
for df in [manifest, dock]:
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].astype(str).str.strip()

# ── Filter dock scan log: keep only rows with Status == 'LOADED' ──
loaded = dock[dock['Status'] == 'LOADED'].copy()  # adjust col name if needed

# ── Keep only the LATEST LOADED scan per (Shipment ID, Carton ID) ──
# Sort by timestamp/sequence column descending, then drop duplicates keeping first
# Identify the timestamp or sequence column from Step 1 output
# If there's a Timestamp column, sort by it; otherwise use index as proxy
time_cols = [c for c in loaded.columns if 'time' in c.lower() or 'stamp' in c.lower() or 'seq' in c.lower() or 'scan' in c.lower() and 'id' not in c.lower()]
print('Potential time columns:', time_cols)

# Sort loaded by the time column descending (adjust column name)
# If no clear time column, sort by index descending
if time_cols:
    sort_col = time_cols[0]  # pick the most likely one
    loaded = loaded.sort_values(sort_col, ascending=False)
else:
    loaded = loaded.iloc[::-1]  # reverse so later rows come first

kept_scans = loaded.drop_duplicates(subset=['Shipment ID', 'Carton ID'], keep='first')
print(f'Kept scans: {len(kept_scans)} unique (Shipment ID, Carton ID) pairs')

# ── Build Formatted Data ──
formatted = manifest.copy()

# Merge with kept scans to get Scanned Zone
merged = formatted.merge(
    kept_scans[['Shipment ID', 'Carton ID', 'Scanned Zone']],  # adjust col names
    on=['Shipment ID', 'Carton ID'],
    how='left',
    suffixes=('', '_scan')
)

# Derive columns
merged['Missing Load Scan'] = merged['Scanned Zone'].isna().astype(int)
merged['Zone Mismatch'] = ((merged['Scanned Zone'].notna()) & (merged['Scanned Zone'] != merged['Planned Zone'])).astype(int)
merged['Total Errors'] = merged['Missing Load Scan'] + merged['Zone Mismatch']

def error_summary(row):
    parts = []
    if row['Missing Load Scan'] == 1:
        parts.append('Missing Load Scan')
    if row['Zone Mismatch'] == 1:
        parts.append('Zone Mismatch')
    return ', '.join(parts) if parts else 'None'

merged['Error Summary'] = merged.apply(error_summary, axis=1)

# Keep only the required 12 columns for Formatted Data
formatted_cols = [
    'Shipment ID', 'Carton ID', 'Planned Zone', 'Route',
    'Expected Weight', 'Hazmat Flag', 'Carrier', 'Wave',
    'Missing Load Scan', 'Zone Mismatch', 'Total Errors', 'Error Summary'
]
formatted_df = merged[formatted_cols].copy()

print('\nFormatted Data sample:')
print(formatted_df.head(10))
print(f'Total Missing Load Scans: {formatted_df["Missing Load Scan"].sum()}')
print(f'Total Zone Mismatches: {formatted_df["Zone Mismatch"].sum()}')
print(f'Total Errors: {formatted_df["Total Errors"].sum()}')

# ── Build Summary ──
summary = formatted_df.groupby(['Route', 'Shipment ID']).agg(
    **{'Missing Load Scans': ('Missing Load Scan', 'sum'),
       'Zone Mismatches': ('Zone Mismatch', 'sum'),
       'Total Errors': ('Total Errors', 'sum')}
).reset_index()

# Keep only groups with Total Errors > 0
summary = summary[summary['Total Errors'] > 0].copy()
summary = summary.sort_values(['Route', 'Shipment ID'], ascending=[True, True]).reset_index(drop=True)

# Grand Total row
grand = pd.DataFrame([{
    'Route': 'Grand Total',
    'Shipment ID': '-',
    'Missing Load Scans': summary['Missing Load Scans'].sum(),
    'Zone Mismatches': summary['Zone Mismatches'].sum(),
    'Total Errors': summary['Total Errors'].sum()
}])
summary_final = pd.concat([summary, grand], ignore_index=True)

print('\nSummary:')
print(summary_final.to_string())

# ── Write Excel workbook ──
wb = openpyxl.load_workbook('/root/Outbound_Audit_Template.xlsx')

# -- RawData sheet --
if 'RawData' in wb.sheetnames:
    del wb['RawData']
ws_raw = wb.create_sheet('RawData')
for r_idx, row in enumerate(dataframe_to_rows(manifest, index=False, header=True), 1):
    for c_idx, val in enumerate(row, 1):
        ws_raw.cell(row=r_idx, column=c_idx, value=val)

# -- Formatted Data sheet --
if 'Formatted Data' in wb.sheetnames:
    del wb['Formatted Data']
ws_fmt = wb.create_sheet('Formatted Data')
for r_idx, row in enumerate(dataframe_to_rows(formatted_df, index=False, header=True), 1):
    for c_idx, val in enumerate(row, 1):
        cell = ws_fmt.cell(row=r_idx, column=c_idx)
        # Convert numpy types to native Python
        if hasattr(val, 'item'):
            val = val.item()
        cell.value = val

# -- Summary sheet --
if 'Summary' in wb.sheetnames:
    del wb['Summary']
ws_sum = wb.create_sheet('Summary')
for r_idx, row in enumerate(dataframe_to_rows(summary_final, index=False, header=True), 1):
    for c_idx, val in enumerate(row, 1):
        cell = ws_sum.cell(row=r_idx, column=c_idx)
        if hasattr(val, 'item'):
            val = val.item()
        cell.value = val

wb.save('/root/Outbound_Load_Audit.xlsx')
print('\nSaved Outbound_Load_Audit.xlsx')
print('Sheets:', openpyxl.load_workbook('/root/Outbound_Load_Audit.xlsx').sheetnames)

# ── Identify high-priority shipment IDs ──
# Top shipment IDs by Total Errors
ship_errors = formatted_df.groupby('Shipment ID')['Total Errors'].sum().sort_values(ascending=False)
print('\nTop shipment IDs by errors:')
print(ship_errors.head(5))
top_ships = ship_errors.head(4).index.tolist()

total_missing = int(formatted_df['Missing Load Scan'].sum())
total_zone = int(formatted_df['Zone Mismatch'].sum())
total_err = int(formatted_df['Total Errors'].sum())

# ── Write Word document ──
doc = Document()
doc.add_heading('Outbound Load Audit – Executive Brief', level=1)

para = (
    f'This audit reviewed outbound carton handoff accuracy by applying two checks to every planned carton: '
    f'(1) Missing Load Scan, which flags any carton that was never recorded as LOADED in the dock scan log, '
    f'and (2) Zone Mismatch, which flags any carton whose scanned loading zone differs from its planned zone. '
    f'Across the dataset, {total_missing} Missing Load Scans, {total_zone} Zone Mismatches, '
    f'and {total_err} Total Errors were identified. '
    f'Shipment IDs {top_ships[0]} and {top_ships[1]} had the highest frequency of exceptions and should be investigated first. '
    f'We recommend implementing real-time scan validation alerts at each dock door to catch missing and misrouted cartons '
    f'before trailers depart, and conducting a root-cause review of the zones assigned to high-error shipments.'
)
doc.add_paragraph(para)
doc.save('/root/Outbound_Load_Brief.docx')
print('Saved Outbound_Load_Brief.docx')
```

**CRITICAL adaptation notes — read before running:**
- After Step 1, adjust ALL column name references in Step 2 to match the EXACT column headers you observed. Watch for variations like 'Scanned Zone' vs 'Scan Zone', 'Status' vs 'Scan Status', etc.
- If the dock scan log has no obvious timestamp column, use the DataFrame index (later rows = later scans) to determine the "latest" LOADED scan.
- Ensure `.str.strip()` is applied to string columns to avoid whitespace mismatches.
- The `Zone Mismatch` check must compare strings after stripping. If Planned Zone or Scanned Zone are numeric, convert both to strings first.
- Make sure the Grand Total row sums come from the filtered summary (groups with errors > 0), matching the rows actually shown.
- For the Word document, explicitly include the numeric totals as plain strings (e.g., the number itself) so the verifier can find them with string matching.
- Mention at least two specific Shipment IDs from the top-error list in the Word document text.

## 3. Validate outputs

```python
import openpyxl
from docx import Document

wb = openpyxl.load_workbook('/root/Outbound_Load_Audit.xlsx')
print('Sheets:', wb.sheetnames)
assert 'Overview' in wb.sheetnames
assert 'RawData' in wb.sheetnames
assert 'Formatted Data' in wb.sheetnames
assert 'Summary' in wb.sheetnames

# Check RawData has data
ws = wb['RawData']
print(f'RawData: {ws.max_row} rows, {ws.max_column} cols')
print('RawData headers:', [ws.cell(1,c).value for c in range(1, ws.max_column+1)])

# Check Formatted Data
ws = wb['Formatted Data']
print(f'Formatted Data: {ws.max_row} rows, {ws.max_column} cols')
headers = [ws.cell(1,c).value for c in range(1, ws.max_column+1)]
print('Headers:', headers)
assert ws.max_column == 12, f'Expected 12 columns, got {ws.max_column}'
assert headers[8] == 'Missing Load Scan'
assert headers[9] == 'Zone Mismatch'
assert headers[10] == 'Total Errors'
assert headers[11] == 'Error Summary'
# Check a few data cells are concrete values not formulas
for r in range(2, min(ws.max_row+1, 5)):
    v = ws.cell(r, 9).value
    print(f'  Row {r} Missing Load Scan = {v} (type: {type(v).__name__})')
    assert v in (0, 1, 0.0, 1.0), f'Unexpected value {v}'

# Check Summary
ws = wb['Summary']
print(f'Summary: {ws.max_row} rows, {ws.max_column} cols')
print('Summary headers:', [ws.cell(1,c).value for c in range(1, ws.max_column+1)])
last_row = ws.max_row
print(f'Last row: {[ws.cell(last_row,c).value for c in range(1, ws.max_column+1)]}')
assert ws.cell(last_row, 1).value == 'Grand Total'
assert ws.cell(last_row, 2).value == '-'

# Check Word doc
doc = Document('/root/Outbound_Load_Brief.docx')
text = ' '.join([p.text for p in doc.paragraphs])
print('Word text length:', len(text))
print('Word text preview:', text[:500])
assert 'Missing Load Scan' in text
assert 'Zone Mismatch' in text

print('\nAll validation checks passed!')
```

If any assertion fails or column names don't match, fix the issue and re-run.

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
Task metadata: author_email=catpaw@meituan.com, author_name=CatPaw Benchmark Builder, category=spreadsheet-audit, difficulty=hard, tags=[excel, openpyxl, docx, audit, logistics].
Verifier config: timeout_sec=900.0.