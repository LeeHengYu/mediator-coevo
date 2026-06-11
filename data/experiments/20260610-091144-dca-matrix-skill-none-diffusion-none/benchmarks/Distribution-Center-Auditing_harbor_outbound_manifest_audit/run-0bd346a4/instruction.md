# Task Instruction

Execute the following Python script in a single step to produce both deliverables.

```python
import openpyxl
from openpyxl.utils import get_column_letter
import pandas as pd
from docx import Document

# ── Step 0: Inspect source files ──────────────────────────────────────────────
manifest = pd.read_excel('/root/Manifest_Plan.xlsx')
scan_log = pd.read_excel('/root/Dock_Scan_Log.xlsx')

print('Manifest columns:', list(manifest.columns))
print('Manifest shape:', manifest.shape)
print('Manifest head:')
print(manifest.head())
print()
print('Scan log columns:', list(scan_log.columns))
print('Scan log shape:', scan_log.shape)
print('Scan log head:')
print(scan_log.head())
print()

# Inspect the template
wb_template = openpyxl.load_workbook('/root/Outbound_Audit_Template.xlsx')
print('Template sheets:', wb_template.sheetnames)
for s in wb_template.sheetnames:
    ws = wb_template[s]
    print(f'  Sheet "{s}": rows={ws.max_row}, cols={ws.max_column}')
    for r in ws.iter_rows(min_row=1, max_row=min(3, ws.max_row), values_only=False):
        print('   ', [c.value for c in r])
wb_template.close()

# ── Step 1: Prepare the LOADED scan lookup ────────────────────────────────────
# Keep only LOADED rows, sort by any timestamp-like column if present, then
# keep the last occurrence per (Shipment ID, Carton ID).
loaded = scan_log[scan_log['Status'] == 'LOADED'].copy()

# Sort to ensure 'latest' is deterministic; use index as proxy if no timestamp col
time_cols = [c for c in loaded.columns if 'time' in c.lower() or 'date' in c.lower() or 'stamp' in c.lower()]
if time_cols:
    loaded = loaded.sort_values(time_cols[0])
else:
    loaded = loaded.sort_index()

loaded = loaded.drop_duplicates(subset=['Shipment ID', 'Carton ID'], keep='last')

# Build a dict: (Shipment ID, Carton ID) -> Scanned Zone
loaded_lookup = {}
for _, row in loaded.iterrows():
    key = (row['Shipment ID'], row['Carton ID'])
    loaded_lookup[key] = row['Scanned Zone']

print(f'\nLoaded scan lookup size: {len(loaded_lookup)}')

# ── Step 2: Build Formatted Data ──────────────────────────────────────────────
# The first 8 columns from manifest
first8_cols = list(manifest.columns[:8])
formatted = manifest[first8_cols].copy()

# Rename to exact required names (in case of slight differences)
required_names = ['Shipment ID', 'Carton ID', 'Planned Zone', 'Route',
                  'Expected Weight', 'Hazmat Flag', 'Carrier', 'Wave']
rename_map = {old: new for old, new in zip(first8_cols, required_names)}
formatted = formatted.rename(columns=rename_map)

missing_list = []
zone_mm_list = []
total_err_list = []
error_sum_list = []

for _, row in formatted.iterrows():
    key = (row['Shipment ID'], row['Carton ID'])
    scanned_zone = loaded_lookup.get(key, None)
    
    if scanned_zone is None:
        missing = 1
    else:
        missing = 0
    
    if scanned_zone is not None and str(scanned_zone).strip() != str(row['Planned Zone']).strip():
        zone_mm = 1
    else:
        zone_mm = 0
    
    total = missing + zone_mm
    
    parts = []
    if missing == 1:
        parts.append('Missing Load Scan')
    if zone_mm == 1:
        parts.append('Zone Mismatch')
    summary_text = ', '.join(parts) if parts else 'None'
    
    missing_list.append(missing)
    zone_mm_list.append(zone_mm)
    total_err_list.append(total)
    error_sum_list.append(summary_text)

formatted['Missing Load Scan'] = missing_list
formatted['Zone Mismatch'] = zone_mm_list
formatted['Total Errors'] = total_err_list
formatted['Error Summary'] = error_sum_list

print('\nFormatted Data shape:', formatted.shape)
print(formatted.head(10))
print('\nError distribution:')
print(formatted['Error Summary'].value_counts())

# ── Step 3: Build Summary ─────────────────────────────────────────────────────
error_rows = formatted[formatted['Total Errors'] > 0].copy()
agg = error_rows.groupby(['Route', 'Shipment ID'], sort=False).agg(
    **{'Missing Load Scans': ('Missing Load Scan', 'sum'),
       'Zone Mismatches': ('Zone Mismatch', 'sum'),
       'Total Errors': ('Total Errors', 'sum')}
).reset_index()

agg = agg.sort_values(['Route', 'Shipment ID'], ascending=[True, True]).reset_index(drop=True)

# Grand Total row
grand = pd.DataFrame([{
    'Route': 'Grand Total',
    'Shipment ID': '-',
    'Missing Load Scans': agg['Missing Load Scans'].sum(),
    'Zone Mismatches': agg['Zone Mismatches'].sum(),
    'Total Errors': agg['Total Errors'].sum()
}])
summary = pd.concat([agg, grand], ignore_index=True)

print('\nSummary table:')
print(summary.to_string(index=False))

total_missing = int(formatted['Missing Load Scan'].sum())
total_zone_mm = int(formatted['Zone Mismatch'].sum())
total_errors = int(formatted['Total Errors'].sum())
print(f'\nTotals: Missing={total_missing}, ZoneMismatch={total_zone_mm}, TotalErrors={total_errors}')

# ── Step 4: Write Excel ───────────────────────────────────────────────────────
wb = openpyxl.load_workbook('/root/Outbound_Audit_Template.xlsx')

# --- RawData sheet ---
if 'RawData' in wb.sheetnames:
    del wb['RawData']
ws_raw = wb.create_sheet('RawData')

# Write headers from manifest
for c_idx, col_name in enumerate(manifest.columns, 1):
    ws_raw.cell(row=1, column=c_idx, value=col_name)

# Write data
for r_idx, row in enumerate(manifest.itertuples(index=False), 2):
    for c_idx, val in enumerate(row, 1):
        ws_raw.cell(row=r_idx, column=c_idx, value=val)

# --- Formatted Data sheet ---
if 'Formatted Data' in wb.sheetnames:
    del wb['Formatted Data']
ws_fmt = wb.create_sheet('Formatted Data')

fmt_headers = list(formatted.columns)
for c_idx, col_name in enumerate(fmt_headers, 1):
    ws_fmt.cell(row=1, column=c_idx, value=col_name)

for r_idx, row in enumerate(formatted.itertuples(index=False), 2):
    for c_idx, val in enumerate(row, 1):
        # Ensure numeric columns are written as numbers
        if c_idx >= 9:  # error columns
            try:
                val = int(val) if isinstance(val, (int, float)) and c_idx <= 11 else val
            except:
                pass
        ws_fmt.cell(row=r_idx, column=c_idx, value=val)

# --- Summary sheet ---
if 'Summary' in wb.sheetnames:
    del wb['Summary']
ws_sum = wb.create_sheet('Summary')

sum_headers = list(summary.columns)
for c_idx, col_name in enumerate(sum_headers, 1):
    ws_sum.cell(row=1, column=c_idx, value=col_name)

for r_idx, row in enumerate(summary.itertuples(index=False), 2):
    for c_idx, val in enumerate(row, 1):
        if c_idx >= 3:
            try:
                val = int(val)
            except:
                pass
        ws_sum.cell(row=r_idx, column=c_idx, value=val)

wb.save('/root/Outbound_Load_Audit.xlsx')
wb.close()
print('\nExcel saved to /root/Outbound_Load_Audit.xlsx')

# ── Step 5: Verify Excel ──────────────────────────────────────────────────────
wb_check = openpyxl.load_workbook('/root/Outbound_Load_Audit.xlsx')
print('Sheets in output:', wb_check.sheetnames)
assert 'Overview' in wb_check.sheetnames, 'Overview sheet missing!'
assert 'RawData' in wb_check.sheetnames, 'RawData sheet missing!'
assert 'Formatted Data' in wb_check.sheetnames, 'Formatted Data sheet missing!'
assert 'Summary' in wb_check.sheetnames, 'Summary sheet missing!'

ws_fd = wb_check['Formatted Data']
print(f'Formatted Data: {ws_fd.max_row} rows, {ws_fd.max_column} cols')
print('FD headers:', [ws_fd.cell(1, c).value for c in range(1, ws_fd.max_column+1)])

ws_s = wb_check['Summary']
print(f'Summary: {ws_s.max_row} rows, {ws_s.max_column} cols')
for r in range(1, ws_s.max_row+1):
    print('  ', [ws_s.cell(r, c).value for c in range(1, ws_s.max_column+1)])
wb_check.close()

# ── Step 6: Find top-2 shipment IDs with most errors ─────────────────────────
ship_errors = formatted.groupby('Shipment ID')['Total Errors'].sum().sort_values(ascending=False)
top2 = ship_errors.head(2)
print('\nTop 2 shipment IDs by errors:')
print(top2)
top2_ids = list(top2.index)

# ── Step 7: Write Word Brief ─────────────────────────────────────────────────
doc = Document()
doc.add_heading('Outbound Load Audit – Executive Summary', level=1)

para = (
    f'This audit evaluated outbound carton handoff accuracy across all planned shipments. '
    f'Two checks were applied: "Missing Load Scan" flags a carton that has no confirmed LOADED scan '
    f'in the dock scan log, indicating it may not have been physically loaded onto the trailer; '
    f'"Zone Mismatch" flags a carton whose scanned dock zone differs from the planned zone, '
    f'suggesting it was loaded at the wrong location. '
    f'Across the dataset, {total_missing} Missing Load Scan(s), {total_zone_mm} Zone Mismatch(es), '
    f'and {total_errors} Total Error(s) were identified. '
    f'Shipment IDs {top2_ids[0]} and {top2_ids[1]} had the highest number of exceptions and should be '
    f'prioritized for root-cause investigation. '
    f'We recommend implementing real-time zone-validation alerts at scan stations and conducting '
    f'a dock-staffing review during peak wave windows to reduce missed scans.'
)
doc.add_paragraph(para)

doc.save('/root/Outbound_Load_Brief.docx')
print('\nWord brief saved to /root/Outbound_Load_Brief.docx')
print('DONE')
```

Run this script with `python` in a single execution. Verify that both `/root/Outbound_Load_Audit.xlsx` and `/root/Outbound_Load_Brief.docx` are created and that the console output shows correct sheet names, row counts, and totals matching the Grand Total row in the Summary sheet.

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