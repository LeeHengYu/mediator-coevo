# Task Instruction

## Task: Outbound Load Audit for Harbor Distribution Center

You must produce two files:
1. `/root/Outbound_Load_Audit.xlsx`
2. `/root/Outbound_Load_Brief.docx`

### Step-by-step execution plan

#### Step 0: Inspect all input files

```bash
pip install openpyxl python-docx pandas
```

Then run a Python script to inspect the three input files:

```python
import openpyxl, pprint

# Inspect Manifest_Plan.xlsx
wb1 = openpyxl.load_workbook('/root/Manifest_Plan.xlsx')
for s in wb1.sheetnames:
    ws = wb1[s]
    print(f'=== Manifest_Plan / {s} ===')
    for i, row in enumerate(ws.iter_rows(values_only=True)):
        if i < 5 or i == ws.max_row - 1:
            print(i, row)
    print(f'Rows: {ws.max_row}, Cols: {ws.max_column}')

# Inspect Dock_Scan_Log.xlsx
wb2 = openpyxl.load_workbook('/root/Dock_Scan_Log.xlsx')
for s in wb2.sheetnames:
    ws = wb2[s]
    print(f'\n=== Dock_Scan_Log / {s} ===')
    for i, row in enumerate(ws.iter_rows(values_only=True)):
        if i < 5 or i == ws.max_row - 1:
            print(i, row)
    print(f'Rows: {ws.max_row}, Cols: {ws.max_column}')

# Inspect Outbound_Audit_Template.xlsx
wb3 = openpyxl.load_workbook('/root/Outbound_Audit_Template.xlsx')
for s in wb3.sheetnames:
    ws = wb3[s]
    print(f'\n=== Template / {s} ===')
    for i, row in enumerate(ws.iter_rows(values_only=True)):
        print(i, row)
    print(f'Rows: {ws.max_row}, Cols: {ws.max_column}')
```

Carefully read ALL output. Note:
- Exact column headers in Manifest_Plan (these become RawData columns)
- Exact column headers in Dock_Scan_Log (find Shipment ID, Carton ID, Status, Scanned Zone, and any timestamp/sequence column)
- Which sheets exist in the template and their exact content (especially `Overview`)

#### Step 1: Build the main processing script

After inspecting the data, write and run a comprehensive Python script that does everything below. **Adapt column names to match what you actually see in the files.**

```python
import pandas as pd
import openpyxl
from openpyxl.utils.dataframe import dataframe_to_rows
from copy import copy
from docx import Document

# ---- Load data ----
manifest = pd.read_excel('/root/Manifest_Plan.xlsx')
dock_log = pd.read_excel('/root/Dock_Scan_Log.xlsx')

# Print columns to confirm
print('Manifest columns:', list(manifest.columns))
print('Dock log columns:', list(dock_log.columns))
print('Manifest shape:', manifest.shape)
print('Dock log shape:', dock_log.shape)

# ---- Identify the correct column names from dock log ----
# Adapt these variable names based on actual column headers found in Step 0
# e.g., shipment_id_col, carton_id_col, status_col, scanned_zone_col, timestamp_col

# ---- Filter dock log: keep only LOADED status ----
loaded = dock_log[dock_log['Status'] == 'LOADED'].copy()  # adapt 'Status' if needed

# ---- Keep latest LOADED scan per (Shipment ID, Carton ID) ----
# Sort by timestamp/sequence descending, then drop_duplicates keeping first
# IMPORTANT: Identify the timestamp or sequence column from Step 0
# If there's a Scan_Timestamp or Scan_Time or similar column, sort by it
loaded = loaded.sort_values(by='Scan_Timestamp', ascending=False)  # adapt column name
loaded_latest = loaded.drop_duplicates(subset=['Shipment ID', 'Carton ID'], keep='first')

# Create a lookup: (Shipment ID, Carton ID) -> Scanned Zone
scan_lookup = {}
for _, row in loaded_latest.iterrows():
    key = (row['Shipment ID'], row['Carton ID'])
    scan_lookup[key] = row['Scanned Zone']  # adapt column name

print(f'Manifest rows: {len(manifest)}, Unique LOADED scans: {len(scan_lookup)}')

# ---- Build Formatted Data ----
formatted = manifest.copy()

# Ensure first 8 columns match required names
# Rename if necessary based on actual manifest columns
required_cols = ['Shipment ID', 'Carton ID', 'Planned Zone', 'Route',
                 'Expected Weight', 'Hazmat Flag', 'Carrier', 'Wave']
# Verify these match: print(list(formatted.columns[:8]))

missing_scan = []
zone_mismatch = []
total_errors = []
error_summary = []

for _, row in formatted.iterrows():
    key = (row['Shipment ID'], row['Carton ID'])
    if key not in scan_lookup:
        ms = 1
        zm = 0  # No scan exists, so no zone comparison possible
    else:
        ms = 0
        scanned_zone = scan_lookup[key]
        zm = 1 if scanned_zone != row['Planned Zone'] else 0
    
    te = ms + zm
    
    # Build error summary string
    parts = []
    if ms == 1:
        parts.append('Missing Load Scan')
    if zm == 1:
        parts.append('Zone Mismatch')
    es = ', '.join(parts) if parts else 'None'
    
    missing_scan.append(ms)
    zone_mismatch.append(zm)
    total_errors.append(te)
    error_summary.append(es)

formatted['Missing Load Scan'] = missing_scan
formatted['Zone Mismatch'] = zone_mismatch
formatted['Total Errors'] = total_errors
formatted['Error Summary'] = error_summary

print(f'Total Missing Load Scans: {sum(missing_scan)}')
print(f'Total Zone Mismatches: {sum(zone_mismatch)}')
print(f'Total Errors: {sum(total_errors)}')

# ---- Build Summary ----
# Aggregate by (Route, Shipment ID) from formatted data
summary_agg = formatted.groupby(['Route', 'Shipment ID']).agg(
    **{'Missing Load Scans': ('Missing Load Scan', 'sum'),
       'Zone Mismatches': ('Zone Mismatch', 'sum'),
       'Total Errors': ('Total Errors', 'sum')}
).reset_index()

# Keep only groups with Total Errors > 0
summary_agg = summary_agg[summary_agg['Total Errors'] > 0].copy()

# Sort by Route asc, Shipment ID asc
summary_agg = summary_agg.sort_values(by=['Route', 'Shipment ID']).reset_index(drop=True)

# Append Grand Total row
grand_total = pd.DataFrame([{
    'Route': 'Grand Total',
    'Shipment ID': '-',
    'Missing Load Scans': summary_agg['Missing Load Scans'].sum(),
    'Zone Mismatches': summary_agg['Zone Mismatches'].sum(),
    'Total Errors': summary_agg['Total Errors'].sum()
}])
summary_final = pd.concat([summary_agg, grand_total], ignore_index=True)

print('\nSummary table:')
print(summary_final.to_string())

# ---- Find top shipment IDs for the Word doc ----
top_shipments = summary_agg.nlargest(2, 'Total Errors')['Shipment ID'].tolist()
print(f'Top error shipments: {top_shipments}')

# ---- Write Excel ----
# Start from template
wb = openpyxl.load_workbook('/root/Outbound_Audit_Template.xlsx')

# Verify Overview sheet exists and will be preserved
assert 'Overview' in wb.sheetnames, 'Overview sheet missing from template!'

# Create RawData sheet
if 'RawData' in wb.sheetnames:
    del wb['RawData']
ws_raw = wb.create_sheet('RawData')
for r_idx, row in enumerate(dataframe_to_rows(manifest, index=False, header=True), 1):
    for c_idx, val in enumerate(row, 1):
        ws_raw.cell(row=r_idx, column=c_idx, value=val)

# Create Formatted Data sheet
if 'Formatted Data' in wb.sheetnames:
    del wb['Formatted Data']
ws_fmt = wb.create_sheet('Formatted Data')
for r_idx, row in enumerate(dataframe_to_rows(formatted, index=False, header=True), 1):
    for c_idx, val in enumerate(row, 1):
        cell = ws_fmt.cell(row=r_idx, column=c_idx, value=val)
        # Ensure numeric columns are written as integers not floats
        if r_idx > 1 and c_idx in [9, 10, 11]:  # Missing Load Scan, Zone Mismatch, Total Errors
            cell.value = int(val)

# Create Summary sheet
if 'Summary' in wb.sheetnames:
    del wb['Summary']
ws_sum = wb.create_sheet('Summary')
for r_idx, row in enumerate(dataframe_to_rows(summary_final, index=False, header=True), 1):
    for c_idx, val in enumerate(row, 1):
        cell = ws_sum.cell(row=r_idx, column=c_idx, value=val)
        # Ensure numeric values are integers
        if r_idx > 1 and c_idx in [3, 4, 5]:
            try:
                cell.value = int(val)
            except (ValueError, TypeError):
                pass

wb.save('/root/Outbound_Load_Audit.xlsx')
print('\nExcel saved successfully.')

# ---- Verify Excel output ----
wb_check = openpyxl.load_workbook('/root/Outbound_Load_Audit.xlsx')
print('Sheets:', wb_check.sheetnames)
assert 'Overview' in wb_check.sheetnames
assert 'RawData' in wb_check.sheetnames
assert 'Formatted Data' in wb_check.sheetnames
assert 'Summary' in wb_check.sheetnames

# Check Formatted Data headers
ws_check = wb_check['Formatted Data']
headers = [ws_check.cell(row=1, column=c).value for c in range(1, 13)]
print('Formatted Data headers:', headers)
assert headers[8] == 'Missing Load Scan'
assert headers[9] == 'Zone Mismatch'
assert headers[10] == 'Total Errors'
assert headers[11] == 'Error Summary'

# Check Summary headers
ws_sum_check = wb_check['Summary']
sum_headers = [ws_sum_check.cell(row=1, column=c).value for c in range(1, 6)]
print('Summary headers:', sum_headers)
assert sum_headers == ['Route', 'Shipment ID', 'Missing Load Scans', 'Zone Mismatches', 'Total Errors']

# Check last row is Grand Total
last_row = ws_sum_check.max_row
assert ws_sum_check.cell(row=last_row, column=1).value == 'Grand Total'
assert ws_sum_check.cell(row=last_row, column=2).value == '-'
print(f'Grand Total row: Missing={ws_sum_check.cell(row=last_row, column=3).value}, Zone={ws_sum_check.cell(row=last_row, column=4).value}, Total={ws_sum_check.cell(row=last_row, column=5).value}')

print('\nAll Excel checks passed!')

# ---- Compute totals for Word doc ----
total_missing = sum(missing_scan)
total_zone = sum(zone_mismatch)
total_err = sum(total_errors)

# ---- Write Word document ----
doc = Document()
doc.add_heading('Outbound Load Audit Brief', level=1)

# Build executive summary with all required elements
paragraph_text = (
    f'This audit evaluated outbound carton handoff accuracy using two key checks: '
    f'a Missing Load Scan check identifies cartons listed in the manifest plan that '
    f'were never scanned as LOADED at the dock, and a Zone Mismatch check flags cartons '
    f'whose scanned dock zone differs from the planned zone assignment. '
    f'Across all audited shipments, the analysis found {total_missing} Missing Load Scans, '
    f'{total_zone} Zone Mismatches, and {total_err} Total Errors. '
    f'Shipments {top_shipments[0]} and {top_shipments[1]} had the highest frequency of '
    f'exceptions and should be prioritized for root-cause investigation. '
    f'We recommend implementing real-time scan validation alerts at dock stations to '
    f'catch missing scans and zone routing errors before trucks depart, and conducting '
    f'a focused review of zone assignment logic for the affected routes.'
)
doc.add_paragraph(paragraph_text)

doc.save('/root/Outbound_Load_Brief.docx')
print('\nWord document saved successfully.')

# Verify Word doc
from docx import Document as DocRead
doc_check = DocRead('/root/Outbound_Load_Brief.docx')
full_text = ' '.join([p.text for p in doc_check.paragraphs])
print('Word doc text preview:', full_text[:500])
assert str(total_missing) in full_text, f'Missing total {total_missing} not in doc'
assert str(total_zone) in full_text, f'Zone total {total_zone} not in doc'
assert str(total_err) in full_text, f'Error total {total_err} not in doc'
assert 'Missing Load Scan' in full_text
assert 'Zone Mismatch' in full_text
for sid in top_shipments:
    assert str(sid) in full_text, f'Shipment {sid} not in doc'
print('All Word checks passed!')
```

#### CRITICAL NOTES based on past failures:

1. **Column name matching**: You MUST inspect the actual column headers in both Excel files before writing the processing code. Do NOT assume column names — adapt the code to match what you find. Common issues: spaces vs underscores, different capitalization, slightly different names like 'Scan_Timestamp' vs 'Timestamp' vs 'Scan Time'.

2. **Missing Load Scan logic**: When `Missing Load Scan = 1` (no LOADED scan exists), `Zone Mismatch` MUST be 0. You cannot compare zones if there's no scan. The Error Summary for this case is just `Missing Load Scan`, NOT `Missing Load Scan, Zone Mismatch`.

3. **Latest LOADED scan**: You must identify the correct column to determine which scan is "latest" (timestamp, datetime, sequence number, or row order). Sort descending and keep first per (Shipment ID, Carton ID).

4. **Numeric values must be integers**: Write 0 and 1 as integers, not floats (0.0, 1.0). The verifier likely checks exact values.

5. **Overview sheet preservation**: Do NOT modify or delete the Overview sheet from the template.

6. **Grand Total row**: The numeric values must equal the sum of all rows in the Summary table (not from Formatted Data directly — though they should match since Summary aggregates from Formatted Data for error rows only).

7. **Word document**: Must contain the actual computed numeric totals as strings, mention at least 2 shipment IDs, define both checks in plain language, and include a recommendation.

8. **If the dock log has no timestamp column**, use the original row order (index) as the sequence — later rows are more recent.

9. **Run the inspection step FIRST and read the output carefully before writing the main script.** Adapt all column references accordingly.

10. **After saving, re-read and verify both output files to confirm correctness.**

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