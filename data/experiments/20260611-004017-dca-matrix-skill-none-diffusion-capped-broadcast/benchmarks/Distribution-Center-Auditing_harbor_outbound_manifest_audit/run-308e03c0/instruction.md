# Task Instruction

Execute the following Python script to produce both deliverables. The script reads the three input files, processes the data according to the audit rules, and writes `/root/Outbound_Load_Audit.xlsx` and `/root/Outbound_Load_Brief.docx`.

```python
import pandas as pd
import openpyxl
from openpyxl.utils.dataframe import dataframe_to_rows
from docx import Document
import numpy as np

# ── 1. Read source data ──────────────────────────────────────────────
manifest = pd.read_excel('/root/Manifest_Plan.xlsx')
dock_scan = pd.read_excel('/root/Dock_Scan_Log.xlsx')

# ── 2. Prepare the template workbook ─────────────────────────────────
# Copy the template so the Overview sheet is preserved exactly.
wb = openpyxl.load_workbook('/root/Outbound_Audit_Template.xlsx')

# ── 3. RawData sheet ─────────────────────────────────────────────────
# Replace NaN with the original cell representation. Read the source
# again with openpyxl to get the exact cell values (preserving 'N/A'
# strings, blanks, etc.).
src_wb = openpyxl.load_workbook('/root/Manifest_Plan.xlsx')
src_ws = src_wb.active

if 'RawData' not in wb.sheetnames:
    ws_raw = wb.create_sheet('RawData')
else:
    ws_raw = wb['RawData']

for row in src_ws.iter_rows(min_row=1, values_only=False):
    ws_raw.append([cell.value for cell in row])

src_wb.close()

# Also build a pandas DataFrame from the openpyxl source for logic below,
# keeping exact cell values (None stays None, 'N/A' stays 'N/A').
raw_rows = list(src_ws.iter_rows(min_row=1, values_only=True))
raw_headers = [str(h) for h in raw_rows[0]]
manifest_exact = pd.DataFrame(raw_rows[1:], columns=raw_headers)

# ── 4. Derive scan status ────────────────────────────────────────────
# Keep only LOADED scans, then keep the latest per (Shipment ID, Carton ID)
loaded = dock_scan[dock_scan['Status'] == 'LOADED'].copy()

# Determine the ordering column for "latest". Use Scan Timestamp if present,
# otherwise fall back to the original row order.
if 'Scan Timestamp' in loaded.columns:
    loaded = loaded.sort_values('Scan Timestamp')

kept = loaded.drop_duplicates(subset=['Shipment ID', 'Carton ID'], keep='last')

# Build a lookup dict: (Shipment ID, Carton ID) -> Scanned Zone
loaded_lookup = {}
for _, r in kept.iterrows():
    loaded_lookup[(r['Shipment ID'], r['Carton ID'])] = r.get('Scanned Zone', None)

# ── 5. Formatted Data sheet ──────────────────────────────────────────
# Use the exact manifest data and the first 8 columns in order.
first8 = ['Shipment ID', 'Carton ID', 'Planned Zone', 'Route',
          'Expected Weight', 'Hazmat Flag', 'Carrier', 'Wave']

formatted = manifest_exact[first8].copy()

missing_list = []
zone_mm_list = []
for _, r in formatted.iterrows():
    key = (r['Shipment ID'], r['Carton ID'])
    if key not in loaded_lookup:
        missing_list.append(1)
        zone_mm_list.append(0)  # no scan → no mismatch
    else:
        missing_list.append(0)
        scanned_zone = loaded_lookup[key]
        if scanned_zone != r['Planned Zone']:
            zone_mm_list.append(1)
        else:
            zone_mm_list.append(0)

formatted['Missing Load Scan'] = missing_list
formatted['Zone Mismatch'] = zone_mm_list
formatted['Total Errors'] = formatted['Missing Load Scan'] + formatted['Zone Mismatch']

def error_summary(row):
    m = row['Missing Load Scan']
    z = row['Zone Mismatch']
    if m == 1 and z == 1:
        return 'Missing Load Scan, Zone Mismatch'
    elif m == 1:
        return 'Missing Load Scan'
    elif z == 1:
        return 'Zone Mismatch'
    else:
        return 'None'

formatted['Error Summary'] = formatted.apply(error_summary, axis=1)

# Write to sheet
if 'Formatted Data' not in wb.sheetnames:
    ws_fmt = wb.create_sheet('Formatted Data')
else:
    ws_fmt = wb['Formatted Data']

# Header
ws_fmt.append(list(formatted.columns))
for _, r in formatted.iterrows():
    row_vals = []
    for col in formatted.columns:
        v = r[col]
        # Convert numpy types to native Python for openpyxl
        if isinstance(v, (np.integer,)):
            v = int(v)
        elif isinstance(v, (np.floating,)):
            v = float(v) if not np.isnan(v) else None
        row_vals.append(v)
    ws_fmt.append(row_vals)

# ── 6. Summary sheet ─────────────────────────────────────────────────
grouped = formatted.groupby(['Route', 'Shipment ID']).agg(
    **{'Missing Load Scans': ('Missing Load Scan', 'sum'),
       'Zone Mismatches': ('Zone Mismatch', 'sum'),
       'Total Errors': ('Total Errors', 'sum')}
).reset_index()

# Only groups with errors
grouped = grouped[grouped['Total Errors'] > 0].copy()
grouped = grouped.sort_values(['Route', 'Shipment ID']).reset_index(drop=True)

# Grand Total row
grand = pd.DataFrame([{
    'Route': 'Grand Total',
    'Shipment ID': '-',
    'Missing Load Scans': int(grouped['Missing Load Scans'].sum()),
    'Zone Mismatches': int(grouped['Zone Mismatches'].sum()),
    'Total Errors': int(grouped['Total Errors'].sum())
}])
summary = pd.concat([grouped, grand], ignore_index=True)

if 'Summary' not in wb.sheetnames:
    ws_sum = wb.create_sheet('Summary')
else:
    ws_sum = wb['Summary']

ws_sum.append(list(summary.columns))
for _, r in summary.iterrows():
    row_vals = []
    for col in summary.columns:
        v = r[col]
        if isinstance(v, (np.integer,)):
            v = int(v)
        elif isinstance(v, (np.floating,)):
            v = float(v) if not np.isnan(v) else None
        row_vals.append(v)
    ws_sum.append(row_vals)

# ── 7. Save workbook ─────────────────────────────────────────────────
wb.save('/root/Outbound_Load_Audit.xlsx')
print('Saved /root/Outbound_Load_Audit.xlsx')

# ── 8. Word brief ────────────────────────────────────────────────────
total_missing = int(formatted['Missing Load Scan'].sum())
total_zone = int(formatted['Zone Mismatch'].sum())
total_errors = int(formatted['Total Errors'].sum())

# Find top shipment IDs by total errors
ship_errors = formatted.groupby('Shipment ID')['Total Errors'].sum().sort_values(ascending=False)
top_ships = ship_errors[ship_errors > 0].head(2).index.tolist()

doc = Document()
doc.add_heading('Outbound Load Audit Brief', level=1)

para = (
    f'This audit evaluated outbound carton handoff accuracy using two checks: '
    f'"Missing Load Scan" flags cartons listed on the manifest that have no '
    f'confirmed LOADED scan in the dock log, and "Zone Mismatch" flags cartons '
    f'whose scanned loading zone differs from the planned zone. '
    f'Across the audited dataset, {total_missing} Missing Load Scan(s), '
    f'{total_zone} Zone Mismatch(es), and {total_errors} Total Error(s) were identified. '
    f'Shipment IDs {top_ships[0]} and {top_ships[1] if len(top_ships) > 1 else top_ships[0]} '
    f'showed the highest frequency of exceptions and should be prioritized for root-cause investigation. '
    f'It is recommended to implement real-time scan validation alerts at dock doors to '
    f'catch missing scans and zone routing errors before trailers depart.'
)

doc.add_paragraph(para)
doc.save('/root/Outbound_Load_Brief.docx')
print('Saved /root/Outbound_Load_Brief.docx')
print('Done.')
```

After running the script, verify:
1. `/root/Outbound_Load_Audit.xlsx` exists and contains sheets: Overview, RawData, Formatted Data, Summary.
2. `/root/Outbound_Load_Brief.docx` exists.
3. Open the Excel file and confirm RawData has the same content as Manifest_Plan.xlsx (including any 'N/A' strings preserved, not turned into None).
4. Confirm Formatted Data has 12 columns with the four derived columns containing concrete values.
5. Confirm Summary has only error rows plus a Grand Total row, sorted by Route then Shipment ID.

Key lessons from cross-task feedback: When reading Excel with pandas, NaN replaces empty cells and strings like 'N/A'. To preserve exact cell values, use openpyxl directly to read source data for the RawData sheet and for building the manifest DataFrame used in logic. This script does exactly that.

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