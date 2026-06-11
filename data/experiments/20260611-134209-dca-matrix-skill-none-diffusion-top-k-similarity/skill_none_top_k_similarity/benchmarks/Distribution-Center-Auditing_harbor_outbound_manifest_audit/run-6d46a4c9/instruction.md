# Task Instruction

Execute the following Python script in `/root` to produce both deliverables. Before running, inspect the input files to understand their structure, then run the script and verify the outputs exist and are well-formed.

```bash
cd /root
python3 << 'PYEOF'
import pandas as pd
import openpyxl
from openpyxl.utils.dataframe import dataframe_to_rows
from copy import copy
from docx import Document

# ── 1. Read inputs ──────────────────────────────────────────────
manifest = pd.read_excel('Manifest_Plan.xlsx')
dock_log = pd.read_excel('Dock_Scan_Log.xlsx')

# ── 2. Prepare scan lookup ──────────────────────────────────────
# Keep only LOADED rows, then keep the latest per (Shipment ID, Carton ID)
loaded = dock_log[dock_log['Status'] == 'LOADED'].copy()
# Determine the recency column – try Timestamp or Scan Timestamp or similar
time_col = None
for c in loaded.columns:
    if 'time' in c.lower() or 'stamp' in c.lower() or 'date' in c.lower() or 'seq' in c.lower():
        time_col = c
        break
if time_col is None:
    # fallback: use original row order (index) as proxy for recency
    loaded = loaded.reset_index().rename(columns={'index': '_orig_idx'})
    time_col = '_orig_idx'

loaded = loaded.sort_values(time_col, ascending=True)
latest_loaded = loaded.drop_duplicates(subset=['Shipment ID', 'Carton ID'], keep='last')

# Build a dict: (Shipment ID, Carton ID) -> Scanned Zone
scan_map = {}
for _, row in latest_loaded.iterrows():
    scan_map[(row['Shipment ID'], row['Carton ID'])] = row['Scanned Zone']

# ── 3. Build Formatted Data ─────────────────────────────────────
fmt = manifest.copy()
# Ensure first 8 columns match expected names
expected_cols = ['Shipment ID', 'Carton ID', 'Planned Zone', 'Route',
                 'Expected Weight', 'Hazmat Flag', 'Carrier', 'Wave']
# Rename if needed (use positional mapping as safety net)
for i, exp in enumerate(expected_cols):
    if fmt.columns[i] != exp:
        fmt.rename(columns={fmt.columns[i]: exp}, inplace=True)

missing_list = []
zone_mm_list = []
total_err_list = []
err_summary_list = []

for _, row in fmt.iterrows():
    key = (row['Shipment ID'], row['Carton ID'])
    if key not in scan_map:
        missing = 1
        zone_mm = 0  # no scan => no zone mismatch
    else:
        missing = 0
        zone_mm = 1 if scan_map[key] != row['Planned Zone'] else 0
    total = missing + zone_mm
    parts = []
    if missing:
        parts.append('Missing Load Scan')
    if zone_mm:
        parts.append('Zone Mismatch')
    summary = ', '.join(parts) if parts else 'None'
    missing_list.append(missing)
    zone_mm_list.append(zone_mm)
    total_err_list.append(total)
    err_summary_list.append(summary)

fmt['Missing Load Scan'] = missing_list
fmt['Zone Mismatch'] = zone_mm_list
fmt['Total Errors'] = total_err_list
fmt['Error Summary'] = err_summary_list

# ── 4. Build Summary ────────────────────────────────────────────
agg = fmt.groupby(['Route', 'Shipment ID'], sort=False).agg(
    **{'Missing Load Scans': ('Missing Load Scan', 'sum'),
       'Zone Mismatches': ('Zone Mismatch', 'sum'),
       'Total Errors': ('Total Errors', 'sum')}
).reset_index()
agg = agg[agg['Total Errors'] > 0].copy()
agg = agg.sort_values(['Route', 'Shipment ID'], ascending=[True, True]).reset_index(drop=True)

grand = pd.DataFrame([{
    'Route': 'Grand Total',
    'Shipment ID': '-',
    'Missing Load Scans': agg['Missing Load Scans'].sum(),
    'Zone Mismatches': agg['Zone Mismatches'].sum(),
    'Total Errors': agg['Total Errors'].sum()
}])
summary_df = pd.concat([agg, grand], ignore_index=True)

# ── 5. Write Excel workbook ─────────────────────────────────────
# Start from the template to preserve Overview sheet
wb = openpyxl.load_workbook('Outbound_Audit_Template.xlsx')

def write_df_to_sheet(wb, sheet_name, df):
    if sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
    else:
        ws = wb.create_sheet(sheet_name)
    for r_idx, row in enumerate(dataframe_to_rows(df, index=False, header=True), 1):
        for c_idx, val in enumerate(row, 1):
            ws.cell(row=r_idx, column=c_idx, value=val)

write_df_to_sheet(wb, 'RawData', manifest)
write_df_to_sheet(wb, 'Formatted Data', fmt)
write_df_to_sheet(wb, 'Summary', summary_df)

wb.save('Outbound_Load_Audit.xlsx')
print('Excel saved.')

# ── 6. Identify high-priority shipments ─────────────────────────
top_shipments = (fmt[fmt['Total Errors'] > 0]
    .groupby('Shipment ID')['Total Errors'].sum()
    .sort_values(ascending=False))
top2 = list(top_shipments.index[:2])

total_missing = int(fmt['Missing Load Scan'].sum())
total_zone = int(fmt['Zone Mismatch'].sum())
total_all = int(fmt['Total Errors'].sum())

# ── 7. Write Word brief ────────────────────────────────────────
doc = Document()
doc.add_heading('Outbound Load Audit – Executive Brief', level=1)

para = doc.add_paragraph()
para.add_run(
    f'This audit examined carton handoff accuracy at the outbound dock. '
    f'A "Missing Load Scan" error means a carton listed in the manifest was never '
    f'recorded as loaded at the dock (no LOADED scan exists). '
    f'A "Zone Mismatch" error means a carton was scanned as loaded but in a '
    f'different zone than originally planned. '
    f'Across the audited dataset, there were {total_missing} Missing Load Scans, '
    f'{total_zone} Zone Mismatches, and {total_all} Total Errors. '
    f'Shipments {top2[0]} and {top2[1]} showed the highest concentration of '
    f'exceptions and should be prioritized for root-cause investigation. '
    f'We recommend implementing a real-time scan-to-plan validation alert at '
    f'each dock door so loaders are notified immediately when a carton is '
    f'directed to the wrong zone or has not been scanned before trailer departure.'
)

doc.save('Outbound_Load_Brief.docx')
print('Word brief saved.')
print(f'Totals => Missing: {total_missing}, Zone Mismatch: {total_zone}, Total: {total_all}')
print(f'Top shipments: {top2}')
PYEOF
```

After the script completes, verify:
1. `ls -la /root/Outbound_Load_Audit.xlsx /root/Outbound_Load_Brief.docx` — both files exist.
2. Open the Excel file with Python and confirm sheets `Overview`, `RawData`, `Formatted Data`, `Summary` all exist.
3. Confirm `Formatted Data` has 12 columns with the exact headers specified.
4. Confirm `Summary` last row has Route='Grand Total'.
5. Confirm the Word doc contains the required content (totals, two shipment IDs, recommendation).

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