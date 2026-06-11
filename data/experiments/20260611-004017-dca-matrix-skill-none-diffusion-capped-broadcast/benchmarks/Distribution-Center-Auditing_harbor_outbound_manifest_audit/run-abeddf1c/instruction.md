# Task Instruction

Execute the following Python script in a single step to produce both deliverables.

```python
import openpyxl
import pandas as pd
from docx import Document

# ── 1. Read source data ──────────────────────────────────────────────
manifest = pd.read_excel('/root/Manifest_Plan.xlsx')
dock_log = pd.read_excel('/root/Dock_Scan_Log.xlsx')

# ── 2. Prepare scan lookup ───────────────────────────────────────────
# Keep only LOADED rows, then keep the latest per (Shipment ID, Carton ID)
loaded = dock_log[dock_log['Status'] == 'LOADED'].copy()
# Sort so the last row per group is the latest scan
loaded = loaded.sort_values('Scan Timestamp')
latest_loaded = loaded.groupby(['Shipment ID', 'Carton ID']).last().reset_index()

# Build a dict keyed by (Shipment ID, Carton ID) -> row from latest_loaded
scan_lookup = {}
for _, row in latest_loaded.iterrows():
    scan_lookup[(row['Shipment ID'], row['Carton ID'])] = row

# ── 3. Build Formatted Data rows ────────────────────────────────────
first8_cols = ['Shipment ID', 'Carton ID', 'Planned Zone', 'Route',
               'Expected Weight', 'Hazmat Flag', 'Carrier', 'Wave']
new_cols = ['Missing Load Scan', 'Zone Mismatch', 'Total Errors', 'Error Summary']

formatted_rows = []
for _, m in manifest.iterrows():
    base = [m[c] for c in first8_cols]
    key = (m['Shipment ID'], m['Carton ID'])
    scan = scan_lookup.get(key)
    if scan is None:
        missing = 1
        zone_mm = 0  # no scan → cannot compare zones
    else:
        missing = 0
        zone_mm = 1 if str(scan['Scanned Zone']).strip() != str(m['Planned Zone']).strip() else 0
    total_err = missing + zone_mm
    if missing == 1 and zone_mm == 1:
        summary = 'Missing Load Scan, Zone Mismatch'
    elif missing == 1:
        summary = 'Missing Load Scan'
    elif zone_mm == 1:
        summary = 'Zone Mismatch'
    else:
        summary = 'None'
    formatted_rows.append(base + [missing, zone_mm, total_err, summary])

all_headers = first8_cols + new_cols

# ── 4. Build Summary rows ───────────────────────────────────────────
from collections import defaultdict
agg = defaultdict(lambda: [0, 0, 0])  # (Route, ShipID) -> [miss, zone, total]
for row in formatted_rows:
    route = row[3]  # Route
    sid = row[0]    # Shipment ID
    agg[(route, sid)][0] += row[8]   # Missing Load Scan
    agg[(route, sid)][1] += row[9]   # Zone Mismatch
    agg[(route, sid)][2] += row[10]  # Total Errors

summary_rows = []
for (route, sid), vals in agg.items():
    if vals[2] > 0:
        summary_rows.append([route, sid, vals[0], vals[1], vals[2]])

summary_rows.sort(key=lambda x: (str(x[0]), str(x[1])))

grand_miss = sum(r[2] for r in summary_rows)
grand_zone = sum(r[3] for r in summary_rows)
grand_total = sum(r[4] for r in summary_rows)
summary_rows.append(['Grand Total', '-', grand_miss, grand_zone, grand_total])

summary_headers = ['Route', 'Shipment ID', 'Missing Load Scans',
                   'Zone Mismatches', 'Total Errors']

# ── 5. Write Excel workbook from template ────────────────────────────
wb = openpyxl.load_workbook('/root/Outbound_Audit_Template.xlsx')

# -- RawData sheet --
if 'RawData' in wb.sheetnames:
    del wb['RawData']
ws_raw = wb.create_sheet('RawData')
raw_headers = list(manifest.columns)
ws_raw.append(raw_headers)
for _, row in manifest.iterrows():
    ws_raw.append([row[c] for c in raw_headers])

# -- Formatted Data sheet --
if 'Formatted Data' in wb.sheetnames:
    del wb['Formatted Data']
ws_fmt = wb.create_sheet('Formatted Data')
ws_fmt.append(all_headers)
for row in formatted_rows:
    ws_fmt.append(row)

# -- Summary sheet --
if 'Summary' in wb.sheetnames:
    del wb['Summary']
ws_sum = wb.create_sheet('Summary')
ws_sum.append(summary_headers)
for row in summary_rows:
    ws_sum.append(row)

wb.save('/root/Outbound_Load_Audit.xlsx')
print('Excel saved.')

# ── 6. Identify high-priority shipment IDs ───────────────────────────
# Pick top-2 shipment IDs by total errors (excluding Grand Total row)
error_by_sid = defaultdict(int)
for row in summary_rows[:-1]:  # skip Grand Total
    error_by_sid[row[1]] += row[4]
top_sids = sorted(error_by_sid.items(), key=lambda x: -x[1])[:2]
top_sid_names = [s[0] for s in top_sids]

# ── 7. Write Word brief ─────────────────────────────────────────────
doc = Document()
doc.add_heading('Outbound Load Audit – Executive Brief', level=1)

para = (
    f'This audit evaluated carton handoff accuracy across all planned shipments. '
    f'A "Missing Load Scan" indicates that a carton listed in the manifest was never '
    f'recorded with a LOADED status at the dock scanner, suggesting it may not have '
    f'been physically loaded onto the trailer. '
    f'A "Zone Mismatch" indicates that a carton was scanned as loaded but in a dock '
    f'zone different from the planned zone, pointing to a potential mis-sort or staging error. '
    f'Across the dataset, the audit identified {grand_miss} Missing Load Scans, '
    f'{grand_zone} Zone Mismatches, and {grand_total} Total Errors. '
    f'Shipment IDs {top_sid_names[0]} and {top_sid_names[1]} exhibited the highest '
    f'frequency of exceptions and should be prioritized for root-cause investigation. '
    f'We recommend implementing a secondary barcode verification step at the dock door '
    f'to reduce missing scans and zone routing errors before trailer departure.'
)
doc.add_paragraph(para)
doc.save('/root/Outbound_Load_Brief.docx')
print('Word brief saved.')
print(f'Grand totals — Missing: {grand_miss}, Zone Mismatch: {grand_zone}, Total: {grand_total}')
print(f'Top shipment IDs: {top_sid_names}')
```

After execution, verify:
1. `/root/Outbound_Load_Audit.xlsx` exists and contains sheets: Overview, RawData, Formatted Data, Summary.
2. `/root/Outbound_Load_Brief.docx` exists.
3. Print the first few rows of each sheet to confirm data integrity.

```python
import openpyxl
wb = openpyxl.load_workbook('/root/Outbound_Load_Audit.xlsx')
print('Sheets:', wb.sheetnames)
for sname in ['Overview', 'RawData', 'Formatted Data', 'Summary']:
    ws = wb[sname]
    print(f'\n--- {sname} ({ws.max_row} rows, {ws.max_column} cols) ---')
    for i, row in enumerate(ws.iter_rows(values_only=True)):
        if i < 5:
            print(row)
        else:
            break
print('\nVerification complete.')
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
Task metadata: author_email=catpaw@meituan.com, author_name=CatPaw Benchmark Builder, category=spreadsheet-audit, difficulty=hard, tags=[excel, openpyxl, docx, audit, logistics].
Verifier config: timeout_sec=900.0.