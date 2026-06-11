# Task Instruction

Execute the following steps in a single Python script to produce `/root/Outbound_Load_Audit.xlsx` and `/root/Outbound_Load_Brief.docx`.

## Step 0 – Inspect inputs
```
import pandas as pd, openpyxl
for f in ['Manifest_Plan.xlsx','Dock_Scan_Log.xlsx','Outbound_Audit_Template.xlsx']:
    wb = openpyxl.load_workbook(f'/root/{f}')
    print(f, wb.sheetnames)
    for s in wb.sheetnames:
        ws = wb[s]
        for r in ws.iter_rows(min_row=1, max_row=min(5, ws.max_row), values_only=True):
            print(r)
    wb.close()
```
Run this first so you know the exact column names in each file. Use the printed column names verbatim in all subsequent pandas operations.

## Step 1 – Read data
```python
import pandas as pd
manifest = pd.read_excel('/root/Manifest_Plan.xlsx')
scans = pd.read_excel('/root/Dock_Scan_Log.xlsx')
```
Print `manifest.columns.tolist()` and `scans.columns.tolist()` and `manifest.head()` and `scans.head()` to confirm column names before proceeding.

## Step 2 – Prepare scan lookup
From `scans`, keep only rows where the Status column equals `LOADED` (use exact column name from Step 0). Then sort by the timestamp/sequence column (if any) ascending and keep the **last** row per `(Shipment ID, Carton ID)` group — this is the "latest LOADED scan."
```python
loaded = scans[scans['Status'] == 'LOADED'].copy()
# If there is a timestamp or scan-sequence column, sort by it ascending first
loaded = loaded.sort_values(by=[<timestamp_col>])  # adjust col name
kept = loaded.drop_duplicates(subset=['Shipment ID','Carton ID'], keep='last')
```
Create a set of kept `(Shipment ID, Carton ID)` tuples and a dict mapping each tuple to its `Scanned Zone`.

## Step 3 – Build Formatted Data
Start from a copy of `manifest`. Rename columns to exactly:
`Shipment ID, Carton ID, Planned Zone, Route, Expected Weight, Hazmat Flag, Carrier, Wave`
(map from the actual manifest column names discovered in Step 0; keep the original row order).

Add four new columns (use concrete values, not formulas):
- `Missing Load Scan`: 1 if the `(Shipment ID, Carton ID)` has no kept LOADED scan, else 0.
- `Zone Mismatch`: 1 if a kept LOADED scan exists AND its `Scanned Zone` != `Planned Zone`, else 0.
- `Total Errors`: sum of the two above.
- `Error Summary`: exactly one of `None`, `Missing Load Scan`, `Zone Mismatch`, `Missing Load Scan, Zone Mismatch`.

Ensure `Missing Load Scan` and `Zone Mismatch` are Python ints (0 or 1), `Total Errors` is int, and `Error Summary` is a plain string.

## Step 4 – Build Summary
From Formatted Data, group by `(Route, Shipment ID)` and sum `Missing Load Scan`, `Zone Mismatch`, `Total Errors`. Filter to groups where `Total Errors > 0`. Sort by Route ascending then Shipment ID ascending.

Append a Grand Total row: Route=`Grand Total`, Shipment ID=`-`, and the three numeric columns = dataset-wide sums from Formatted Data (not just the filtered groups — use the full Formatted Data totals).

Rename columns to exactly: `Route, Shipment ID, Missing Load Scans, Zone Mismatches, Total Errors`.

## Step 5 – Write Excel
```python
from openpyxl import load_workbook
from openpyxl.utils.dataframe import dataframe_to_rows

wb = load_workbook('/root/Outbound_Audit_Template.xlsx')

# RawData sheet
ws_raw = wb.create_sheet('RawData')
for r in dataframe_to_rows(manifest, index=False, header=True):
    ws_raw.append(r)

# Formatted Data sheet
ws_fmt = wb.create_sheet('Formatted Data')
for r in dataframe_to_rows(formatted_df, index=False, header=True):
    ws_fmt.append(r)

# Summary sheet
ws_sum = wb.create_sheet('Summary')
for r in dataframe_to_rows(summary_df, index=False, header=True):
    ws_sum.append(r)

wb.save('/root/Outbound_Load_Audit.xlsx')
```
Do NOT modify the `Overview` sheet in any way.

After saving, re-open the file and print sheet names and first few rows of each sheet to verify correctness.

## Step 6 – Write Word Brief
```python
from docx import Document
doc = Document()
```
Write 3-6 sentences that include:
1. Plain-language definition of `Missing Load Scan` (a carton in the manifest that was never scanned as loaded at the dock).
2. Plain-language definition of `Zone Mismatch` (a carton that was scanned as loaded but in a different zone than planned).
3. The exact computed totals for Missing Load Scans, Zone Mismatches, and Total Errors (use the Grand Total row values).
4. At least one actionable recommendation (e.g., implement secondary scan verification, zone-assignment review).
5. Name at least two specific Shipment IDs with the highest error counts from the Summary table.

Save as `/root/Outbound_Load_Brief.docx`.

## Step 7 – Final Validation
Re-open both output files and print:
- Excel: sheet names, row counts per sheet, first 3 and last 3 rows of Formatted Data and Summary.
- Word: full paragraph text.
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
Task metadata: author_email=catpaw@meituan.com, author_name=CatPaw Benchmark Builder, category=spreadsheet-audit, difficulty=hard, tags=[excel, openpyxl, docx, audit, logistics].
Verifier config: timeout_sec=900.0.