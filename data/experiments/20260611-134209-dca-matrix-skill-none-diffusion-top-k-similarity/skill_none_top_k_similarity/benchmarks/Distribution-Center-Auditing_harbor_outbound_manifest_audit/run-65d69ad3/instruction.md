# Task Instruction

Execute the following steps in order to produce `/root/Outbound_Load_Audit.xlsx` and `/root/Outbound_Load_Brief.docx`.

## Step 0 – Inspect inputs
1. Read `/root/Manifest_Plan.xlsx` and print its sheet names, column headers, and first 5 rows.
2. Read `/root/Dock_Scan_Log.xlsx` and print its sheet names, column headers, and first 5 rows. Pay special attention to columns: `Shipment ID`, `Carton ID`, `Status`, `Scanned Zone`, and any timestamp/sequence column that can determine recency.
3. Open `/root/Outbound_Audit_Template.xlsx`, list all sheet names, and print the contents of the `Overview` sheet so you know exactly what must be preserved.

## Step 1 – Build the workbook (Python, openpyxl + pandas)
Write and run a single Python script that does everything below.

### 1-A  Load data
```python
import pandas as pd
from copy import copy
from openpyxl import load_workbook
from openpyxl.utils.dataframe import dataframe_to_rows

manifest = pd.read_excel('/root/Manifest_Plan.xlsx')
scan_log = pd.read_excel('/root/Dock_Scan_Log.xlsx')
```

### 1-B  Preserve the template
- Load the template workbook with `load_workbook('/root/Outbound_Audit_Template.xlsx')`.
- Do **not** modify the `Overview` sheet in any way (no cell edits, no style changes).

### 1-C  `RawData` sheet
- Create (or get) a sheet named exactly `RawData`.
- Write the manifest DataFrame verbatim (same columns, same row order, same values).

### 1-D  `Formatted Data` sheet
- Start with the manifest DataFrame.
- Ensure the first 8 columns are exactly: `Shipment ID`, `Carton ID`, `Planned Zone`, `Route`, `Expected Weight`, `Hazmat Flag`, `Carrier`, `Wave`. Rename if the manifest uses different names; inspect first.
- From the scan log, keep only rows where `Status == 'LOADED'`. Among those, for each `(Shipment ID, Carton ID)` group keep only the **latest** row (use whatever timestamp or row-index column indicates recency; if there is a datetime column sort by it descending, otherwise sort by the original row index descending and take the first).
- Merge (left join) manifest onto the kept scans on `(Shipment ID, Carton ID)`.
- Compute four new columns **as concrete int/str values** (no Excel formulas):
  - `Missing Load Scan` = 1 if no kept LOADED scan exists for that pair, else 0.
  - `Zone Mismatch` = 1 if a kept LOADED scan exists AND `Scanned Zone != Planned Zone`, else 0.
  - `Total Errors` = `Missing Load Scan + Zone Mismatch`.
  - `Error Summary`: exactly one of `"None"`, `"Missing Load Scan"`, `"Zone Mismatch"`, `"Missing Load Scan, Zone Mismatch"` (choose based on the two flags).
- Write to a sheet named exactly `Formatted Data`, preserving the original manifest row order.

### 1-E  `Summary` sheet
- From the Formatted Data, group by `(Route, Shipment ID)` and sum `Missing Load Scan`, `Zone Mismatch`, `Total Errors`.
- Keep only groups where `Total Errors > 0`.
- Sort by `Route` ascending then `Shipment ID` ascending.
- Append a final row: `Route`=`Grand Total`, `Shipment ID`=`-`, and the remaining three columns = dataset-wide totals.
- Headers exactly: `Route`, `Shipment ID`, `Missing Load Scans`, `Zone Mismatches`, `Total Errors`.
- Write to a sheet named exactly `Summary`.

### 1-F  Save
- Save the workbook as `/root/Outbound_Load_Audit.xlsx`.

## Step 2 – Build the Word brief (python-docx)
In the same or a follow-up script:
- Compute the grand totals for Missing Load Scans, Zone Mismatches, Total Errors.
- Identify at least two Shipment IDs with the highest Total Errors.
- Create `/root/Outbound_Load_Brief.docx` containing an executive summary paragraph (3–6 sentences) that:
  1. Defines **Missing Load Scan** in plain language (a carton in the manifest that was never scanned as LOADED at the dock).
  2. Defines **Zone Mismatch** in plain language (a carton scanned as LOADED but at a different zone than planned).
  3. States the computed totals (e.g., "The audit identified X missing load scans, Y zone mismatches, and Z total errors.").
  4. Names at least two high-priority shipment IDs.
  5. Gives at least one actionable recommendation (e.g., implement zone-gate barcode validation).

## Step 3 – Validate
1. Re-open `/root/Outbound_Load_Audit.xlsx` and confirm:
   - Sheet names include `Overview`, `RawData`, `Formatted Data`, `Summary`.
   - `Overview` content is identical to the template.
   - `RawData` row count matches manifest row count.
   - `Formatted Data` has 12 columns with the exact headers listed above.
   - `Summary` last row has Route == `Grand Total`.
2. Re-open `/root/Outbound_Load_Brief.docx` and print its text to confirm it contains the required elements.
3. Print "DONE – all validations passed" only if every check succeeds.

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