# Task Instruction

## Task: Outbound Load Audit

You must produce two deliverables:
1. `/root/Outbound_Load_Audit.xlsx`
2. `/root/Outbound_Load_Brief.docx`

### Step-by-step Instructions

#### Step 0: Inspect all input files
- Read `/root/Manifest_Plan.xlsx` and print all sheet names, column headers, and the first 5+ rows.
- Read `/root/Dock_Scan_Log.xlsx` and print all sheet names, column headers, and the first 5+ rows. Pay close attention to the exact column names (e.g., is it `Scanned Zone` or `Zone`? `Status` or `Scan Status`?).
- Read `/root/Outbound_Audit_Template.xlsx` and list all sheet names. Print the contents of the `Overview` sheet and any other sheets. Note the exact sheet names present.

#### Step 1: Build the Excel workbook

Use Python with `openpyxl` (and `pandas` for data manipulation). The approach:

```python
import pandas as pd
from openpyxl import load_workbook
from openpyxl.utils.dataframe import dataframe_to_rows
from copy import copy

# Load input data
manifest = pd.read_excel('/root/Manifest_Plan.xlsx')
dock_log = pd.read_excel('/root/Dock_Scan_Log.xlsx')

# Load the template workbook (preserves Overview sheet exactly)
wb = load_workbook('/root/Outbound_Audit_Template.xlsx')
```

##### 1a) `RawData` sheet
- Create a sheet named `RawData` in the workbook.
- Copy the manifest plan table exactly (all columns, all rows, same order).
- Write headers in row 1, data starting row 2.

##### 1b) `Formatted Data` sheet
- Create a sheet named `Formatted Data`.
- Start with the manifest data in the same row order as RawData.
- The first 8 columns must be exactly: `Shipment ID`, `Carton ID`, `Planned Zone`, `Route`, `Expected Weight`, `Hazmat Flag`, `Carrier`, `Wave`.
  - IMPORTANT: Map from whatever column names exist in the manifest to these exact headers. Inspect the manifest columns carefully. If the manifest has columns with slightly different names (e.g., `Weight` vs `Expected Weight`), map them correctly.
- Filter `Dock_Scan_Log.xlsx`: keep only rows where the Status column equals `LOADED`. Then for each `(Shipment ID, Carton ID)` group, keep only the latest row (by timestamp/scan time column or by last occurrence if no timestamp). This is the "kept scan" for that pair.
- Derive columns 9-12:
  - `Missing Load Scan`: 1 if no kept LOADED scan exists for that (Shipment ID, Carton ID), else 0.
  - `Zone Mismatch`: 1 if a kept LOADED scan exists AND the scan's zone column != the manifest's `Planned Zone`, else 0.
  - `Total Errors`: sum of the two above.
  - `Error Summary`: exactly one of `None`, `Missing Load Scan`, `Zone Mismatch`, or `Missing Load Scan, Zone Mismatch`.
- Write concrete values (not formulas) for columns 9-12.

##### 1c) `Summary` sheet
- Create a sheet named `Summary`.
- Headers exactly: `Route`, `Shipment ID`, `Missing Load Scans`, `Zone Mismatches`, `Total Errors`.
- Aggregate from the Formatted Data by (Route, Shipment ID): sum Missing Load Scan -> Missing Load Scans, sum Zone Mismatch -> Zone Mismatches, sum Total Errors -> Total Errors.
- Include ONLY groups where Total Errors > 0.
- Sort by Route ascending, then Shipment ID ascending.
- Append a Grand Total row: Route=`Grand Total`, Shipment ID=`-`, and the remaining columns are the dataset-wide totals.

##### 1d) Preserve Overview
- The `Overview` sheet from the template must remain completely unchanged. Do NOT delete it, rename it, or modify any cells.
- If the template has other sheets, keep them as-is unless they conflict with the required sheet names.

##### 1e) Save
- Save as `/root/Outbound_Load_Audit.xlsx`.

#### Step 2: Build the Word document

Use `python-docx`:

```python
from docx import Document
doc = Document()
```

- Write a short executive summary (3-6 sentences) that:
  1. Defines `Missing Load Scan` in plain language (a carton in the manifest that was never scanned as loaded at the dock).
  2. Defines `Zone Mismatch` in plain language (a carton scanned as loaded but in a different zone than planned).
  3. States the computed totals: X missing load scans, Y zone mismatches, Z total errors.
  4. Mentions at least two specific high-priority Shipment IDs that had the most errors.
  5. Provides at least one actionable recommendation (e.g., implement real-time zone validation, retrain dock staff on zone assignments).
- Save as `/root/Outbound_Load_Brief.docx`.

#### Step 3: Validate

- Reopen `/root/Outbound_Load_Audit.xlsx` and verify:
  - Sheet names include `Overview`, `RawData`, `Formatted Data`, `Summary`.
  - `RawData` row count matches manifest row count.
  - `Formatted Data` has 12 columns with exact headers specified.
  - `Formatted Data` columns 9-12 contain concrete values (integers and strings, not formulas).
  - `Summary` has 5 columns with exact headers, rows are sorted correctly, Grand Total row is last.
  - `Overview` sheet content is unchanged from template.
- Reopen `/root/Outbound_Load_Brief.docx` and print its text to confirm it meets requirements.

### Critical Details
- When filtering dock scans for LOADED status, check the exact value in the Status column (it might be `LOADED`, `Loaded`, etc.). Match case-insensitively if needed, but prefer exact match after inspecting the data.
- When finding the "latest" LOADED scan per (Shipment ID, Carton ID), look for a timestamp or scan-time column. If there's no such column, use the last row in file order.
- The `Zone Mismatch` check: if there is no kept LOADED scan (i.e., Missing Load Scan = 1), then Zone Mismatch must be 0 (you can't have a zone mismatch without a scan).
- `Error Summary` when both Missing Load Scan=1 and Zone Mismatch=1 should be `Missing Load Scan, Zone Mismatch` — but logically this shouldn't happen (if missing, no zone to compare). Still, follow the rules as stated.
- Install any needed packages: `pip install openpyxl python-docx pandas` if not already available.

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