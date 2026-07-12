# Task Instruction

Execute the following steps in order to produce `/root/Outbound_Load_Audit.xlsx` and `/root/Outbound_Load_Brief.docx`.

## Step 0 – Inspect source files
1. Read `/root/Manifest_Plan.xlsx` – note sheet names, column headers, row count, and sample rows.
2. Read `/root/Dock_Scan_Log.xlsx` – note sheet names, column headers (especially `Shipment ID`, `Carton ID`, `Status`, `Scanned Zone`, and any timestamp/sequence column), row count, and sample rows.
3. Read `/root/Outbound_Audit_Template.xlsx` – note every sheet name (expect at least `Overview`), and for the `Overview` sheet record its full content so it can be preserved byte-for-byte.
4. Print all findings so you can reference them in later steps.

## Step 1 – Write a single Python script `/root/build_audit.py`

Use `openpyxl` for Excel and `python-docx` for Word. The script must:

### 1a) Load data
- Load `Manifest_Plan.xlsx` into a list-of-dicts (preserve original column order and all rows). Call this `manifest_rows`.
- Load `Dock_Scan_Log.xlsx` into a list-of-dicts. Call this `scan_rows`.

### 1b) Derive the kept scans
- Filter `scan_rows` to only rows where `Status` == `LOADED`.
- Among those, for each unique `(Shipment ID, Carton ID)` keep only the latest row. "Latest" means the row with the highest index (last occurrence) if there is no explicit timestamp column, or the row with the maximum timestamp if one exists. Store the kept scans in a dict keyed by `(Shipment ID, Carton ID)` → row dict.

### 1c) Build Formatted Data rows
For each manifest row (same order), compute:
- `Missing Load Scan`: 1 if `(Shipment ID, Carton ID)` not in kept-scans dict, else 0.
- `Zone Mismatch`: 1 if it IS in kept-scans dict AND kept scan's `Scanned Zone` != manifest row's `Planned Zone`, else 0.
- `Total Errors`: sum of the two above.
- `Error Summary`: exactly one of `None`, `Missing Load Scan`, `Zone Mismatch`, `Missing Load Scan, Zone Mismatch` (use comma-space separator, and the string `None` when no errors).

### 1d) Build Summary rows
- Group Formatted Data by `(Route, Shipment ID)`.
- For each group sum `Missing Load Scan`, `Zone Mismatch`, `Total Errors`.
- Keep only groups where `Total Errors > 0`.
- Sort by Route ascending then Shipment ID ascending.
- Append a Grand Total row: Route=`Grand Total`, Shipment ID=`-`, sums of the three numeric columns.

### 1e) Write the Excel workbook
- Open `/root/Outbound_Audit_Template.xlsx` with `openpyxl` (with `data_only=False` to preserve formulas/formatting in Overview).
- Do NOT modify the `Overview` sheet at all.
- Create sheet `RawData`. Write the manifest table exactly (headers in row 1, data starting row 2). Preserve every column and value from `Manifest_Plan.xlsx`.
- Create sheet `Formatted Data`. Headers in row 1 must be exactly: `Shipment ID`, `Carton ID`, `Planned Zone`, `Route`, `Expected Weight`, `Hazmat Flag`, `Carrier`, `Wave`, `Missing Load Scan`, `Zone Mismatch`, `Total Errors`, `Error Summary`. Data rows follow in the same order as RawData. Write concrete values (int 0/1, strings), not formulas.
- Create sheet `Summary`. Headers: `Route`, `Shipment ID`, `Missing Load Scans`, `Zone Mismatches`, `Total Errors`. Then the sorted group rows, then the Grand Total row.
- Save as `/root/Outbound_Load_Audit.xlsx`.

### 1f) Write the Word document
- Create `/root/Outbound_Load_Brief.docx`.
- Add a heading "Outbound Load Audit – Executive Brief".
- Write 3-6 sentences that include ALL of the following:
  - A plain-language definition of `Missing Load Scan` (a carton in the manifest that was never scanned as LOADED at the dock).
  - A plain-language definition of `Zone Mismatch` (a carton that was scanned as LOADED but in a different zone than planned).
  - The exact computed totals: "X Missing Load Scans, Y Zone Mismatches, and Z Total Errors" (use the Grand Total numbers).
  - Identify at least two specific Shipment IDs with the highest error counts and name them explicitly (e.g., "High-priority shipments include SHP-XXX and SHP-YYY which accounted for …"). Pick the top 2 (or more) Shipment IDs by Total Errors from the Summary table.
  - At least one actionable recommendation (e.g., implement real-time zone validation at scan stations).
- Save the document.

### 1g) Verification prints
After saving both files, print:
- The list of sheet names in the saved Excel file.
- The first 3 and last 3 rows of `Formatted Data`.
- The full `Summary` table.
- The full text content of the Word document.

## Step 2 – Run the script
```bash
cd /root && python build_audit.py
```
Review the output. If any errors occur, fix and re-run.

## Step 3 – Validate
- Reopen `/root/Outbound_Load_Audit.xlsx` with openpyxl and confirm:
  - `Overview` sheet exists and is unmodified (compare content with template).
  - `RawData` row count matches manifest.
  - `Formatted Data` has 12 columns with correct headers.
  - `Summary` last row is Grand Total with correct sums.
- Reopen `/root/Outbound_Load_Brief.docx` and confirm it mentions at least two shipment IDs, both check definitions, numeric totals, and a recommendation.
- Print confirmation of all checks.

If any check fails, diagnose and fix before finishing.

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