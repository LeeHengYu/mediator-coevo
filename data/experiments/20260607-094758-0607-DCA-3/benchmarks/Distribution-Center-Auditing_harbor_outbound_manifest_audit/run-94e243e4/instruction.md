# Task Instruction

Complete the outbound load audit task by creating two files: `/root/Outbound_Load_Audit.xlsx` and `/root/Outbound_Load_Brief.docx`.

## Step 1: Inspect Inputs First
Before writing code, read and inspect all three source files:
- `/root/Manifest_Plan.xlsx` — note column names, dtypes, and any string literals like 'N/A', 'NA', '-', or blanks.
- `/root/Dock_Scan_Log.xlsx` — note columns including Shipment ID, Carton ID, Status, Scanned Zone, and any timestamp column used to pick the latest row.
- `/root/Outbound_Audit_Template.xlsx` — list all worksheets and confirm the `Overview` sheet's exact contents (cells, formatting if visible).

## Step 2: Build `/root/Outbound_Load_Audit.xlsx`

Start by copying the template file to the output path so `Overview` is preserved byte-identically. Then add/overwrite the three required sheets using openpyxl (not pandas-to-excel for the whole workbook), to avoid disturbing `Overview`.

### RawData sheet
- Read `Manifest_Plan.xlsx` preserving literal strings. Use `pd.read_excel(..., dtype=str, keep_default_na=False, na_values=[])` (or equivalent openpyxl iteration) so that values like `'N/A'`, `'NA'`, `'-'`, empty strings stay exactly as in the source. Do NOT let pandas convert `'N/A'` to NaN/None.
- Write the table to a fresh `RawData` sheet exactly as read: same column order, same header text, same row order, same cell values (including literal `'N/A'`).
- After writing, re-open the file and spot-check that any source `'N/A'` cells are still the string `'N/A'` (not blank, not `None`).

### Formatted Data sheet
- Same row order as RawData.
- Columns 1–8 exactly: Shipment ID, Carton ID, Planned Zone, Route, Expected Weight, Hazmat Flag, Carrier, Wave (copied from RawData, preserving literals).
- Columns 9–12 headers exactly: `Missing Load Scan`, `Zone Mismatch`, `Total Errors`, `Error Summary`.
- Compute scan status from `Dock_Scan_Log.xlsx`:
  - Filter to rows where `Status == 'LOADED'`.
  - For each `(Shipment ID, Carton ID)`, keep the latest row (use the scan timestamp column; if multiple candidates, pick the one that looks like a datetime — document which column you used).
  - Build a lookup keyed by `(Shipment ID, Carton ID)` → `Scanned Zone`.
- Per manifest row:
  - `Missing Load Scan` = 1 if no kept LOADED scan exists for that key, else 0.
  - `Zone Mismatch` = 1 if a kept scan exists AND `Scanned Zone != Planned Zone`, else 0. (If missing, Zone Mismatch must be 0.)
  - `Total Errors` = sum of the two.
  - `Error Summary`: exactly one of `None`, `Missing Load Scan`, `Zone Mismatch`, `Missing Load Scan, Zone Mismatch` based on which flags are 1.
- Write concrete numeric/text values (no formulas).

### Summary sheet
- Headers exactly: `Route`, `Shipment ID`, `Missing Load Scans`, `Zone Mismatches`, `Total Errors`.
- Aggregate Formatted Data by `(Route, Shipment ID)` summing the three error columns.
- Keep only groups where `Total Errors > 0`.
- Sort by `Route` ascending, then `Shipment ID` ascending.
- Append final row: `Route='Grand Total'`, `Shipment ID='-'`, then dataset totals (sum of the kept groups' columns — i.e., the totals of the rows shown).

### Overview sheet
- Must remain exactly as in the template. Verify after writing by comparing cell values to the original template.

## Step 3: Build `/root/Outbound_Load_Brief.docx`
Using python-docx, write a 3–6 sentence executive summary that includes:
- Plain-language definitions of `Missing Load Scan` (no LOADED dock scan was recorded for that carton) and `Zone Mismatch` (the carton's LOADED scan zone differs from the planned zone).
- The computed totals: total Missing Load Scans, total Zone Mismatches, total Total Errors (use dataset-wide sums from Formatted Data).
- At least one actionable recommendation (e.g., re-train dock staff on zone routing, audit the top offending shipments).
- At least two high-priority Shipment IDs with the most exceptions (pick the top 2 by Total Errors from the Summary).

## Step 4: Validation Before Finishing
1. Re-open `/root/Outbound_Load_Audit.xlsx` and confirm sheet names are exactly: `Overview`, `RawData`, `Formatted Data`, `Summary` (Overview unchanged).
2. Confirm RawData literal values match source (especially `'N/A'` strings — they must remain the string `'N/A'`, not blank or None).
3. Confirm Formatted Data has exactly 12 columns with the specified headers and that `Error Summary` values are from the allowed set.
4. Confirm Summary is filtered to `Total Errors > 0`, sorted correctly, and ends with a `Grand Total` row whose numeric columns equal column sums of the rows above.
5. Confirm the .docx exists, has the required content, and references ≥2 shipment IDs.

## Critical Reminders
- Preserve string literals like `'N/A'` exactly when copying to RawData and Formatted Data. Use `keep_default_na=False, na_values=[]` (or equivalent) when reading with pandas. Never let `'N/A'` become `None`/`NaN`/blank.
- Do not modify the `Overview` sheet.
- Write concrete values, not formulas.
- Use exact header text and sheet names.

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