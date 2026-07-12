# Task Instruction

## Task: Outbound Manifest Audit

You must produce two deliverables:
1. `/root/Outbound_Load_Audit.xlsx`
2. `/root/Outbound_Load_Brief.docx`

### Step-by-step Instructions

#### Step 0: Inspect inputs
- Read and display the contents of `/root/Manifest_Plan.xlsx`, `/root/Dock_Scan_Log.xlsx`, and `/root/Outbound_Audit_Template.xlsx` (list sheet names, column headers, and a few sample rows for each).
- For the template, list all worksheet names and note the contents of the `Overview` sheet (this must be preserved exactly).

#### Step 1: Build the workbook using openpyxl

Use Python with `openpyxl` (and `pandas` for data manipulation). The strategy:

1. **Load the template** (`Outbound_Audit_Template.xlsx`) with `openpyxl.load_workbook` preserving formatting if possible.
2. **Read Manifest_Plan.xlsx** into a pandas DataFrame. Read **Dock_Scan_Log.xlsx** into a pandas DataFrame.
3. **Create `RawData` sheet**: Copy the manifest plan table exactly (all columns, all rows, same order) into a new worksheet named `RawData`. Write headers in row 1, data starting row 2.
4. **Create `Formatted Data` sheet**:
   - Start with the manifest data in the same row order as RawData.
   - Keep the first 8 columns exactly as: `Shipment ID`, `Carton ID`, `Planned Zone`, `Route`, `Expected Weight`, `Hazmat Flag`, `Carrier`, `Wave`.
   - From `Dock_Scan_Log.xlsx`, for each `(Shipment ID, Carton ID)` group, keep only the latest row (by timestamp or row order—check if there's a timestamp column; if not, use last occurrence) where `Status == 'LOADED'`. Call this the "kept scan" set.
   - Compute columns 9-12:
     - `Missing Load Scan`: 1 if no kept LOADED scan exists for that (Shipment ID, Carton ID), else 0.
     - `Zone Mismatch`: 1 if a kept LOADED scan exists AND its `Scanned Zone` != `Planned Zone`, else 0.
     - `Total Errors`: sum of the above two.
     - `Error Summary`: exactly one of `None`, `Missing Load Scan`, `Zone Mismatch`, or `Missing Load Scan, Zone Mismatch` based on which flags are 1.
   - Write all values as concrete numbers/strings (no formulas).
5. **Create `Summary` sheet**:
   - Headers: `Route`, `Shipment ID`, `Missing Load Scans`, `Zone Mismatches`, `Total Errors`.
   - Group `Formatted Data` by `(Route, Shipment ID)`, sum the error columns.
   - Include only groups where `Total Errors > 0`.
   - Sort by `Route` ascending, then `Shipment ID` ascending.
   - Append a final row: `Grand Total`, `-`, and dataset totals for the three numeric columns.
6. **Preserve `Overview` sheet**: Do NOT modify the existing `Overview` worksheet in any way. Verify after saving that it still exists and is unchanged.
7. **Save** as `/root/Outbound_Load_Audit.xlsx`.

#### Step 2: Verify the workbook
- Re-open `/root/Outbound_Load_Audit.xlsx` with openpyxl.
- Confirm sheet names include `Overview`, `RawData`, `Formatted Data`, `Summary`.
- Print the first few rows and last few rows of each data sheet.
- Print the `Overview` sheet contents to confirm it's unchanged.
- Print the `Summary` sheet completely to verify Grand Total row.
- Confirm `Formatted Data` column count is 12 and headers match exactly.

#### Step 3: Create the Word document
Use `python-docx` to create `/root/Outbound_Load_Brief.docx`.

Content requirements (3-6 sentences in a single paragraph or short section titled "Executive Summary"):
- Define both checks in plain language:
  - "Missing Load Scan" = a planned carton that was never scanned as loaded at the dock.
  - "Zone Mismatch" = a carton that was scanned as loaded but in a different zone than planned.
- State the computed totals: X missing load scans, Y zone mismatches, Z total errors (use actual numbers from your computation).
- Mention at least two specific high-priority Shipment IDs that have the most errors.
- Include at least one actionable recommendation (e.g., retraining dock staff, adding zone verification scanners, etc.).

Save as `/root/Outbound_Load_Brief.docx`.

#### Step 4: Final verification
- Confirm both files exist: `/root/Outbound_Load_Audit.xlsx` and `/root/Outbound_Load_Brief.docx`.
- Re-read the docx and print its text to verify it meets all content requirements.
- Re-read the xlsx and print sheet names to confirm.

### Important Notes
- When reading the Dock_Scan_Log, carefully inspect the column names (they may have different casing or spacing). Adapt accordingly but match to the logic described.
- If there is a timestamp/scan-time column in Dock_Scan_Log, use it to determine "latest". If not, use the last row occurrence.
- The `Error Summary` string must use exactly the specified phrases with exact punctuation (comma-space between the two error types when both apply).
- The `Formatted Data` headers for columns 9-12 must be exactly: `Missing Load Scan`, `Zone Mismatch`, `Total Errors`, `Error Summary`.
- The `Summary` headers must be exactly: `Route`, `Shipment ID`, `Missing Load Scans`, `Zone Mismatches`, `Total Errors`.
- Do NOT delete or rename the `Overview` sheet. If the template has other sheets, preserve them too, but the required sheets must exist.

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