# Task Instruction

## Task: Outbound Load Audit for Harbor Distribution Center

You must produce two deliverables:
1. `/root/Outbound_Load_Audit.xlsx`
2. `/root/Outbound_Load_Brief.docx`

### Step-by-step Instructions

#### Step 0: Inspect all input files
- Read `/root/Manifest_Plan.xlsx` and print its sheet names, column headers, and first 5 rows.
- Read `/root/Dock_Scan_Log.xlsx` and print its sheet names, column headers, and first 5 rows.
- Read `/root/Outbound_Audit_Template.xlsx` and print ALL sheet names. For the `Overview` sheet, print its full contents (all rows/columns) so you can preserve it exactly. For any other sheets, print their headers.

Do NOT proceed until you have inspected all three files and understand their structure.

#### Step 1: Build the workbook using openpyxl (to preserve the Overview sheet)

Use a Python script with `openpyxl` and `pandas`. The approach:
1. Copy `/root/Outbound_Audit_Template.xlsx` to `/root/Outbound_Load_Audit.xlsx` first.
2. Open the copy with openpyxl, keeping existing sheets intact (especially `Overview`).
3. Create/populate the three new worksheets: `RawData`, `Formatted Data`, `Summary`.

#### Step 2: RawData worksheet
- Read the manifest plan table from `Manifest_Plan.xlsx` into a DataFrame.
- Write it verbatim (headers + all rows) into a new sheet called `RawData`.
- Preserve exact column order and all values.

#### Step 3: Formatted Data worksheet
- Start with the same DataFrame from the manifest plan (same row order).
- Keep the first 8 columns exactly as: `Shipment ID`, `Carton ID`, `Planned Zone`, `Route`, `Expected Weight`, `Hazmat Flag`, `Carrier`, `Wave`.
  - IMPORTANT: If the manifest columns have slightly different names, map them carefully. Print the exact column names from the manifest to verify.
- Read `Dock_Scan_Log.xlsx` into a DataFrame.
- For deriving scan status:
  1. Filter `Dock_Scan_Log` to only rows where `Status == 'LOADED'`.
  2. Among those, for each `(Shipment ID, Carton ID)` group, keep only the LATEST row. Determine "latest" by the row's timestamp/datetime column if one exists, or by row order (last occurrence) if no timestamp. Print the scan log columns to identify the right sorting column.
  3. This gives you a lookup table of kept LOADED scans.
- For each row in the manifest:
  - `Missing Load Scan`: 1 if no kept LOADED scan exists for that (Shipment ID, Carton ID), else 0.
  - `Zone Mismatch`: 1 if a kept LOADED scan exists AND `Scanned Zone` (from the scan log) != `Planned Zone` (from the manifest), else 0. If no LOADED scan exists, this is 0.
  - `Total Errors` = `Missing Load Scan` + `Zone Mismatch`.
  - `Error Summary`: exactly one of:
    - `None` (if Total Errors == 0)
    - `Missing Load Scan` (if only missing scan)
    - `Zone Mismatch` (if only zone mismatch)
    - `Missing Load Scan, Zone Mismatch` (if both)
- Write concrete numeric/text values (NOT formulas) into columns 9-12.
- The headers for columns 9-12 must be exactly: `Missing Load Scan`, `Zone Mismatch`, `Total Errors`, `Error Summary`.

#### Step 4: Summary worksheet
- From the Formatted Data, group by `(Route, Shipment ID)`.
- For each group, sum `Missing Load Scan`, `Zone Mismatch`, `Total Errors`.
- Keep only groups where `Total Errors > 0`.
- Sort by `Route` ascending, then `Shipment ID` ascending.
- Headers must be exactly: `Route`, `Shipment ID`, `Missing Load Scans`, `Zone Mismatches`, `Total Errors`.
- Append a final row: Route=`Grand Total`, Shipment ID=`-`, and the remaining 3 columns = dataset-wide totals (sum of all rows in this summary, which equals the sum across the entire Formatted Data since we only include error rows but the Grand Total should reflect the FULL dataset totals — actually, the Grand Total should be the sum of the summary rows which equals the full dataset totals since non-error groups contribute 0).
- Actually to be safe: compute Grand Total as the sum of Missing Load Scan, Zone Mismatch, and Total Errors across ALL rows in Formatted Data (not just the filtered summary rows). This ensures correctness.

#### Step 5: Verify the Overview sheet is preserved
- After writing all sheets, re-open the file and confirm the `Overview` sheet still has its original content. Print its contents to verify.
- Confirm all 4 sheets exist: `Overview`, `RawData`, `Formatted Data`, `Summary`.

#### Step 6: Create the Word document
Using `python-docx`, create `/root/Outbound_Load_Brief.docx` with:
- A title/heading.
- An executive summary of 3-6 sentences that includes:
  1. A plain-language definition of both checks: explain what `Missing Load Scan` means (a carton in the manifest was never scanned as loaded at the dock) and what `Zone Mismatch` means (a carton was scanned as loaded but in a different zone than planned).
  2. The computed totals: state the exact numbers for Missing Load Scans, Zone Mismatches, and Total Errors from your Grand Total row.
  3. At least one actionable recommendation (e.g., implement real-time zone validation alerts, retrain dock staff on scan procedures).
  4. Mention at least two specific high-priority Shipment IDs that have the most exceptions. To find these, look at your Summary data and pick the two Shipment IDs with the highest Total Errors.

#### Step 7: Final Validation
- Re-read `/root/Outbound_Load_Audit.xlsx` and print:
  - Sheet names
  - `RawData`: row count and first 3 rows
  - `Formatted Data`: row count, first 3 rows, column headers (verify 12 columns)
  - `Summary`: all rows including Grand Total
  - `Overview`: confirm unchanged
- Confirm `/root/Outbound_Load_Brief.docx` exists and print its text content.
- Verify filenames are exactly as specified.

### Critical Notes
- Install any needed packages: `pip install openpyxl python-docx pandas` if not already available.
- Do NOT delete or modify the `Overview` sheet. The safest approach is to copy the template file first, then open it and add sheets.
- If the template already contains sheets named `RawData`, `Formatted Data`, or `Summary` (possibly empty), write into those existing sheets rather than creating duplicates.
- All worksheet names must match EXACTLY: `RawData`, `Formatted Data`, `Summary`, `Overview`.
- All output filenames must match EXACTLY: `/root/Outbound_Load_Audit.xlsx` and `/root/Outbound_Load_Brief.docx`.
- Write concrete values, not Excel formulas, in the error columns of Formatted Data.

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