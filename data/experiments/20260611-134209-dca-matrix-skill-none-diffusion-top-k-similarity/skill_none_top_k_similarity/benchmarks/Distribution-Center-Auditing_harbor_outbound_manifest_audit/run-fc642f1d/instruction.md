# Task Instruction

## Task: Outbound Load Audit for Harbor Distribution Center

You must produce two deliverable files:
1. `/root/Outbound_Load_Audit.xlsx`
2. `/root/Outbound_Load_Brief.docx`

### Step-by-step Instructions

#### Step 0: Inspect all source files
- Read `/root/Manifest_Plan.xlsx` — note all sheet names, column headers, row count, and data types.
- Read `/root/Dock_Scan_Log.xlsx` — note all sheet names, column headers (especially `Shipment ID`, `Carton ID`, `Status`, `Scanned Zone`, and any timestamp/sequence column), row count.
- Read `/root/Outbound_Audit_Template.xlsx` — note ALL sheet names (especially `Overview` and any others). Record the exact content of the `Overview` sheet so you can preserve it byte-for-byte.
- Print samples of each file (first 10 rows) so you understand the data.

#### Step 1: Build the workbook using openpyxl (or pandas + openpyxl)

Use Python. Install any needed packages (`pip install openpyxl python-docx pandas` if not already available).

**Load the template:**
- Open `/root/Outbound_Audit_Template.xlsx` with openpyxl (keeping styles if possible). This preserves the `Overview` sheet.
- Do NOT modify the `Overview` sheet in any way.

**Create `RawData` sheet:**
- Copy the entire manifest plan table from `Manifest_Plan.xlsx` exactly (all columns, all rows, same order, same values).
- The first row should be the headers, subsequent rows the data.

**Create `Formatted Data` sheet:**
- Start with the same rows and order as `RawData`.
- Keep the first 8 columns with EXACTLY these headers: `Shipment ID`, `Carton ID`, `Planned Zone`, `Route`, `Expected Weight`, `Hazmat Flag`, `Carrier`, `Wave`.
  - If the source columns have different names, map them correctly based on content.
- Process `Dock_Scan_Log.xlsx`:
  - Filter to rows where `Status == 'LOADED'`.
  - For each `(Shipment ID, Carton ID)` group, keep only the LATEST row (use timestamp, sequence number, or row order — inspect the data to determine which column indicates recency; if there's a timestamp column use that, otherwise use the last occurrence by row position).
  - This gives you the "kept scan" lookup table.
- For each manifest row, compute:
  - `Missing Load Scan`: 1 if no kept LOADED scan exists for that (Shipment ID, Carton ID), else 0.
  - `Zone Mismatch`: 1 if a kept LOADED scan exists AND its `Scanned Zone` != the row's `Planned Zone`, else 0. (If Missing Load Scan=1, Zone Mismatch must be 0.)
  - `Total Errors`: Missing Load Scan + Zone Mismatch.
  - `Error Summary`: exactly one of: `None`, `Missing Load Scan`, `Zone Mismatch`, `Missing Load Scan, Zone Mismatch`.
- Write all values as concrete literals (strings/integers), NOT Excel formulas.

**Create `Summary` sheet:**
- Headers (row 1): `Route`, `Shipment ID`, `Missing Load Scans`, `Zone Mismatches`, `Total Errors`.
- Aggregate from `Formatted Data` by (Route, Shipment ID): sum Missing Load Scan → Missing Load Scans, sum Zone Mismatch → Zone Mismatches, sum Total Errors → Total Errors.
- Include ONLY groups where Total Errors > 0.
- Sort by Route ascending, then Shipment ID ascending.
- Append a final row: Route=`Grand Total`, Shipment ID=`-`, and the remaining columns are the dataset-wide totals (sum of all Missing Load Scans, all Zone Mismatches, all Total Errors across the entire Formatted Data sheet, not just the filtered groups — but since groups with 0 errors contribute 0, summing the filtered groups gives the same result).

**Save** the workbook as `/root/Outbound_Load_Audit.xlsx`.

#### Step 2: Verify the Excel output
- Re-open `/root/Outbound_Load_Audit.xlsx` and confirm:
  - Sheet names include `Overview`, `RawData`, `Formatted Data`, `Summary` (print all sheet names).
  - `Overview` sheet content matches the template exactly (compare first few cells).
  - `RawData` row count matches Manifest_Plan row count.
  - `Formatted Data` has 12 columns with correct headers. Print first 5 and last 5 rows.
  - `Summary` has 5 columns with correct headers. Print all rows. Verify the Grand Total row is last.
  - Spot-check 2-3 specific (Shipment ID, Carton ID) pairs manually against the scan log to verify Missing Load Scan and Zone Mismatch are correct.

#### Step 3: Create the Word document
Using `python-docx`, create `/root/Outbound_Load_Brief.docx` with:
- A title/heading: "Outbound Load Audit Brief" or similar.
- An executive summary paragraph (3-6 sentences) that includes:
  1. Plain-language definition of `Missing Load Scan` (a carton in the manifest that has no confirmed LOADED scan in the dock log).
  2. Plain-language definition of `Zone Mismatch` (a carton whose scanned dock zone differs from the planned zone).
  3. The computed Grand Total numbers: X missing load scans, Y zone mismatches, Z total errors (use the actual numbers from your Summary Grand Total row).
  4. At least one actionable recommendation (e.g., retraining dock staff, adding zone verification scanners, investigating specific routes).
  5. Mention at least two specific high-priority Shipment IDs that had the most errors (look at the Summary sheet to identify them).

#### Step 4: Final verification
- Confirm both files exist: `ls -la /root/Outbound_Load_Audit.xlsx /root/Outbound_Load_Brief.docx`
- Re-read the Word doc and print its text to verify all required elements are present.
- Re-read the Excel file one more time and print the Summary sheet completely to confirm correctness.

### Critical Constraints
- Do NOT alter the `Overview` worksheet from the template.
- Sheet names must be EXACTLY: `Overview`, `RawData`, `Formatted Data`, `Summary`.
- Output filenames must be EXACTLY as specified.
- Column headers must match EXACTLY as specified (case-sensitive, spacing matters).
- Error Summary values must be EXACTLY one of the four specified strings.
- All computed columns must be literal values, not formulas.
- The Grand Total row in Summary must use the string `Grand Total` in the Route column and `-` in the Shipment ID column.

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