# Task Instruction

## Task: Complete the Harbor Outbound Manifest Audit

You must produce two deliverables:
1. `/root/Outbound_Load_Audit.xlsx`
2. `/root/Outbound_Load_Brief.docx`

### Step-by-step Plan

#### Step 0: Inspect all input files
- Read `/root/Manifest_Plan.xlsx` — note all sheet names, column headers, row count, and data types. Print the first 5 and last 5 rows.
- Read `/root/Dock_Scan_Log.xlsx` — note all sheet names, column headers, row count. Print the first 5 and last 5 rows. Pay special attention to the `Status` column values and the `Scanned Zone` column name (exact spelling/casing).
- Read `/root/Outbound_Audit_Template.xlsx` — list every sheet name. For the `Overview` sheet, capture its full content so you can preserve it exactly. Check whether `RawData`, `Formatted Data`, and `Summary` sheets already exist (they may be blank or absent).

#### Step 1: Build the workbook using openpyxl
Use `openpyxl` to load the template and save as `/root/Outbound_Load_Audit.xlsx`. Do NOT use pandas ExcelWriter with engine that would drop existing sheets. Instead:

1. **Load** `/root/Outbound_Audit_Template.xlsx` with `openpyxl.load_workbook()`.
2. **Verify** the `Overview` sheet is present and untouched (do not modify it at all).
3. **Create or get** sheets named exactly `RawData`, `Formatted Data`, `Summary`. If they already exist in the template, clear their contents but keep the sheet. If they don't exist, create them.

#### Step 2: Populate `RawData`
- Copy the entire manifest plan table from `Manifest_Plan.xlsx` into the `RawData` sheet. Include the header row and all data rows. Preserve exact values (strings, numbers, etc.). Use the same column order as the source.

#### Step 3: Populate `Formatted Data`
- Use the same rows (same order) as `RawData`.
- The first 8 columns must have exactly these headers: `Shipment ID`, `Carton ID`, `Planned Zone`, `Route`, `Expected Weight`, `Hazmat Flag`, `Carrier`, `Wave`. Map from the manifest data columns to these names. If the manifest columns already match, use them directly. If names differ slightly, map carefully.
- **Derive scan status:**
  - From `Dock_Scan_Log.xlsx`, filter to rows where `Status == 'LOADED'`.
  - For each `(Shipment ID, Carton ID)` group among the LOADED rows, keep only the latest row. "Latest" means the row that appears last if there's a timestamp column, or the last occurrence in the file if no timestamp. Check if there's a timestamp or sequence column and use it; if not, use the last occurrence.
  - Build a lookup dictionary: key = `(Shipment ID, Carton ID)` → value = the kept LOADED scan row (specifically its `Scanned Zone`).
- **Compute columns 9-12 for each manifest row:**
  - `Missing Load Scan`: 1 if the `(Shipment ID, Carton ID)` has no entry in the lookup dict, else 0.
  - `Zone Mismatch`: 1 if there IS a kept LOADED scan AND its `Scanned Zone` != the row's `Planned Zone`, else 0. (If missing load scan, Zone Mismatch = 0.)
  - `Total Errors` = `Missing Load Scan` + `Zone Mismatch` (write as integer).
  - `Error Summary`: exactly one of these strings:
    - `"None"` if Total Errors == 0
    - `"Missing Load Scan"` if Missing Load Scan == 1 and Zone Mismatch == 0
    - `"Zone Mismatch"` if Missing Load Scan == 0 and Zone Mismatch == 1
    - `"Missing Load Scan, Zone Mismatch"` if both == 1
- **Write concrete values** (integers and strings), NOT Excel formulas.

#### Step 4: Populate `Summary`
- Headers (row 1): `Route`, `Shipment ID`, `Missing Load Scans`, `Zone Mismatches`, `Total Errors`
- From `Formatted Data`, group by `(Route, Shipment ID)`. For each group, sum `Missing Load Scan`, `Zone Mismatch`, `Total Errors`.
- **Include only groups where Total Errors > 0.**
- Sort by `Route` ascending (alphabetical/lexicographic), then `Shipment ID` ascending.
- After all data rows, append a Grand Total row:
  - Column 1: `Grand Total`
  - Column 2: `-`
  - Columns 3-5: sums across ALL included rows (i.e., dataset totals for Missing Load Scans, Zone Mismatches, Total Errors).

#### Step 5: Save the workbook
- Save to `/root/Outbound_Load_Audit.xlsx`.
- After saving, re-open and verify:
  - `Overview` sheet exists and its content matches the template exactly.
  - `RawData` has the correct number of rows (header + data).
  - `Formatted Data` has 12 columns with correct headers.
  - `Summary` has 5 columns, correct headers, data rows sorted properly, and a Grand Total row at the end.
  - Print the Summary sheet contents to confirm.

#### Step 6: Create `/root/Outbound_Load_Brief.docx`
Use `python-docx` to create the Word document.
- Write a short executive summary paragraph (3-6 sentences) that includes:
  1. A plain-language definition of both checks: explain what `Missing Load Scan` means (a carton in the manifest was never scanned as loaded at the dock) and what `Zone Mismatch` means (a carton was scanned as loaded but in a different zone than planned).
  2. The computed totals: state the exact numbers for Missing Load Scans, Zone Mismatches, and Total Errors (use the Grand Total row values).
  3. At least one actionable recommendation (e.g., implement real-time zone validation alerts, retrain dock staff on zone assignment protocols).
  4. Mention at least two specific high-priority Shipment IDs that had the most exceptions (identify the top 2 Shipment IDs by Total Errors from the Summary data).
- Save to `/root/Outbound_Load_Brief.docx`.

#### Step 7: Final Validation
- Re-open `/root/Outbound_Load_Audit.xlsx` and print:
  - Sheet names (must include `Overview`, `RawData`, `Formatted Data`, `Summary`)
  - First 3 and last 3 rows of `Formatted Data` (verify columns 9-12 have concrete values)
  - All rows of `Summary` (verify sort order, Grand Total row)
- Re-open `/root/Outbound_Load_Brief.docx` and print its full text to verify all required elements are present.

### Critical Constraints
- Do NOT modify the `Overview` sheet in any way.
- Sheet names must be exactly: `RawData`, `Formatted Data`, `Summary` (case-sensitive, exact spacing).
- Output filenames must be exactly `/root/Outbound_Load_Audit.xlsx` and `/root/Outbound_Load_Brief.docx`.
- All derived columns (9-12) in `Formatted Data` must contain concrete values (integers and strings), not formulas.
- The `Error Summary` strings must match exactly: `None`, `Missing Load Scan`, `Zone Mismatch`, or `Missing Load Scan, Zone Mismatch`.
- Install any needed Python packages (openpyxl, python-docx) if not already available.

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