# Task Instruction

## Task: Cycle Count Variance Audit

You must produce two deliverable files:
1. `/root/Cycle_Count_Variance_Audit.xlsx`
2. `/root/Cycle_Count_Variance_Brief.docx`

### Step-by-step instructions:

#### Step 0: Inspect the input files
- Read `/root/Cycle_Plan.xlsx` — list all sheet names and print the first 10 rows of each sheet. Note the column names exactly.
- Read `/root/Count_Event_Log.xlsx` — list all sheet names and print the first 10 rows. Note column names exactly (especially `Event Type`, `Count Qty`, `Facility`, `Session ID`, `Bin ID`).
- Read `/root/Cycle_Template.xlsx` — list all sheet names. For the `Overview` sheet, print its full contents (it must be copied verbatim).

#### Step 1: Build the Excel workbook

Use Python with `openpyxl` (and `pandas` for data manipulation). Create `/root/Cycle_Count_Variance_Audit.xlsx` with exactly 4 sheets in this order: `Overview`, `RawData`, `Formatted Data`, `Summary`.

##### Sheet 1: `Overview`
- Copy the `Overview` sheet from `Cycle_Template.xlsx` exactly — preserve all cell values, merged cells, formatting if possible. Do NOT modify any content.
- IMPORTANT: Use openpyxl to read the template and copy cell-by-cell to preserve content faithfully. Copy all rows and columns that have data.

##### Sheet 2: `RawData`
- Copy the plan table from `Cycle_Plan.xlsx` exactly as-is (all columns, all rows, same order).

##### Sheet 3: `Formatted Data`
- Start with the same rows (same order) as `RawData`.
- Keep the first 7 columns with exactly these headers:
  1. `Facility`
  2. `Session ID`
  3. `Bin ID`
  4. `Product ID`
  5. `Expected Qty`
  6. `Allowed Variance`
  7. `Approval Needed`
- Map columns from the plan data to these headers. If the source column names differ slightly, map them correctly based on content.
- Add 4 new columns (8-11) with exactly these headers: `Missing Final Count`, `Approval Gap`, `Total Errors`, `Error Summary`.

**Deriving the final count lookup:**
1. Load `Count_Event_Log.xlsx` into a DataFrame.
2. Filter to rows where `Event Type` == `FINAL` (case-sensitive match on the value; check actual values in the data).
3. Drop rows where any of `Facility`, `Session ID`, `Bin ID`, or `Count Qty` is blank/NaN.
4. For each unique `(Facility, Session ID, Bin ID)` group, keep only the LAST row (latest row by position in the spreadsheet — i.e., the row with the highest index). This gives you the "kept FINAL event" lookup table.

**Computing the 4 new columns for each row in Formatted Data:**
- Look up the row's `(Facility, Session ID, Bin ID)` in the kept-FINAL lookup.
- `Missing Final Count`: 1 if no matching kept FINAL event exists; else 0.
- `Approval Gap`: 1 if ALL three conditions hold:
  (a) A kept FINAL event exists (Missing Final Count == 0),
  (b) `Approval Needed` == `YES` (case-insensitive comparison),
  (c) abs(`Expected Qty` - `Count Qty` from the kept FINAL event) > `Allowed Variance`.
  Otherwise 0.
- `Total Errors` = `Missing Final Count` + `Approval Gap`.
- `Error Summary`: exactly one of these strings:
  - `None` (if Total Errors == 0)
  - `Missing Final Count` (if only that flag is 1)
  - `Approval Gap` (if only that flag is 1)
  - `Missing Final Count, Approval Gap` (if both are 1)

**CRITICAL**: Write concrete numeric values (int 0 or 1) and text strings — NOT Excel formulas.

##### Sheet 4: `Summary`
- Aggregate from `Formatted Data` by `(Facility, Session ID)`.
- Columns with exactly these headers: `Facility`, `Session ID`, `Missing Final Counts`, `Approval Gaps`, `Total Errors`.
- Sum `Missing Final Count` → `Missing Final Counts`, sum `Approval Gap` → `Approval Gaps`, sum `Total Errors` → `Total Errors` for each group.
- Include ONLY groups where `Total Errors > 0`.
- Sort by `Facility` ascending, then `Session ID` ascending.
- Append a final Grand Total row: `Facility` = `Grand Total`, `Session ID` = `-`, and the remaining 3 columns = sums across all included groups.

#### Step 2: Build the Word document

Use `python-docx` to create `/root/Cycle_Count_Variance_Brief.docx`.

Write an executive summary paragraph (3-6 sentences) that includes:
1. A plain-language definition of both checks:
   - "Missing Final Count" means a bin had no valid final count event recorded.
   - "Approval Gap" means a bin's final count deviated from the expected quantity beyond the allowed variance threshold and required approval.
2. The computed totals: total Missing Final Counts, total Approval Gaps, and total Total Errors (use the Grand Total row values).
3. At least one actionable recommendation (e.g., prioritize recounts, improve count procedures).
4. Mention at least two specific high-priority `(Facility, Session ID)` combinations that had the most exceptions (highest Total Errors from the Summary sheet).

#### Step 3: Validation

After creating both files, verify:
1. Open `/root/Cycle_Count_Variance_Audit.xlsx` and confirm it has exactly 4 sheets named `Overview`, `RawData`, `Formatted Data`, `Summary`.
2. Print the first 5 rows of each sheet to confirm structure.
3. Print the `Formatted Data` headers — must be exactly: `Facility`, `Session ID`, `Bin ID`, `Product ID`, `Expected Qty`, `Allowed Variance`, `Approval Needed`, `Missing Final Count`, `Approval Gap`, `Total Errors`, `Error Summary`.
4. Print the `Summary` headers — must be exactly: `Facility`, `Session ID`, `Missing Final Counts`, `Approval Gaps`, `Total Errors`.
5. Confirm the last row of `Summary` has `Facility` = `Grand Total` and `Session ID` = `-`.
6. Print the Grand Total row values.
7. Confirm `/root/Cycle_Count_Variance_Brief.docx` exists and print its text content.

### Environment notes
- Install any needed packages: `pip install openpyxl python-docx pandas` if not already available.
- All file paths are absolute under `/root/`.
- Do NOT rename sheets or files from the exact names specified.
- When copying the Overview sheet, be especially careful to preserve all content exactly.

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
Task metadata: author_email=catpaw@meituan.com, author_name=CatPaw Benchmark Builder, category=spreadsheet-audit, difficulty=expert, tags=[excel, openpyxl, docx, audit, inventory].
Verifier config: timeout_sec=900.0.