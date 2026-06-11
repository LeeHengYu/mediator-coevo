# Task Instruction

## Task: Cycle Count Variance Audit

You must produce two deliverables:
1. `/root/Cycle_Count_Variance_Audit.xlsx`
2. `/root/Cycle_Count_Variance_Brief.docx`

### Step-by-step Instructions

#### Step 0: Inspect Input Files
- Read and display the contents of `/root/Cycle_Plan.xlsx` (all sheets, show column names and first few rows).
- Read and display the contents of `/root/Count_Event_Log.xlsx` (all sheets, show column names and first few rows).
- Read and display the contents of `/root/Cycle_Template.xlsx` (list all sheet names; for the `Overview` sheet, show its full contents including any merged cells, formatting notes, or text).
- Print the exact column names for each file. This is critical — do NOT assume column names. Use the actual names from the files.

#### Step 1: Build the Excel workbook `/root/Cycle_Count_Variance_Audit.xlsx`

Use `openpyxl` for all Excel operations to preserve formatting from the template.

##### Sheet 1: `Overview`
- Copy the `Overview` sheet from `Cycle_Template.xlsx` exactly — preserve all cell values, merged cells, column widths, and formatting as much as possible. Use openpyxl to read the template and copy cell-by-cell.

##### Sheet 2: `RawData`
- Copy the plan table from `Cycle_Plan.xlsx` exactly (all rows, all columns, same order).

##### Sheet 3: `Formatted Data`
- Start with the same rows (same order) as `RawData`.
- The first 7 columns must be exactly: `Facility`, `Session ID`, `Bin ID`, `Product ID`, `Expected Qty`, `Allowed Variance`, `Approval Needed`. Map from the actual column names in `Cycle_Plan.xlsx` if they differ slightly (but print what you find first).
- Process `Count_Event_Log.xlsx`:
  - Filter to rows where `Event Type` equals `FINAL` (case-insensitive comparison).
  - Drop rows where any of `Facility`, `Session ID`, `Bin ID`, or `Count Qty` is blank/NaN.
  - For each unique `(Facility, Session ID, Bin ID)` group, keep only the LAST row (latest row by position in the spreadsheet, i.e., highest index — unless there's an explicit timestamp column, in which case use that). This gives the "kept final event" lookup.
- For each row in the plan:
  - Look up the kept final event by matching `(Facility, Session ID, Bin ID)`.
  - `Missing Final Count` = 1 if no matching kept final event exists, else 0.
  - `Approval Gap` = 1 if ALL three conditions hold:
    1. A kept final event exists (Missing Final Count == 0).
    2. `Approval Needed` equals `YES` (case-insensitive, strip whitespace).
    3. `abs(Expected Qty - Count Qty)` is strictly greater than `Allowed Variance`.
    Otherwise 0.
  - `Total Errors` = `Missing Final Count` + `Approval Gap`.
  - `Error Summary`:
    - If both flags are 0: `None`
    - If only Missing Final Count is 1: `Missing Final Count`
    - If only Approval Gap is 1: `Approval Gap`
    - If both are 1: `Missing Final Count, Approval Gap`
- Write concrete values (integers and strings), NOT Excel formulas.

##### Sheet 4: `Summary`
- Headers exactly: `Facility`, `Session ID`, `Missing Final Counts`, `Approval Gaps`, `Total Errors`.
- Aggregate from `Formatted Data` by `(Facility, Session ID)`: sum `Missing Final Count` → `Missing Final Counts`, sum `Approval Gap` → `Approval Gaps`, sum `Total Errors` → `Total Errors`.
- Include ONLY groups where `Total Errors > 0`.
- Sort by `Facility` ascending, then `Session ID` ascending.
- Append a final row: `Facility` = `Grand Total`, `Session ID` = `-`, and the remaining columns are the grand totals across all included rows (sum of the filtered groups).

#### Step 2: Save the workbook
- Ensure sheet order is exactly: `Overview`, `RawData`, `Formatted Data`, `Summary`.
- Save to `/root/Cycle_Count_Variance_Audit.xlsx`.

#### Step 3: Build the Word document `/root/Cycle_Count_Variance_Brief.docx`

Use `python-docx`.

Write a short executive summary (3–6 sentences) that includes:
- A plain-language definition of both checks: what `Missing Final Count` means (a planned bin that was never given a final count) and what `Approval Gap` means (a bin where the variance between expected and actual quantity exceeded the allowed threshold and required approval).
- The computed grand totals for Missing Final Counts, Approval Gaps, and Total Errors (use the actual numbers from your Summary Grand Total row).
- At least one actionable recommendation (e.g., prioritize recounts, review approval workflows).
- Mention at least two specific high-priority `(Facility, Session ID)` combinations that had the most exceptions (highest Total Errors).

Save to `/root/Cycle_Count_Variance_Brief.docx`.

#### Step 4: Verification
- Re-read `/root/Cycle_Count_Variance_Audit.xlsx` and verify:
  - It has exactly 4 sheets named `Overview`, `RawData`, `Formatted Data`, `Summary`.
  - `RawData` row count matches `Cycle_Plan.xlsx`.
  - `Formatted Data` has 11 columns with the exact headers specified.
  - `Formatted Data` row count matches `RawData`.
  - `Summary` last row has `Facility` = `Grand Total`.
  - `Summary` Grand Total `Total Errors` equals the sum of all `Total Errors` in `Formatted Data`.
- Re-read `/root/Cycle_Count_Variance_Brief.docx` and print its text to confirm it has the required content.
- Print confirmation of all checks passing.

### Important Notes
- Do NOT skip the initial inspection step. The exact column names in the input files are critical.
- If `Count_Event_Log.xlsx` has a timestamp column, use it to determine the latest FINAL event. Otherwise use row order (last row = latest).
- All numeric values in columns 8-11 of `Formatted Data` must be written as Python integers, not floats or strings.
- `Error Summary` values must be exact strings: `None`, `Missing Final Count`, `Approval Gap`, or `Missing Final Count, Approval Gap`.
- Preserve the `Overview` sheet from the template as faithfully as possible.

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