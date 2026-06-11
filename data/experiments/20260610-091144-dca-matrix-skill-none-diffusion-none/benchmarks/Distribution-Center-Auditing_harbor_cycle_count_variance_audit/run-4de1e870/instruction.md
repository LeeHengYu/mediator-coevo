# Task Instruction

## Task: Cycle Count Variance Audit

You must produce two deliverables:
1. `/root/Cycle_Count_Variance_Audit.xlsx`
2. `/root/Cycle_Count_Variance_Brief.docx`

### Step-by-step Instructions

#### Step 0: Inspect Input Files
- Read and display the contents of `/root/Cycle_Plan.xlsx`, `/root/Count_Event_Log.xlsx`, and `/root/Cycle_Template.xlsx` to understand their structure (sheet names, column names, sample rows, data types).
- Pay special attention to the `Overview` sheet in `Cycle_Template.xlsx` — note its exact structure (merged cells, content, formatting) so you can copy it faithfully.
- Note the column names in `Cycle_Plan.xlsx` and `Count_Event_Log.xlsx` exactly as they appear.

#### Step 1: Build the Excel workbook `/root/Cycle_Count_Variance_Audit.xlsx`

Use `openpyxl` for all Excel operations so you can copy the Overview sheet faithfully.

##### Sheet 1: `Overview`
- Copy the `Overview` sheet from `Cycle_Template.xlsx` exactly — preserve all cell values, merged cells, formatting, column widths, and row heights. Do NOT modify any content.
- Use openpyxl to read the template workbook and replicate every cell (value, style, number format, font, fill, border, alignment) and every merged cell range into the output workbook's `Overview` sheet.

##### Sheet 2: `RawData`
- Copy the entire plan table from `Cycle_Plan.xlsx` exactly as-is (all columns, all rows, same order, same values). Write it starting at row 1 with headers.

##### Sheet 3: `Formatted Data`
- Start with the same rows (same order) as `RawData`.
- The first 7 columns must be exactly: `Facility`, `Session ID`, `Bin ID`, `Product ID`, `Expected Qty`, `Allowed Variance`, `Approval Needed`. Map from whatever column names exist in Cycle_Plan.xlsx to these exact header names.
- Process `Count_Event_Log.xlsx`:
  - Filter to rows where `Event Type` equals `FINAL` (case-insensitive comparison).
  - Drop rows where any of `Facility`, `Session ID`, `Bin ID`, or `Count Qty` is blank/NaN.
  - For each unique `(Facility, Session ID, Bin ID)` key, keep only the LAST row (latest by row order in the file — i.e., the row with the highest index). This gives the final count lookup table.
- For each row in the plan table, look up the matching `(Facility, Session ID, Bin ID)` in the final count lookup:
  - **Missing Final Count**: 1 if no matching final event exists, else 0.
  - **Approval Gap**: 1 if ALL three conditions hold: (a) a matching final event exists, (b) `Approval Needed` is `YES` (case-insensitive), (c) `abs(Expected Qty - Count Qty) > Allowed Variance`. Otherwise 0.
  - **Total Errors**: `Missing Final Count + Approval Gap`.
  - **Error Summary**: Exactly one of: `None`, `Missing Final Count`, `Approval Gap`, or `Missing Final Count, Approval Gap` — constructed by listing the error names for which the flag is 1, joined by `, `. If neither flag is 1, use `None`.
- Write all values as concrete numbers/strings (no Excel formulas).
- Column headers for columns 8-11 must be exactly: `Missing Final Count`, `Approval Gap`, `Total Errors`, `Error Summary`.

##### Sheet 4: `Summary`
- Aggregate from `Formatted Data` by `(Facility, Session ID)`.
- Columns: `Facility`, `Session ID`, `Missing Final Counts`, `Approval Gaps`, `Total Errors`.
- Sum `Missing Final Count` → `Missing Final Counts`, sum `Approval Gap` → `Approval Gaps`, sum `Total Errors` → `Total Errors` for each group.
- Include ONLY groups where `Total Errors > 0`.
- Sort by `Facility` ascending then `Session ID` ascending.
- Append a final row: `Facility` = `Grand Total`, `Session ID` = `-`, and the remaining columns are the grand totals across the entire dataset.

**IMPORTANT**: The sheet order in the workbook must be exactly: `Overview`, `RawData`, `Formatted Data`, `Summary`. Verify this after creation.

#### Step 2: Build the Word document `/root/Cycle_Count_Variance_Brief.docx`

Use `python-docx` to create `/root/Cycle_Count_Variance_Brief.docx`.

Write an executive summary paragraph (3-6 sentences) that includes:
1. A plain-language definition of both checks: explain what `Missing Final Count` means (a planned bin has no final count recorded) and what `Approval Gap` means (the variance between expected and counted quantity exceeds the allowed threshold for items requiring approval).
2. The computed totals: state the total number of Missing Final Counts, Approval Gaps, and Total Errors from the Grand Total row.
3. At least one actionable recommendation (e.g., prioritize recounts for bins with missing finals, review approval workflows).
4. Mention at least two specific high-priority `(Facility, Session ID)` combinations that have the most exceptions (highest Total Errors).

#### Step 3: Validation

After creating both files, run these checks:
1. Re-open `/root/Cycle_Count_Variance_Audit.xlsx` with openpyxl and verify:
   - Sheet names are exactly `['Overview', 'RawData', 'Formatted Data', 'Summary']`.
   - `RawData` row count matches the plan table.
   - `Formatted Data` has 11 columns with the correct headers.
   - `Formatted Data` has the same number of data rows as `RawData`.
   - `Summary` last row has `Facility` = `Grand Total`.
   - The sum of `Total Errors` in `Formatted Data` matches the Grand Total row in `Summary`.
   - `Overview` sheet has content (is not empty).
2. Re-open `/root/Cycle_Count_Variance_Brief.docx` and verify it contains text mentioning both check names and numeric totals.
3. Print all validation results. If any check fails, fix the issue and re-validate.

### Technical Notes
- Install any needed packages: `pip install openpyxl python-docx pandas`
- When copying the Overview sheet, be thorough with merged cells and cell styles. Use `copy` module for style objects if needed.
- Be careful with data type comparisons: ensure numeric columns are compared as numbers, not strings.
- When matching keys between the plan and event log, ensure the key values are compared consistently (strip whitespace, consistent types).
- Write the entire solution as a single Python script for reliability, then run it.

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