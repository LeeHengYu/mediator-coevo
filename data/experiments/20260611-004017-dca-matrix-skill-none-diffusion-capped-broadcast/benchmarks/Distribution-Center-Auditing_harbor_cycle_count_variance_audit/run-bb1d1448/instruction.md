# Task Instruction

## Task: Cycle Count Variance Audit

You must produce two deliverable files:
1. `/root/Cycle_Count_Variance_Audit.xlsx`
2. `/root/Cycle_Count_Variance_Brief.docx`

### Step-by-step Instructions

#### Step 0: Inspect Input Files
- Read and inspect `/root/Cycle_Plan.xlsx`, `/root/Count_Event_Log.xlsx`, and `/root/Cycle_Template.xlsx`.
- Print the sheet names of each file.
- Print the first few rows of each relevant sheet to understand column names, data types, and structure.
- Pay special attention to the `Overview` sheet in `Cycle_Template.xlsx` — note its exact structure (merged cells, formatting, content).

#### Step 1: Build the `RawData` sheet data
- Read the plan table from `Cycle_Plan.xlsx`. Note the exact column names. This data will be copied verbatim into the `RawData` sheet.
- Confirm the first 7 columns correspond to: Facility, Session ID, Bin ID, Product ID, Expected Qty, Allowed Variance, Approval Needed. If column names differ slightly, note the exact names from the file.

#### Step 2: Process the Count Event Log
- Read `Count_Event_Log.xlsx`.
- Filter to rows where `Event Type` equals `FINAL` (case-insensitive comparison).
- Drop rows where any of `Facility`, `Session ID`, `Bin ID`, or `Count Qty` is blank/NaN.
- For each unique `(Facility, Session ID, Bin ID)` group, keep only the **last** row (latest row by position in the spreadsheet, i.e., the row with the largest index). This gives the final count for each bin.
- Store this as a lookup dictionary: key = `(Facility, Session ID, Bin ID)` → value = `Count Qty`.

#### Step 3: Build the `Formatted Data` sheet
- Start with the same rows (same order) as `RawData`.
- Keep the first 7 columns exactly as they are (with headers: Facility, Session ID, Bin ID, Product ID, Expected Qty, Allowed Variance, Approval Needed).
- For each row, look up the final count using key `(Facility, Session ID, Bin ID)` from Step 2.
- Compute the 4 new columns:
  - **Missing Final Count**: 1 if no FINAL event exists for this key, else 0.
  - **Approval Gap**: 1 if ALL three conditions hold: (a) a FINAL event exists (Missing Final Count == 0), (b) `Approval Needed` equals `YES` (case-insensitive), (c) `abs(Expected Qty - Count Qty) > Allowed Variance`. Otherwise 0.
  - **Total Errors**: `Missing Final Count + Approval Gap`.
  - **Error Summary**: Exactly one of these strings:
    - `None` (if Total Errors == 0)
    - `Missing Final Count` (if Missing Final Count == 1 and Approval Gap == 0)
    - `Approval Gap` (if Missing Final Count == 0 and Approval Gap == 1)
    - `Missing Final Count, Approval Gap` (if both == 1)
- Write concrete values (integers for numeric columns, exact strings for Error Summary). Do NOT use Excel formulas.

#### Step 4: Build the `Summary` sheet
- From the `Formatted Data`, group by `(Facility, Session ID)`.
- For each group, sum `Missing Final Count`, `Approval Gap`, and `Total Errors`.
- Keep only groups where `Total Errors > 0`.
- Sort by `Facility` ascending, then `Session ID` ascending.
- Append a Grand Total row: Facility = `Grand Total`, Session ID = `-`, and the remaining 3 columns = sums across all kept groups.
- Headers must be exactly: `Facility`, `Session ID`, `Missing Final Counts`, `Approval Gaps`, `Total Errors`.

#### Step 5: Write `/root/Cycle_Count_Variance_Audit.xlsx`
- Use `openpyxl` to create the workbook.
- **Overview sheet**: Copy the `Overview` sheet from `Cycle_Template.xlsx` as faithfully as possible. Use openpyxl to read the template workbook and copy cell values, merged cells, column widths, and formatting. The sheet must be named exactly `Overview`.
- **RawData sheet**: Write the plan data with its original headers. Sheet name exactly `RawData`.
- **Formatted Data sheet**: Write the formatted data with all 11 columns. Sheet name exactly `Formatted Data`.
- **Summary sheet**: Write the summary table. Sheet name exactly `Summary`.
- Ensure the sheet order is: Overview, RawData, Formatted Data, Summary.
- Save the file.

#### Step 6: Verify the Excel output
- Re-read `/root/Cycle_Count_Variance_Audit.xlsx` and confirm:
  - Sheet names are exactly `['Overview', 'RawData', 'Formatted Data', 'Summary']`.
  - `RawData` row count matches the plan.
  - `Formatted Data` has 11 columns with correct headers.
  - `Summary` has correct headers and a Grand Total row.
  - Print the Summary sheet contents to confirm correctness.
  - Print a few rows of Formatted Data to spot-check.

#### Step 7: Create `/root/Cycle_Count_Variance_Brief.docx`
- Use `python-docx` to create a Word document.
- Write an executive summary (3-6 sentences) that includes:
  1. A plain-language definition of **Missing Final Count** (a bin that was scheduled for counting but has no final count recorded) and **Approval Gap** (a bin where the variance between expected and actual count exceeds the allowed threshold and requires approval).
  2. The computed totals: total Missing Final Counts, total Approval Gaps, and total Total Errors (use the Grand Total row values from the Summary sheet).
  3. At least one actionable recommendation (e.g., investigate bins with missing counts, enforce approval workflows).
  4. Mention at least two specific high-priority `(Facility, Session ID)` combinations that have the most exceptions (pick the top 2 by Total Errors from the Summary sheet).
- Save the file.

#### Step 8: Final Validation
- Confirm both files exist: `/root/Cycle_Count_Variance_Audit.xlsx` and `/root/Cycle_Count_Variance_Brief.docx`.
- Re-read the docx and print its text to verify content requirements are met.
- Print 'DONE' when complete.

### Important Notes
- Install any needed packages (`openpyxl`, `python-docx`, `pandas`) if not already available.
- When copying the Overview sheet, preserve it as faithfully as possible — do not modify any cell values.
- All sheet names and file names must be exactly as specified (case-sensitive).
- For `Formatted Data`, write hardcoded values, not formulas.
- For the `Summary` sheet, note the column headers use plural forms: `Missing Final Counts`, `Approval Gaps`, `Total Errors`.
- When comparing strings like `Approval Needed` == `YES`, use case-insensitive comparison and strip whitespace.

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