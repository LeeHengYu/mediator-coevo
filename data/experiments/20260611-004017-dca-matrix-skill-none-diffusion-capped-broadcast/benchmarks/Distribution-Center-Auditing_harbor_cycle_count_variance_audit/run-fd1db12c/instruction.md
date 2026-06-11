# Task Instruction

Create a Python script `/root/solve.py` and execute it. The script must do the following:

## Step 0: Inspect source files
- Read `/root/Cycle_Plan.xlsx`, `/root/Count_Event_Log.xlsx`, and `/root/Cycle_Template.xlsx` using openpyxl and/or pandas.
- Print column names and first few rows of each to understand the data structure.
- Print sheet names of `Cycle_Template.xlsx`.

## Step 1: Build `/root/Cycle_Count_Variance_Audit.xlsx` with exactly 4 sheets in this order: `Overview`, `RawData`, `Formatted Data`, `Summary`.

### Overview sheet
- Copy the `Overview` sheet from `Cycle_Template.xlsx` exactly, preserving all cell values, formatting, and merged cells as much as possible. Use openpyxl to copy cell-by-cell.

### RawData sheet
- Copy the plan table from `Cycle_Plan.xlsx` exactly. Use openpyxl to copy cell-by-cell to preserve original values (including any 'N/A' strings that pandas might convert to NaN). **Critical**: Do NOT use pandas for this copy — use openpyxl directly to avoid NaN conversion of empty or 'N/A' cells.

### Formatted Data sheet
- Read `Cycle_Plan.xlsx` with pandas for computation, but be careful: use `keep_default_na=False` or `na_values=[]` to prevent converting 'N/A' strings to NaN.
- Keep the same row order as RawData.
- First 7 columns exactly: Facility, Session ID, Bin ID, Product ID, Expected Qty, Allowed Variance, Approval Needed.
- Read `Count_Event_Log.xlsx` (also with `keep_default_na=False`). Filter for rows where `Event Type` == 'FINAL'. Drop rows with blank keys (Facility, Session ID, Bin ID) or blank `Count Qty`. For each unique (Facility, Session ID, Bin ID), keep only the latest row (last occurrence or by timestamp if available).
- Add columns 8-11:
  - `Missing Final Count`: 1 if no kept FINAL event exists for that (Facility, Session ID, Bin ID), else 0.
  - `Approval Gap`: 1 if ALL three conditions hold: (a) a kept final event exists, (b) `Approval Needed` == 'YES' (case-insensitive), (c) abs(Expected Qty - Count Qty) > Allowed Variance. Otherwise 0.
  - `Total Errors`: Missing Final Count + Approval Gap.
  - `Error Summary`: exactly one of: 'None', 'Missing Final Count', 'Approval Gap', 'Missing Final Count, Approval Gap'.
- Write concrete numeric/text values (no formulas).
- When writing this sheet, ensure all original column values are preserved exactly (use the same `keep_default_na=False` approach).

### Summary sheet
- Headers: Facility, Session ID, Missing Final Counts, Approval Gaps, Total Errors.
- Aggregate from Formatted Data by (Facility, Session ID).
- Include only groups where Total Errors > 0.
- Sort by Facility ascending, then Session ID ascending.
- Append a Grand Total row: Facility='Grand Total', Session ID='-', and sum columns.

## Step 2: Build `/root/Cycle_Count_Variance_Brief.docx`
- Use python-docx to create a Word document.
- Write an executive summary (3-6 sentences) that includes:
  - Plain-language definition of both checks: Missing Final Count means a bin had no finalized recount event recorded; Approval Gap means a finalized count deviated from expected quantity beyond the allowed variance threshold for a bin requiring approval.
  - The computed totals for Missing Final Counts, Approval Gaps, and Total Errors (use actual numbers from the data).
  - At least one actionable recommendation.
  - Mention at least two high-priority facility-session combinations with the most exceptions. Format these as 'FACXXX SESSXXX' (facility and session ID separated by a space) so they are recognizable as facility-session pairs.

## Step 3: Validation
- Re-read the generated Excel file and print:
  - Sheet names
  - First 3 rows of each sheet
  - Row counts for Formatted Data and Summary
  - Sample of computed columns from Formatted Data
  - Grand Total row from Summary
- Re-read the Word doc and print its text content.

## Critical Implementation Notes
- **NaN/None prevention**: When reading Excel files with pandas, always use `keep_default_na=False` to prevent 'N/A' or empty strings from becoming NaN. For the RawData sheet, use openpyxl cell-by-cell copy to guarantee exact preservation.
- **Ensure numeric types**: Convert Expected Qty, Allowed Variance, and Count Qty to numeric types before arithmetic comparisons.
- **Sheet order matters**: Create sheets in the exact order: Overview, RawData, Formatted Data, Summary.
- **Install dependencies if needed**: `pip install openpyxl python-docx pandas` at the start of the script.

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