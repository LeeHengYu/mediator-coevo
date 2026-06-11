# Task Instruction

Execute the following steps in order:

## Step 1: Inspect the source workbook
- Read `/root/Timesheet_Submissions.xlsx` to understand its structure.
- List the sheet names. Read the `Entries` sheet (all rows and columns) and the `BreakRules` sheet (all rows and columns).
- Print the first few rows of each to understand column names and data types.

## Step 2: Build the audit workbook `/root/Timesheet_Compliance_Audit.xlsx`

Use Python with `openpyxl` (and `pandas` for data manipulation if helpful). Create the workbook with exactly three worksheets named `RawData`, `Formatted Data`, and `Summary`.

### 2a) `RawData` sheet
- Copy the `Entries` table exactly (headers + all data rows, preserving order and values).

### 2b) `Formatted Data` sheet
- Same rows in the same order as `RawData`.
- First 8 columns must be exactly: `Week Ending`, `Employee ID`, `Role`, `Hours Worked`, `Break Minutes`, `Approval Code`, `Project Code`, `Manager`.
- Add 4 new columns (columns 9–12) with headers: `Break Deficit`, `Approval Missing`, `Total Errors`, `Error Summary`.
- To compute these, load the `BreakRules` sheet into a dictionary keyed by `Role`. For each row:
  - `Break Deficit` = 1 if `Break Minutes` < `Min Break Minutes` for that Role, else 0.
  - `Approval Missing` = 1 if `Hours Worked` > `Overtime Threshold` for that Role AND `Approval Code` is blank/empty/NaN, else 0.
  - `Total Errors` = `Break Deficit` + `Approval Missing`.
  - `Error Summary`: if both flags are 1 → `Break Deficit, Approval Missing`; if only break → `Break Deficit`; if only approval → `Approval Missing`; if neither → `None`.
- Write concrete numeric values (integers 0 or 1) and text strings — no Excel formulas.
- IMPORTANT: When checking if `Approval Code` is blank, treat `None`, `NaN`, empty string, and whitespace-only strings all as blank.

### 2c) `Summary` sheet
- Headers: `Employee ID`, `Week Ending`, `Break Deficits`, `Approval Gaps`, `Total Errors`.
- Aggregate from the Formatted Data by (Employee ID, Week Ending): sum `Break Deficit` → `Break Deficits`, sum `Approval Missing` → `Approval Gaps`, sum `Total Errors` → `Total Errors`.
- Include only groups where `Total Errors > 0`.
- Sort by `Employee ID` ascending, then `Week Ending` ascending.
- Append a final row: `Employee ID` = `Grand Total`, `Week Ending` = `-`, and the remaining three columns = dataset-wide totals (sum of all Break Deficits, Approval Gaps, Total Errors from the included rows — which should equal the totals across ALL rows since we're summing the flags).
- IMPORTANT: The Grand Total row should sum across ALL rows of Formatted Data (not just the filtered groups), to represent dataset totals. Actually re-read the spec: "remaining columns = dataset totals" — this means totals across the entire dataset. Compute the grand totals from the full Formatted Data, not just the filtered summary rows.

## Step 3: Build the Word document `/root/Timesheet_Compliance_Brief.docx`

Use `python-docx`. Create `/root/Timesheet_Compliance_Brief.docx` with:
- A heading (e.g., "Timesheet Compliance Brief" or "Executive Summary").
- A paragraph of 3–6 sentences that includes:
  - Plain-language definition of both checks: Break Deficit (break time below the role-specific minimum) and Approval Missing (overtime hours without required approval code).
  - The computed totals for Break Deficits, Approval Gaps, and Total Errors (use the actual numbers from your computation).
  - At least one actionable recommendation.
  - Mention at least two specific high-priority Employee IDs that have the most frequent exceptions (identify these from the Summary data — pick the top 2 by Total Errors).

## Step 4: Validate
- Re-open `/root/Timesheet_Compliance_Audit.xlsx` and verify:
  - Exactly 3 sheets with exact names `RawData`, `Formatted Data`, `Summary`.
  - `RawData` row count matches original Entries.
  - `Formatted Data` has 12 columns with correct headers.
  - `Summary` last row has `Grand Total` in first column.
  - Print a few sample rows from each sheet.
- Re-open `/root/Timesheet_Compliance_Brief.docx` and print its text to confirm content.
- Print the BreakRules thresholds used and a few computed rows to confirm correctness of the flag logic.

If `python-docx` is not installed, install it with `pip install python-docx`. Similarly ensure `openpyxl` is available.

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
Task metadata: author_email=catpaw@meituan.com, author_name=CatPaw Benchmark Builder, category=spreadsheet-audit, difficulty=hard, tags=[excel, openpyxl, docx, audit, timesheet].
Verifier config: timeout_sec=900.0.