# Task Instruction

Build a timesheet compliance audit from `/root/Timesheet_Submissions.xlsx` and produce two deliverables: `/root/Timesheet_Compliance_Audit.xlsx` and `/root/Timesheet_Compliance_Brief.docx`.

## Step 1: Inspect source workbook
- Open `/root/Timesheet_Submissions.xlsx` with pandas/openpyxl.
- Read both sheets: `Entries` and `BreakRules`.
- Print the columns, dtypes, and a few sample rows of each so you understand the exact column names and role values. Verify `BreakRules` has columns for Role, Min Break Minutes, and Overtime Threshold (use the actual header names you find).

## Step 2: Create `/root/Timesheet_Compliance_Audit.xlsx` with exactly three sheets in this order: `RawData`, `Formatted Data`, `Summary`.

### Sheet `RawData`
- Write the `Entries` dataframe exactly as read (same columns, same row order, same values).

### Sheet `Formatted Data`
- Start from the `Entries` data, preserving row order.
- Keep exactly these first 8 columns in this order with these exact headers:
  1. Week Ending
  2. Employee ID
  3. Role
  4. Hours Worked
  5. Break Minutes
  6. Approval Code
  7. Project Code
  8. Manager
- Build a lookup from `BreakRules` keyed by Role -> (Min Break Minutes, Overtime Threshold). Do not hardcode thresholds by role name.
- Compute and write concrete numeric/text values (not formulas) for columns 9-12:
  9. `Break Deficit`: 1 if `Break Minutes` < role's Min Break Minutes, else 0.
  10. `Approval Missing`: 1 if `Hours Worked` > role's Overtime Threshold AND `Approval Code` is blank/NaN/empty-string, else 0.
  11. `Total Errors`: sum of the two above.
  12. `Error Summary`: exactly one of `None`, `Break Deficit`, `Approval Missing`, `Break Deficit, Approval Missing` based on which flags are 1.
- Treat "blank" Approval Code as NaN, None, or whitespace-only string.

### Sheet `Summary`
- Aggregate `Formatted Data` grouped by (`Employee ID`, `Week Ending`):
  - `Break Deficits` = sum of Break Deficit
  - `Approval Gaps` = sum of Approval Missing
  - `Total Errors` = sum of Total Errors
- Keep only groups where `Total Errors > 0`.
- Sort by `Employee ID` ascending, then `Week Ending` ascending.
- Headers must be exactly: `Employee ID`, `Week Ending`, `Break Deficits`, `Approval Gaps`, `Total Errors`.
- Append final row: `Employee ID` = `Grand Total`, `Week Ending` = `-`, and the remaining three columns = totals across the included rows (which equal dataset totals since excluded rows contribute 0).

## Step 3: Create `/root/Timesheet_Compliance_Brief.docx`
Using `python-docx`, write a 3-6 sentence executive summary that includes ALL of the following:
- A plain-language definition of `Break Deficit` (insufficient break minutes vs. the role's minimum) and `Approval Missing` (overtime worked without an approval code).
- The computed dataset totals for Break Deficits, Approval Gaps, and Total Errors (use the numbers you computed).
- At least one actionable recommendation (e.g., manager training, automated approval reminders).
- Explicit mention of at least two high-priority employee IDs that have the most exceptions (pick the top 2 by Total Errors from the Summary table, breaking ties consistently).

## Step 4: Validation before finishing
- Reopen `/root/Timesheet_Compliance_Audit.xlsx` and confirm:
  - Sheet names are exactly `RawData`, `Formatted Data`, `Summary`.
  - `Formatted Data` has exactly 12 columns with the exact headers listed above.
  - `Error Summary` values are only from the allowed set.
  - Summary has exact headers, is filtered to Total Errors > 0, sorted correctly, and ends with the `Grand Total` row.
  - Values are concrete (not formulas).
- Reopen `/root/Timesheet_Compliance_Brief.docx` and confirm it contains both definitions, the three totals, a recommendation, and at least two employee IDs.

## Constraints
- Use thresholds dynamically from `BreakRules`; do not hardcode role-specific numbers.
- Preserve exact filenames, sheet names, and column headers.
- Preserve `RawData` row order and content exactly as in `Entries`.

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