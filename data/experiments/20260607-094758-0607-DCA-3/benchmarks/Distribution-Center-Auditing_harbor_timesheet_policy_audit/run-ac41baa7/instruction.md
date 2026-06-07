# Task Instruction

Build two deliverables from `/root/Timesheet_Submissions.xlsx`: an Excel audit workbook and a Word executive brief. Follow these steps exactly.

## Step 1: Inspect the source workbook
- Load `/root/Timesheet_Submissions.xlsx` with pandas/openpyxl.
- Read both sheets: `Entries` (line-level submissions) and `BreakRules` (role-based thresholds).
- Print the columns, dtypes, and a few sample rows of each so you understand the exact column names in `BreakRules` (expect something like `Role`, `Min Break Minutes`, `Overtime Threshold` — confirm before coding).
- Confirm the `Entries` sheet has the 8 columns listed in the task (Week Ending, Employee ID, Role, Hours Worked, Break Minutes, Approval Code, Project Code, Manager).

## Step 2: Create `/root/Timesheet_Compliance_Audit.xlsx` with three sheets in this order: `RawData`, `Formatted Data`, `Summary`.

### Sheet `RawData`
- Write the `Entries` dataframe exactly as-is (same columns, same row order, same values).

### Sheet `Formatted Data`
- Start from the same rows in the same order as `RawData`.
- Keep the first 8 columns exactly with these headers in this order: `Week Ending`, `Employee ID`, `Role`, `Hours Worked`, `Break Minutes`, `Approval Code`, `Project Code`, `Manager`.
- Build a lookup dict from `BreakRules`: `role -> (min_break_minutes, overtime_threshold)` using the actual column names from BreakRules (do NOT hardcode role-specific numbers).
- For each row compute concrete numeric/text values (not formulas):
  - `Break Deficit` (int): 1 if `Break Minutes` < that role's `Min Break Minutes`, else 0.
  - `Approval Missing` (int): 1 if `Hours Worked` > that role's `Overtime Threshold` AND `Approval Code` is blank (NaN, None, or empty/whitespace string), else 0.
  - `Total Errors` (int): sum of the two.
  - `Error Summary` (str): exactly one of `None`, `Break Deficit`, `Approval Missing`, `Break Deficit, Approval Missing` based on which flags are set.
- Append these as columns 9–12 with headers: `Break Deficit`, `Approval Missing`, `Total Errors`, `Error Summary`.

### Sheet `Summary`
- Headers exactly: `Employee ID`, `Week Ending`, `Break Deficits`, `Approval Gaps`, `Total Errors`.
- Group `Formatted Data` by `(Employee ID, Week Ending)` and sum `Break Deficit`, `Approval Missing`, `Total Errors` (rename to `Break Deficits`, `Approval Gaps`, `Total Errors`).
- Keep only groups where `Total Errors > 0`.
- Sort by `Employee ID` ascending, then `Week Ending` ascending.
- Append one final row: `Employee ID` = `Grand Total`, `Week Ending` = `-`, and the remaining three columns equal to the dataset-wide totals (sum across all rows of `Formatted Data`, not just filtered groups — i.e. totals of `Break Deficit`, `Approval Missing`, `Total Errors` columns).

Write all three sheets using openpyxl engine. Verify by reopening the file and checking sheet names and headers.

## Step 3: Create `/root/Timesheet_Compliance_Brief.docx`
Use python-docx. Write a 3–6 sentence executive summary that includes:
- Plain-language definitions of both checks: `Break Deficit` (employee logged fewer break minutes than their role's required minimum) and `Approval Missing` (employee exceeded their role's overtime threshold without an approval code).
- The computed totals for `Break Deficits`, `Approval Gaps`, and `Total Errors` (use the Grand Total values you just computed — substitute real numbers).
- At least one concrete actionable recommendation (e.g., require managers to submit approval codes before payroll close, or add automated break-minimum prompts).
- Mention at least two specific high-priority Employee IDs — pick the top two Employee IDs by total errors from the Summary table; include the actual ID strings.

## Step 4: Validation before finishing
- Reopen `/root/Timesheet_Compliance_Audit.xlsx` and confirm: exactly the three sheet names, `Formatted Data` has 12 columns with the exact headers above, `Summary` has exactly 5 columns with the exact headers above and ends with a `Grand Total` row.
- Confirm `Error Summary` values are only from the allowed 4 strings.
- Confirm `/root/Timesheet_Compliance_Brief.docx` exists and contains the totals and at least two employee IDs.
- Do not hardcode role thresholds; they must come from the `BreakRules` sheet.
- Do not write formulas in the added columns — write concrete values.

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