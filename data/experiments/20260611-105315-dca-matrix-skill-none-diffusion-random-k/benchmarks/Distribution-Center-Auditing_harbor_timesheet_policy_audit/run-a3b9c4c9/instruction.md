# Task Instruction

Execute the following steps to produce the two deliverables.

## 1. Inspect the source workbook
```python
import openpyxl, pandas as pd
wb = openpyxl.load_workbook('/root/Timesheet_Submissions.xlsx')
print(wb.sheetnames)
```
Then read both sheets:
```python
entries = pd.read_excel('/root/Timesheet_Submissions.xlsx', sheet_name='Entries')
break_rules = pd.read_excel('/root/Timesheet_Submissions.xlsx', sheet_name='BreakRules')
print(entries.head(10))
print(entries.columns.tolist())
print(break_rules.head(10))
print(break_rules.columns.tolist())
print(entries.dtypes)
print(break_rules.dtypes)
```
Confirm column names before proceeding. The Entries sheet should have columns including Week Ending, Employee ID, Role, Hours Worked, Break Minutes, Approval Code, Project Code, Manager. The BreakRules sheet should have Role, Min Break Minutes, and Overtime Threshold (or similar).

## 2. Build the output Excel workbook

Write a single Python script that:

### 2a. RawData sheet
- Copy the `Entries` DataFrame exactly (same columns, same order) into a sheet named `RawData`.

### 2b. Formatted Data sheet
- Start from the Entries DataFrame.
- Merge with BreakRules on the `Role` column to get `Min Break Minutes` and `Overtime Threshold` per row.
- Compute four new columns:
  - **Break Deficit**: 1 if `Break Minutes` < `Min Break Minutes` for that role, else 0.
  - **Approval Missing**: 1 if `Hours Worked` > `Overtime Threshold` for that role AND `Approval Code` is blank/NaN/empty-string, else 0. Use: `(row['Hours Worked'] > row['Overtime Threshold']) & (row['Approval Code'].isna() | (row['Approval Code'].astype(str).str.strip() == ''))`.
  - **Total Errors**: Break Deficit + Approval Missing.
  - **Error Summary**: Exactly one of: `None`, `Break Deficit`, `Approval Missing`, `Break Deficit, Approval Missing` — built by joining the names of non-zero flags with `, `, defaulting to `None` if both are 0.
- Keep only these 12 columns in order: Week Ending, Employee ID, Role, Hours Worked, Break Minutes, Approval Code, Project Code, Manager, Break Deficit, Approval Missing, Total Errors, Error Summary.
- Same row order as RawData.
- Write concrete values (no Excel formulas).

### 2c. Summary sheet
- From Formatted Data, group by (Employee ID, Week Ending).
- Aggregate: Break Deficits = sum of Break Deficit, Approval Gaps = sum of Approval Missing, Total Errors = sum of Total Errors.
- Filter to groups where Total Errors > 0.
- Sort by Employee ID ascending, then Week Ending ascending.
- Append a Grand Total row: Employee ID = 'Grand Total', Week Ending = '-', and sums of the three numeric columns.
- Headers exactly: Employee ID, Week Ending, Break Deficits, Approval Gaps, Total Errors.

Write all three sheets to `/root/Timesheet_Compliance_Audit.xlsx` using `openpyxl` as the engine (via `pd.ExcelWriter`). Make sure sheet names are exactly `RawData`, `Formatted Data`, `Summary`.

## 3. Build the Word document

Using `python-docx`:
- Create `/root/Timesheet_Compliance_Brief.docx`.
- Write a heading: "Timesheet Compliance Brief".
- Write an executive summary paragraph (3-6 sentences) that:
  1. Defines both checks in plain language: "Break Deficit flags entries where an employee's recorded break time falls below the minimum required for their role" and "Approval Missing flags overtime entries that lack a required approval code."
  2. States the computed totals: total Break Deficits, total Approval Gaps, and total Total Errors (use the Grand Total row values).
  3. Identifies at least two high-priority Employee IDs — pick the top 2 employees by Total Errors from the Summary data and mention them by ID.
  4. Includes at least one actionable recommendation, e.g., "We recommend that managers prioritize reviewing timesheets for [Employee IDs] and enforce mandatory break compliance training."

## 4. Validate
- Re-read the output Excel file and print sheet names, column headers for each sheet, row counts, and the last few rows of Summary (including Grand Total).
- Re-read the Word document and print all paragraph texts to confirm content.
- Confirm files exist at the exact paths: `/root/Timesheet_Compliance_Audit.xlsx` and `/root/Timesheet_Compliance_Brief.docx`.

## Important notes
- Install any needed packages (`pip install openpyxl python-docx pandas`) before running.
- Use thresholds dynamically from BreakRules — do NOT hardcode role-specific numbers.
- Adapt column name references to the actual column names found in step 1 (they should match the spec, but verify).
- For the Approval Code blank check, handle both NaN and empty strings.
- The Error Summary string `None` should be the literal text string "None", not Python's None/NaN.

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