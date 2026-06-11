# Task Instruction

Execute the following steps in order to produce the two deliverables.

## Step 1 – Inspect the source workbook
```python
import openpyxl
wb = openpyxl.load_workbook('/root/Timesheet_Submissions.xlsx')
print('Sheet names:', wb.sheetnames)

# Print Entries headers + first 5 rows
ws_entries = wb['Entries']
for i, row in enumerate(ws_entries.iter_rows(values_only=True)):
    print(row)
    if i >= 5:
        break

# Print BreakRules headers + all rows
ws_br = wb['BreakRules']
for row in ws_br.iter_rows(values_only=True):
    print(row)

wb.close()
```
Understand the column positions, data types (especially dates), and the BreakRules mapping before proceeding.

## Step 2 – Build both output files with a single Python script

Write and run `/root/build_audit.py` with the logic below. Key requirements:

### Date handling (CRITICAL – previous failure)
- When writing `Week Ending` to **both** `Formatted Data` and `Summary` sheets, convert every date value to a **plain string in 'YYYY-MM-DD' format** using:
  ```python
  str(val.date()) if hasattr(val, 'date') else str(val)[:10]
  ```
  This must produce strings like `'2026-02-07'`, never `datetime` objects and never strings with `' 00:00:00'`.

### Reading data
1. Read the `Entries` sheet into a list of dicts (or list of lists with a header row).
2. Read the `BreakRules` sheet into a dict keyed by `Role` → `{min_break_minutes, overtime_threshold}`.

### RawData sheet
- Copy the `Entries` data exactly as-is (preserve original types including dates).

### Formatted Data sheet
- Same row order as RawData.
- First 8 columns: Week Ending (as YYYY-MM-DD string), Employee ID, Role, Hours Worked, Break Minutes, Approval Code, Project Code, Manager.
- Columns 9-12 with **exactly** these headers: `Break Deficit`, `Approval Missing`, `Total Errors`, `Error Summary`.
- Compute:
  - `Break Deficit` = 1 if `Break Minutes < min_break_minutes` for that role, else 0.
  - `Approval Missing` = 1 if `Hours Worked > overtime_threshold` for that role AND `Approval Code` is blank/None/empty, else 0.
  - `Total Errors` = Break Deficit + Approval Missing.
  - `Error Summary`:
    - Both flags 1 → `'Break Deficit, Approval Missing'`
    - Only break → `'Break Deficit'`
    - Only approval → `'Approval Missing'`
    - Neither → `'None'`
- Write concrete values (int 0/1 and strings), not Excel formulas.

### Summary sheet
- Headers: `Employee ID`, `Week Ending`, `Break Deficits`, `Approval Gaps`, `Total Errors`.
- Group `Formatted Data` rows by `(Employee ID, Week Ending)`.
- Include only groups where sum of `Total Errors > 0`.
- Sort by Employee ID ascending then Week Ending ascending.
- Append a Grand Total row: Employee ID = `'Grand Total'`, Week Ending = `'-'`, remaining cols = dataset-wide sums of Break Deficits, Approval Gaps, Total Errors.
- `Week Ending` values in this sheet must also be YYYY-MM-DD strings.

### Word document
- Create `/root/Timesheet_Compliance_Brief.docx` using `python-docx`.
- Title paragraph: "Timesheet Compliance Brief"
- Body: 3-6 sentence executive summary that:
  - Defines both checks in plain language.
  - States the computed grand totals for Break Deficits, Approval Gaps, and Total Errors.
  - Names at least two Employee IDs with the most Total Errors as high-priority.
  - Gives at least one actionable recommendation.

## Step 3 – Validate outputs
After generating, run validation:
```python
import openpyxl
wb = openpyxl.load_workbook('/root/Timesheet_Compliance_Audit.xlsx')
print('Sheets:', wb.sheetnames)

# Check Formatted Data date types and values
ws = wb['Formatted Data']
for i, row in enumerate(ws.iter_rows(values_only=True)):
    if i == 0:
        print('Headers:', row)
    elif i <= 3:
        print('Row', i, ':', row)
        val = row[0]
        print(f'  Week Ending type={type(val).__name__}, value={val!r}')
    else:
        break

# Check Summary
ws2 = wb['Summary']
for row in ws2.iter_rows(values_only=True):
    print(row)

wb.close()

# Check Word doc exists
import os
print('DOCX exists:', os.path.exists('/root/Timesheet_Compliance_Brief.docx'))
```

Confirm:
- `Week Ending` values in Formatted Data are **str** type like `'2026-02-07'`.
- `Week Ending` values in Summary are **str** type like `'2026-02-07'`.
- Summary last row has Employee ID = `'Grand Total'`.
- Sheet names are exactly `['RawData', 'Formatted Data', 'Summary']`.
- Both output files exist at the correct paths.

If any check fails, fix and re-run before finishing.

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