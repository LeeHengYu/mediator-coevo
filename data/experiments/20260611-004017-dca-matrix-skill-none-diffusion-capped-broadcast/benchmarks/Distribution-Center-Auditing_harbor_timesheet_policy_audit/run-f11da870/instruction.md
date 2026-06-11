# Task Instruction

Execute the following steps in order.

## Step 0 – Inspect the source workbook
```bash
cd /root
python3 - <<'PY'
import openpyxl
wb = openpyxl.load_workbook('Timesheet_Submissions.xlsx', data_only=True)
for name in wb.sheetnames:
    ws = wb[name]
    print(f'\n=== Sheet: {name}  rows={ws.max_row}  cols={ws.max_column} ===')
    for i, row in enumerate(ws.iter_rows(min_row=1, max_row=min(6, ws.max_row), values_only=False), 1):
        print([c.value for c in row])
wb.close()
PY
```
Record the exact header names in `Entries` and `BreakRules`, their column order, data types, and sample values. You will need these to map correctly.

## Step 1 – Build the audit workbook and Word brief

Write and run a single Python script (`/root/build_audit.py`) that does everything below.

### 1-A  Read source data
```python
import pandas as pd
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
from docx import Document

entries = pd.read_excel('Timesheet_Submissions.xlsx', sheet_name='Entries')
break_rules = pd.read_excel('Timesheet_Submissions.xlsx', sheet_name='BreakRules')

print('Entries columns:', list(entries.columns))
print('BreakRules columns:', list(break_rules.columns))
print(break_rules)
```

### 1-B  Build `RawData`
Copy the `Entries` dataframe as-is into a sheet named exactly `RawData`.

### 1-C  Build `Formatted Data`
1. Start from a copy of the Entries dataframe.
2. Map the first 8 columns to exactly these names (rename if the source names differ):
   `Week Ending`, `Employee ID`, `Role`, `Hours Worked`, `Break Minutes`, `Approval Code`, `Project Code`, `Manager`.
   **Use the column order from the source; match by semantics if names differ.**
3. Merge with `BreakRules` on `Role` to get `Min Break Minutes` and `Overtime Threshold` for each row.
4. Compute:
   - `Break Deficit` = 1 if `Break Minutes` < `Min Break Minutes`, else 0  (use int)
   - `Approval Missing` = 1 if `Hours Worked` > `Overtime Threshold` AND `Approval Code` is blank/NaN, else 0  (use int)
     - Treat empty string as blank as well.
   - `Total Errors` = `Break Deficit` + `Approval Missing`  (int)
   - `Error Summary`:
     - If both flags 1 → `Break Deficit, Approval Missing`
     - If only Break Deficit → `Break Deficit`
     - If only Approval Missing → `Approval Missing`
     - If neither → `None`
5. Keep only the 12 required columns in the exact order listed above.
6. Write concrete values (no Excel formulas).

### 1-D  Build `Summary`
1. Group `Formatted Data` by (`Employee ID`, `Week Ending`).
2. Sum `Break Deficit` → `Break Deficits`, `Approval Missing` → `Approval Gaps`, `Total Errors` → `Total Errors`.
3. Keep only groups where `Total Errors > 0`.
4. Sort by `Employee ID` ascending then `Week Ending` ascending.
5. Append a Grand Total row: `Employee ID` = `Grand Total`, `Week Ending` = `-`, sums of the three numeric columns.
6. Headers exactly: `Employee ID`, `Week Ending`, `Break Deficits`, `Approval Gaps`, `Total Errors`.

### 1-E  Write `/root/Timesheet_Compliance_Audit.xlsx`
Use openpyxl to write the three sheets in order: `RawData`, `Formatted Data`, `Summary`. Make sure sheet names are exactly as specified. Write header rows. Convert any datetime/Timestamp to plain date or string before writing.

### 1-F  Write `/root/Timesheet_Compliance_Brief.docx`
1. Compute grand totals for Break Deficits, Approval Gaps, Total Errors from the Summary grand-total row.
2. Identify the two Employee IDs with the highest Total Errors (from the Summary, excluding Grand Total).
3. Write a 3-6 sentence executive summary paragraph that:
   - Defines Break Deficit: a shift where break minutes fell below the role-required minimum.
   - Defines Approval Missing: overtime hours logged without a required approval code.
   - States the computed totals.
   - Names at least two high-priority employee IDs.
   - Gives at least one actionable recommendation (e.g., enforce approval workflows, train managers).
4. Save as `/root/Timesheet_Compliance_Brief.docx`.

## Step 2 – Validate outputs
```bash
python3 - <<'PY'
import openpyxl, os
from docx import Document

# Check Excel
assert os.path.exists('/root/Timesheet_Compliance_Audit.xlsx')
wb = openpyxl.load_workbook('/root/Timesheet_Compliance_Audit.xlsx')
assert wb.sheetnames == ['RawData', 'Formatted Data', 'Summary'], f'Sheets: {wb.sheetnames}'

ws = wb['Formatted Data']
headers = [c.value for c in next(ws.iter_rows(min_row=1, max_row=1))]
expected = ['Week Ending','Employee ID','Role','Hours Worked','Break Minutes',
            'Approval Code','Project Code','Manager',
            'Break Deficit','Approval Missing','Total Errors','Error Summary']
assert headers == expected, f'FD headers: {headers}'

ws2 = wb['Summary']
sh = [c.value for c in next(ws2.iter_rows(min_row=1, max_row=1))]
assert sh == ['Employee ID','Week Ending','Break Deficits','Approval Gaps','Total Errors'], f'Sum headers: {sh}'

# Last row should be Grand Total
last = [c.value for c in list(ws2.iter_rows())[-1]]
assert last[0] == 'Grand Total', f'Last row col0: {last[0]}'
assert last[1] == '-', f'Last row col1: {last[1]}'

print('Excel OK')

# Check Word
assert os.path.exists('/root/Timesheet_Compliance_Brief.docx')
doc = Document('/root/Timesheet_Compliance_Brief.docx')
text = ' '.join(p.text for p in doc.paragraphs)
assert 'Break Deficit' in text
assert 'Approval Missing' in text or 'approval' in text.lower()
print('Word OK')
print('All validations passed.')
PY
```

If any validation fails, diagnose and fix before finishing. Do NOT skip the validation step.

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