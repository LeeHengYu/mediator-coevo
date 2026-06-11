# Task Instruction

Execute the following steps to produce `/root/Timesheet_Compliance_Audit.xlsx` and `/root/Timesheet_Compliance_Brief.docx`.

### Step 0 – Inspect the source workbook
```python
import openpyxl, pandas as pd

# Read both sheets to understand structure
entries = pd.read_excel('/root/Timesheet_Submissions.xlsx', sheet_name='Entries')
break_rules = pd.read_excel('/root/Timesheet_Submissions.xlsx', sheet_name='BreakRules')
print('Entries columns:', list(entries.columns))
print('Entries shape:', entries.shape)
print(entries.head(10).to_string())
print('\nBreakRules columns:', list(break_rules.columns))
print(break_rules.to_string())
```
Review the output carefully before proceeding. Note the exact column names in both sheets.

### Step 1 – Build the audit workbook
```python
import pandas as pd
import numpy as np
from openpyxl import Workbook
from copy import copy

# ---- Load data ----
entries = pd.read_excel('/root/Timesheet_Submissions.xlsx', sheet_name='Entries')
break_rules = pd.read_excel('/root/Timesheet_Submissions.xlsx', sheet_name='BreakRules')

# ---- RawData: exact copy ----
raw = entries.copy()

# ---- Formatted Data ----
fmt = entries.copy()

# Merge thresholds on Role
fmt = fmt.merge(break_rules[['Role', 'Min Break Minutes', 'Overtime Threshold']],
                on='Role', how='left')

# Break Deficit: 1 if Break Minutes < Min Break Minutes for that Role
fmt['Break Deficit'] = (fmt['Break Minutes'] < fmt['Min Break Minutes']).astype(int)

# Approval Missing: 1 if Hours Worked > Overtime Threshold AND Approval Code is blank/NaN
def approval_missing(row):
    over_threshold = row['Hours Worked'] > row['Overtime Threshold']
    code = row['Approval Code']
    is_blank = pd.isna(code) or str(code).strip() == ''
    return 1 if (over_threshold and is_blank) else 0

fmt['Approval Missing'] = fmt.apply(approval_missing, axis=1)

# Total Errors
fmt['Total Errors'] = fmt['Break Deficit'] + fmt['Approval Missing']

# Error Summary
def error_summary(row):
    parts = []
    if row['Break Deficit'] == 1:
        parts.append('Break Deficit')
    if row['Approval Missing'] == 1:
        parts.append('Approval Missing')
    return ', '.join(parts) if parts else 'None'

fmt['Error Summary'] = fmt.apply(error_summary, axis=1)

# Keep only the required 12 columns in order
fmt_out = fmt[['Week Ending', 'Employee ID', 'Role', 'Hours Worked',
               'Break Minutes', 'Approval Code', 'Project Code', 'Manager',
               'Break Deficit', 'Approval Missing', 'Total Errors', 'Error Summary']].copy()

# ---- Summary ----
agg = fmt_out[fmt_out['Total Errors'] > 0].groupby(
    ['Employee ID', 'Week Ending'], sort=False).agg(
    **{'Break Deficits': ('Break Deficit', 'sum'),
       'Approval Gaps': ('Approval Missing', 'sum'),
       'Total Errors': ('Total Errors', 'sum')}
).reset_index()

agg = agg.sort_values(['Employee ID', 'Week Ending']).reset_index(drop=True)

# Grand Total row
grand = pd.DataFrame([{
    'Employee ID': 'Grand Total',
    'Week Ending': '-',
    'Break Deficits': agg['Break Deficits'].sum(),
    'Approval Gaps': agg['Approval Gaps'].sum(),
    'Total Errors': agg['Total Errors'].sum()
}])
summary = pd.concat([agg, grand], ignore_index=True)

# ---- Write workbook ----
out_path = '/root/Timesheet_Compliance_Audit.xlsx'
with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
    raw.to_excel(writer, sheet_name='RawData', index=False)
    fmt_out.to_excel(writer, sheet_name='Formatted Data', index=False)
    summary.to_excel(writer, sheet_name='Summary', index=False)

print('Audit workbook written.')
print('Summary:')
print(summary.to_string())
```

### Step 2 – Build the Word brief
```python
from docx import Document

total_bd = int(summary.loc[summary['Employee ID'] == 'Grand Total', 'Break Deficits'].values[0])
total_ag = int(summary.loc[summary['Employee ID'] == 'Grand Total', 'Approval Gaps'].values[0])
total_err = int(summary.loc[summary['Employee ID'] == 'Grand Total', 'Total Errors'].values[0])

# Identify top-2 employees by total errors
emp_errors = agg.groupby('Employee ID')['Total Errors'].sum().sort_values(ascending=False)
top2 = list(emp_errors.index[:2])

doc = Document()
doc.add_heading('Timesheet Compliance Brief', level=1)

para = (
    f'This audit reviewed weekly consultant timesheets against two compliance checks. '
    f'A "Break Deficit" is flagged when an employee\'s recorded break minutes fall below '
    f'the minimum required for their role as defined in the BreakRules policy. '
    f'An "Approval Missing" flag is raised when an employee works more hours than the '
    f'overtime threshold for their role yet has no approval code on file. '
    f'Across the dataset, {total_bd} Break Deficit(s), {total_ag} Approval Gap(s), '
    f'and {total_err} Total Error(s) were identified. '
    f'Employees {top2[0]} and {top2[1]} had the highest frequency of exceptions and '
    f'should be prioritized for corrective action. '
    f'We recommend that managers enforce mandatory break logging and pre-approve all '
    f'overtime shifts before they are worked to reduce future non-compliance.'
)
doc.add_paragraph(para)
doc.save('/root/Timesheet_Compliance_Brief.docx')
print('Word brief written.')
```

### Step 3 – Validate outputs
```python
import os
for f in ['/root/Timesheet_Compliance_Audit.xlsx', '/root/Timesheet_Compliance_Brief.docx']:
    assert os.path.exists(f), f'Missing: {f}'

# Verify worksheet names
wb = openpyxl.load_workbook('/root/Timesheet_Compliance_Audit.xlsx')
assert wb.sheetnames == ['RawData', 'Formatted Data', 'Summary'], f'Unexpected sheets: {wb.sheetnames}'

# Verify Formatted Data headers
ws = wb['Formatted Data']
headers = [ws.cell(1, c).value for c in range(1, 13)]
expected = ['Week Ending', 'Employee ID', 'Role', 'Hours Worked',
            'Break Minutes', 'Approval Code', 'Project Code', 'Manager',
            'Break Deficit', 'Approval Missing', 'Total Errors', 'Error Summary']
assert headers == expected, f'Header mismatch: {headers}'

# Verify Summary headers
ws2 = wb['Summary']
sum_headers = [ws2.cell(1, c).value for c in range(1, 6)]
assert sum_headers == ['Employee ID', 'Week Ending', 'Break Deficits', 'Approval Gaps', 'Total Errors']

# Verify Grand Total row exists
last_row = ws2.max_row
assert ws2.cell(last_row, 1).value == 'Grand Total', 'Grand Total row missing'

print('All validations passed.')
```

Execute each step in order. If Step 0 reveals unexpected column names, adapt the column references in Steps 1-2 accordingly before proceeding. Do not hardcode thresholds by role name—always join on the BreakRules table.

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