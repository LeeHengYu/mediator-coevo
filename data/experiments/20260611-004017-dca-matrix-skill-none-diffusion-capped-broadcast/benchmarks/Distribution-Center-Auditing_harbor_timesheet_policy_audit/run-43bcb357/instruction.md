# Task Instruction

Execute the following Python script to produce both deliverables. The script reads the source workbook, computes compliance flags using the BreakRules thresholds, builds the three-sheet audit Excel file, and writes the Word summary.

Steps:
1. First, inspect the source file to understand its structure:
```bash
cd /root && python3 -c "
import openpyxl
wb = openpyxl.load_workbook('Timesheet_Submissions.xlsx')
for name in wb.sheetnames:
    ws = wb[name]
    print(f'=== {name} ===')
    for i, row in enumerate(ws.iter_rows(values_only=True)):
        print(row)
        if i > 5:
            print('...')
            break
"
```

2. Then run the main generation script:
```python
import pandas as pd
import numpy as np
from openpyxl import load_workbook
from docx import Document

# ── Read source data ──
entries = pd.read_excel('/root/Timesheet_Submissions.xlsx', sheet_name='Entries')
break_rules = pd.read_excel('/root/Timesheet_Submissions.xlsx', sheet_name='BreakRules')

print('Entries columns:', list(entries.columns))
print('Entries shape:', entries.shape)
print('Entries head:')
print(entries.head())
print('BreakRules columns:', list(break_rules.columns))
print('BreakRules:')
print(break_rules)
print('Entries dtypes:')
print(entries.dtypes)

# ── RawData: exact copy ──
raw_data = entries.copy()

# ── Formatted Data ──
formatted = entries.copy()

# Merge with BreakRules on Role
formatted = formatted.merge(break_rules, on='Role', how='left')

# Compute flags
formatted['Break Deficit'] = (formatted['Break Minutes'] < formatted['Min Break Minutes']).astype(int)

# Approval Missing: Hours > Overtime Threshold AND Approval Code is blank
def is_blank(val):
    if val is None:
        return True
    if isinstance(val, float) and np.isnan(val):
        return True
    if isinstance(val, str) and val.strip() == '':
        return True
    return False

formatted['Approval Missing'] = formatted.apply(
    lambda r: 1 if r['Hours Worked'] > r['Overtime Threshold'] and is_blank(r['Approval Code']) else 0,
    axis=1
)

formatted['Total Errors'] = formatted['Break Deficit'] + formatted['Approval Missing']

def error_summary(row):
    parts = []
    if row['Break Deficit'] == 1:
        parts.append('Break Deficit')
    if row['Approval Missing'] == 1:
        parts.append('Approval Missing')
    return ', '.join(parts) if parts else 'None'

formatted['Error Summary'] = formatted.apply(error_summary, axis=1)

# Keep only the 12 required columns in order
first_8 = ['Week Ending', 'Employee ID', 'Role', 'Hours Worked', 'Break Minutes',
           'Approval Code', 'Project Code', 'Manager']
new_4 = ['Break Deficit', 'Approval Missing', 'Total Errors', 'Error Summary']
formatted_out = formatted[first_8 + new_4].copy()

print('\nFormatted Data head:')
print(formatted_out.head(10))

# ── Summary ──
agg = formatted[['Employee ID', 'Week Ending', 'Break Deficit', 'Approval Missing', 'Total Errors']].copy()
grouped = agg.groupby(['Employee ID', 'Week Ending'], sort=False).agg(
    **{'Break Deficits': ('Break Deficit', 'sum'),
       'Approval Gaps': ('Approval Missing', 'sum'),
       'Total Errors': ('Total Errors', 'sum')}
).reset_index()

# Filter to groups with Total Errors > 0
grouped = grouped[grouped['Total Errors'] > 0].copy()

# Sort by Employee ID asc, then Week Ending asc
grouped = grouped.sort_values(['Employee ID', 'Week Ending']).reset_index(drop=True)

# Grand Total row
grand = pd.DataFrame([{
    'Employee ID': 'Grand Total',
    'Week Ending': '-',
    'Break Deficits': grouped['Break Deficits'].sum(),
    'Approval Gaps': grouped['Approval Gaps'].sum(),
    'Total Errors': grouped['Total Errors'].sum()
}])
summary = pd.concat([grouped, grand], ignore_index=True)

print('\nSummary:')
print(summary)

total_break_deficits = int(grand['Break Deficits'].iloc[0])
total_approval_gaps = int(grand['Approval Gaps'].iloc[0])
total_errors = int(grand['Total Errors'].iloc[0])

# ── Write Excel ──
# Important: fill NaN with empty string for RawData and formatted to avoid
# None vs string mismatches. But be careful: we want to preserve the original
# data as-is. Let's check what the source actually has.
# For safety, we will NOT fill NaN in RawData (to keep it exact copy).
# Actually, let's be more careful: read the raw values and preserve them.

outpath = '/root/Timesheet_Compliance_Audit.xlsx'
with pd.ExcelWriter(outpath, engine='openpyxl') as writer:
    raw_data.to_excel(writer, sheet_name='RawData', index=False)
    formatted_out.to_excel(writer, sheet_name='Formatted Data', index=False)
    summary.to_excel(writer, sheet_name='Summary', index=False)

print(f'\nWrote {outpath}')

# Verify sheets
wb = load_workbook(outpath)
print('Sheets:', wb.sheetnames)
for sn in wb.sheetnames:
    ws = wb[sn]
    print(f'  {sn}: {ws.max_row} rows x {ws.max_column} cols')
    # Print header
    headers = [ws.cell(1, c).value for c in range(1, ws.max_column+1)]
    print(f'    Headers: {headers}')
wb.close()

# ── Word Document ──
# Find top employees by total errors
emp_errors = formatted.groupby('Employee ID')['Total Errors'].sum().sort_values(ascending=False)
top_emps = emp_errors[emp_errors > 0].head(2).index.tolist()
print(f'\nTop employees: {top_emps}')

doc = Document()
doc.add_heading('Timesheet Compliance Brief', level=1)

para_text = (
    f'This report summarizes the results of the weekly timesheet compliance audit. '
    f'Two checks were applied to each submission: "Break Deficit" flags entries where '
    f'the recorded break minutes fall below the minimum required for the employee\'s role, '
    f'and "Approval Missing" flags entries where hours worked exceed the overtime threshold '
    f'for the role but no approval code was provided. '
    f'Across all submissions, the audit identified {total_break_deficits} Break Deficit(s), '
    f'{total_approval_gaps} Approval Gap(s), and {total_errors} Total Error(s). '
    f'Employees {top_emps[0]} and {top_emps[1]} were identified as high-priority cases '
    f'due to their frequent exceptions. '
    f'It is recommended that managers enforce mandatory break compliance training for '
    f'flagged roles and implement automated overtime-approval workflows to prevent '
    f'future gaps.'
)
doc.add_paragraph(para_text)

docpath = '/root/Timesheet_Compliance_Brief.docx'
doc.save(docpath)
print(f'Wrote {docpath}')
```

3. After running, verify both files exist and spot-check:
```bash
ls -la /root/Timesheet_Compliance_Audit.xlsx /root/Timesheet_Compliance_Brief.docx
python3 -c "
import openpyxl
wb = openpyxl.load_workbook('/root/Timesheet_Compliance_Audit.xlsx')
print('Sheets:', wb.sheetnames)
for sn in wb.sheetnames:
    ws = wb[sn]
    print(f'{sn}: rows={ws.max_row}, cols={ws.max_column}')
    for r in range(1, min(ws.max_row+1, 4)):
        print([ws.cell(r, c).value for c in range(1, ws.max_column+1)])
    if sn == 'Summary':
        last = ws.max_row
        print('Last row:', [ws.cell(last, c).value for c in range(1, ws.max_column+1)])
wb.close()
"
python3 -c "
from docx import Document
doc = Document('/root/Timesheet_Compliance_Brief.docx')
for p in doc.paragraphs:
    print(p.text)
"
```

Key points to watch:
- The first 8 columns of Formatted Data must exactly match the Entries columns in name and order.
- Break Deficit and Approval Missing must be computed using merged BreakRules thresholds, not hardcoded values.
- Approval Code blank check must handle NaN/None/empty string.
- Summary must only include groups with Total Errors > 0.
- Summary Grand Total row must have 'Grand Total' in Employee ID and '-' in Week Ending.
- Error Summary must use exact strings: 'None', 'Break Deficit', 'Approval Missing', or 'Break Deficit, Approval Missing'.
- The Word doc must mention both check definitions, the three totals, at least two employee IDs, and a recommendation.

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