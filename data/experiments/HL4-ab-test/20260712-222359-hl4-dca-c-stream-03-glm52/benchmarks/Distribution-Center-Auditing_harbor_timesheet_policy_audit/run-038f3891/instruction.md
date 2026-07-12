# Task Instruction

Execute the following Python script to produce both deliverables. Before writing the script, inspect the source workbook to understand its structure.

## Step 1: Inspect the source data
```bash
cd /root
python3 -c "
import openpyxl
wb = openpyxl.load_workbook('Timesheet_Submissions.xlsx')
print('Sheet names:', wb.sheetnames)
for sn in wb.sheetnames:
    ws = wb[sn]
    print(f'\n=== {sn} ===')
    for i, row in enumerate(ws.iter_rows(values_only=True)):
        print(row)
        if i >= 5:
            print('...')
            break
    print(f'Total rows (incl header): {ws.max_row}, cols: {ws.max_column}')
"
```

## Step 2: After inspecting the output, write and run the full solution script

Create `/root/solve.py` with the following logic:

```python
import openpyxl
from openpyxl import Workbook
from docx import Document
from collections import defaultdict

# --- Load source ---
src = openpyxl.load_workbook('/root/Timesheet_Submissions.xlsx')

# Read BreakRules sheet
br_ws = src['BreakRules']
br_headers = [c.value for c in next(br_ws.iter_rows(min_row=1, max_row=1))]
print('BreakRules headers:', br_headers)

# Build lookup: role -> {min_break_minutes, overtime_threshold}
break_rules = {}
for row in br_ws.iter_rows(min_row=2, values_only=True):
    row_dict = dict(zip(br_headers, row))
    role = row_dict.get('Role')
    # Find the min break minutes and overtime threshold columns (flexible naming)
    min_break = None
    ot_thresh = None
    for k, v in row_dict.items():
        if k and 'min' in k.lower() and 'break' in k.lower():
            min_break = v
        if k and ('overtime' in k.lower() or 'threshold' in k.lower()) and 'break' not in k.lower():
            ot_thresh = v
    break_rules[role] = {'min_break': min_break, 'ot_thresh': ot_thresh}

print('Break rules:', break_rules)

# Read Entries sheet
en_ws = src['Entries']
en_headers = [c.value for c in next(en_ws.iter_rows(min_row=1, max_row=1))]
print('Entries headers:', en_headers)

entries = []
for row in en_ws.iter_rows(min_row=2, values_only=True):
    entries.append(dict(zip(en_headers, row)))

print(f'Total entries: {len(entries)}')

# --- Determine column name mapping ---
# We need: Week Ending, Employee ID, Role, Hours Worked, Break Minutes, Approval Code, Project Code, Manager
# Map from whatever the source headers are
def find_col(headers, keywords):
    for h in headers:
        if h is None:
            continue
        hl = h.lower().replace('_', ' ')
        if all(k in hl for k in keywords):
            return h
    return None

col_week = find_col(en_headers, ['week'])
col_empid = find_col(en_headers, ['employee']) or find_col(en_headers, ['emp'])
col_role = find_col(en_headers, ['role'])
col_hours = find_col(en_headers, ['hours'])
col_break = find_col(en_headers, ['break'])
col_approval = find_col(en_headers, ['approval'])
col_project = find_col(en_headers, ['project'])
col_manager = find_col(en_headers, ['manager'])

print('Column mapping:', col_week, col_empid, col_role, col_hours, col_break, col_approval, col_project, col_manager)

# --- Build output workbook ---
out = Workbook()

# Sheet 1: RawData - exact copy of Entries
raw_ws = out.active
raw_ws.title = 'RawData'
for row in en_ws.iter_rows(min_row=1, values_only=True):
    raw_ws.append(list(row))

# Sheet 2: Formatted Data
fmt_ws = out.create_sheet('Formatted Data')
fmt_headers = ['Week Ending', 'Employee ID', 'Role', 'Hours Worked', 'Break Minutes',
               'Approval Code', 'Project Code', 'Manager',
               'Break Deficit', 'Approval Missing', 'Total Errors', 'Error Summary']
fmt_ws.append(fmt_headers)

formatted_rows = []
for e in entries:
    week = e[col_week]
    empid = e[col_empid]
    role = e[col_role]
    hours = e[col_hours] if e[col_hours] is not None else 0
    brk = e[col_break] if e[col_break] is not None else 0
    approval = e[col_approval]
    project = e[col_project]
    manager = e[col_manager]

    rule = break_rules.get(role, {})
    min_brk = rule.get('min_break', 0) or 0
    ot_thr = rule.get('ot_thresh', 9999) or 9999

    break_deficit = 1 if brk < min_brk else 0
    approval_missing = 1 if (hours > ot_thr and (approval is None or str(approval).strip() == '')) else 0
    total_errors = break_deficit + approval_missing

    if total_errors == 0:
        err_summary = 'None'
    elif break_deficit == 1 and approval_missing == 1:
        err_summary = 'Break Deficit, Approval Missing'
    elif break_deficit == 1:
        err_summary = 'Break Deficit'
    else:
        err_summary = 'Approval Missing'

    row_out = [week, empid, role, hours, brk, approval, project, manager,
               break_deficit, approval_missing, total_errors, err_summary]
    fmt_ws.append(row_out)
    formatted_rows.append(row_out)

# Sheet 3: Summary
sum_ws = out.create_sheet('Summary')
sum_headers = ['Employee ID', 'Week Ending', 'Break Deficits', 'Approval Gaps', 'Total Errors']
sum_ws.append(sum_headers)

# Aggregate by (Employee ID, Week Ending)
agg = defaultdict(lambda: {'bd': 0, 'am': 0, 'te': 0})
for r in formatted_rows:
    key = (r[1], r[0])  # (Employee ID, Week Ending)
    agg[key]['bd'] += r[8]
    agg[key]['am'] += r[9]
    agg[key]['te'] += r[10]

# Filter and sort
filtered = [(k, v) for k, v in agg.items() if v['te'] > 0]
filtered.sort(key=lambda x: (str(x[0][0]), str(x[0][1]) if x[0][1] is not None else ''))

grand_bd = 0
grand_am = 0
grand_te = 0
for (empid, week), vals in filtered:
    sum_ws.append([empid, week, vals['bd'], vals['am'], vals['te']])
    grand_bd += vals['bd']
    grand_am += vals['am']
    grand_te += vals['te']

sum_ws.append(['Grand Total', '-', grand_bd, grand_am, grand_te])

out.save('/root/Timesheet_Compliance_Audit.xlsx')
print('Excel saved.')

# --- Word Document ---
# Find top employee IDs by total errors
emp_errors = defaultdict(int)
for (empid, week), vals in agg.items():
    emp_errors[empid] += vals['te']

top_emps = sorted(emp_errors.items(), key=lambda x: -x[1])
# Pick at least 2 with errors
high_priority = [e for e, c in top_emps if c > 0][:max(2, 2)]

doc = Document()
doc.add_heading('Timesheet Compliance Brief', level=1)

# Build the paragraph
hp_str = ', '.join(str(e) for e in high_priority)

paragraph = (
    f'This audit reviewed weekly consultant timesheets against two compliance checks: '
    f'Break Deficit flags entries where the reported break minutes fall below the minimum '
    f'required for the employee\'s role, and Approval Missing flags entries where hours '
    f'worked exceed the overtime threshold for the role but no approval code was provided. '
    f'Across all submissions, the audit identified {grand_bd} Break Deficit(s), '
    f'{grand_am} Approval Gap(s), and {grand_te} Total Error(s). '
    f'High-priority employees with frequent exceptions include {hp_str}. '
    f'It is recommended that managers enforce mandatory break logging and ensure overtime '
    f'approval codes are submitted before timesheet finalization to reduce recurring violations.'
)

doc.add_paragraph(paragraph)
doc.save('/root/Timesheet_Compliance_Brief.docx')
print('Word doc saved.')
print('Done.')
```

Run the script:
```bash
pip install openpyxl python-docx 2>/dev/null
python3 /root/solve.py
```

## Step 3: Validate the outputs
```bash
python3 -c "
import openpyxl
wb = openpyxl.load_workbook('/root/Timesheet_Compliance_Audit.xlsx')
print('Sheets:', wb.sheetnames)
for sn in wb.sheetnames:
    ws = wb[sn]
    print(f'\n=== {sn} ===')
    for i, row in enumerate(ws.iter_rows(values_only=True)):
        print(row)
        if i >= 5:
            print(f'... total rows: {ws.max_row}')
            break
    # Print last 3 rows
    if ws.max_row > 6:
        print('Last rows:')
        for row in ws.iter_rows(min_row=max(2, ws.max_row-2), values_only=True):
            print(row)

from docx import Document
doc = Document('/root/Timesheet_Compliance_Brief.docx')
for p in doc.paragraphs:
    print(p.text)
"
```

## Important Notes
- After Step 1 inspection, adapt column name mappings in the script if the actual header names differ from expected. The `find_col` function uses keyword matching but verify it finds the right columns.
- The `BreakRules` lookup must use actual column names from the sheet. Check the print output and adjust if needed.
- Make sure the Employee IDs mentioned in the Word document match actual IDs from the data (not invented).
- The Word doc must mention at least two employee IDs. Use the format `EMP-XXX` or whatever format appears in the data, separated by commas.
- Ensure `Error Summary` values are exactly: `None`, `Break Deficit`, `Approval Missing`, or `Break Deficit, Approval Missing`.
- The `Grand Total` row in Summary must have `Employee ID` = `Grand Total` and `Week Ending` = `-` (the string hyphen).
- Write concrete values (not Excel formulas) in columns 9-12 of Formatted Data.

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