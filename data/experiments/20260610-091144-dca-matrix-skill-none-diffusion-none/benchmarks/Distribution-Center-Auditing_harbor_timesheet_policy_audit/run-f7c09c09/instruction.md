# Task Instruction

Execute the following Python script to produce both deliverables. The script reads the source workbook, applies business rules from the BreakRules sheet, builds the three-sheet audit Excel file, and generates the Word brief.

```python
import openpyxl
from openpyxl import Workbook
from docx import Document

# ── 1. Read source data ──────────────────────────────────────────────
src = openpyxl.load_workbook('/root/Timesheet_Submissions.xlsx', data_only=True)

# Read Entries sheet
entries_ws = src['Entries']
entries_headers = [c.value for c in next(entries_ws.iter_rows(min_row=1, max_row=1))]
entries_rows = []
for row in entries_ws.iter_rows(min_row=2, values_only=True):
    entries_rows.append(list(row))

print(f'Entries headers: {entries_headers}')
print(f'Entries row count: {len(entries_rows)}')
if entries_rows:
    print(f'Sample row: {entries_rows[0]}')

# Read BreakRules sheet
br_ws = src['BreakRules']
br_headers = [c.value for c in next(br_ws.iter_rows(min_row=1, max_row=1))]
br_rows = []
for row in br_ws.iter_rows(min_row=2, values_only=True):
    br_rows.append(list(row))

print(f'BreakRules headers: {br_headers}')
for r in br_rows:
    print(f'  {r}')

src.close()

# ── 2. Build lookup from BreakRules ──────────────────────────────────
# Identify column indices in BreakRules
br_role_idx = None
br_min_break_idx = None
br_ot_thresh_idx = None
for i, h in enumerate(br_headers):
    hl = str(h).strip().lower() if h else ''
    if 'role' in hl:
        br_role_idx = i
    if 'min' in hl and 'break' in hl:
        br_min_break_idx = i
    if 'overtime' in hl or ('threshold' in hl and 'over' in hl) or 'ot' in hl.replace(' ',''):
        br_ot_thresh_idx = i
    # fallback: if header contains 'threshold' but not 'break'
    if br_ot_thresh_idx is None and 'threshold' in hl:
        br_ot_thresh_idx = i

print(f'BreakRules indices -> role:{br_role_idx}, min_break:{br_min_break_idx}, ot_thresh:{br_ot_thresh_idx}')

rule_map = {}  # role_str -> {min_break: float, ot_thresh: float}
for r in br_rows:
    role_val = str(r[br_role_idx]).strip()
    min_brk = float(r[br_min_break_idx]) if r[br_min_break_idx] is not None else 0.0
    ot_thr = float(r[br_ot_thresh_idx]) if r[br_ot_thresh_idx] is not None else 999999.0
    rule_map[role_val] = {'min_break': min_brk, 'ot_thresh': ot_thr}

print(f'Rule map: {rule_map}')

# ── 3. Identify column indices in Entries ────────────────────────────
# Expected first 8 columns: Week Ending, Employee ID, Role, Hours Worked,
# Break Minutes, Approval Code, Project Code, Manager
col = {}
for i, h in enumerate(entries_headers):
    hl = str(h).strip().lower() if h else ''
    if 'week' in hl and 'end' in hl:
        col['week_ending'] = i
    elif 'employee' in hl or 'emp' in hl:
        col['employee_id'] = i
    elif hl == 'role':
        col['role'] = i
    elif 'hours' in hl and 'work' in hl:
        col['hours_worked'] = i
    elif 'break' in hl and 'min' in hl:
        col['break_minutes'] = i
    elif 'approval' in hl:
        col['approval_code'] = i
    elif 'project' in hl:
        col['project_code'] = i
    elif 'manager' in hl:
        col['manager'] = i

print(f'Entries column map: {col}')

# ── 4. Build Formatted Data rows ────────────────────────────────────
formatted_rows = []  # each: first8 + [break_deficit, approval_missing, total_errors, error_summary]
for row in entries_rows:
    role_val = str(row[col['role']]).strip() if row[col['role']] is not None else ''
    hours = float(row[col['hours_worked']]) if row[col['hours_worked']] is not None else 0.0
    brk_min = float(row[col['break_minutes']]) if row[col['break_minutes']] is not None else 0.0
    approval = row[col['approval_code']]
    approval_blank = (approval is None or str(approval).strip() == '')

    rules = rule_map.get(role_val, {'min_break': 0.0, 'ot_thresh': 999999.0})

    break_deficit = 1 if brk_min < rules['min_break'] else 0
    approval_missing = 1 if (hours > rules['ot_thresh'] and approval_blank) else 0
    total_errors = break_deficit + approval_missing

    if total_errors == 0:
        error_summary = 'None'
    elif break_deficit == 1 and approval_missing == 1:
        error_summary = 'Break Deficit, Approval Missing'
    elif break_deficit == 1:
        error_summary = 'Break Deficit'
    else:
        error_summary = 'Approval Missing'

    # first 8 columns in canonical order
    first8 = [
        row[col['week_ending']],
        row[col['employee_id']],
        row[col['role']],
        row[col['hours_worked']],
        row[col['break_minutes']],
        row[col['approval_code']],
        row[col['project_code']],
        row[col['manager']],
    ]
    formatted_rows.append(first8 + [break_deficit, approval_missing, total_errors, error_summary])

formatted_headers = [
    'Week Ending', 'Employee ID', 'Role', 'Hours Worked',
    'Break Minutes', 'Approval Code', 'Project Code', 'Manager',
    'Break Deficit', 'Approval Missing', 'Total Errors', 'Error Summary'
]

# ── 5. Build Summary rows ────────────────────────────────────────────
from collections import defaultdict
agg = defaultdict(lambda: {'break_deficits': 0, 'approval_gaps': 0, 'total_errors': 0})

for fr in formatted_rows:
    emp_id = fr[1]  # Employee ID
    week_ending = fr[0]  # Week Ending
    bd = fr[8]
    am = fr[9]
    te = fr[10]
    key = (emp_id, week_ending)
    agg[key]['break_deficits'] += bd
    agg[key]['approval_gaps'] += am
    agg[key]['total_errors'] += te

# Filter and sort
summary_rows = []
for (emp, we), vals in agg.items():
    if vals['total_errors'] > 0:
        summary_rows.append([emp, we, vals['break_deficits'], vals['approval_gaps'], vals['total_errors']])

# Sort by Employee ID asc, then Week Ending asc
summary_rows.sort(key=lambda x: (str(x[0]), str(x[1])))

# Grand Total
total_bd = sum(r[2] for r in summary_rows)
total_ag = sum(r[3] for r in summary_rows)
total_te = sum(r[4] for r in summary_rows)
summary_rows.append(['Grand Total', '-', total_bd, total_ag, total_te])

summary_headers = ['Employee ID', 'Week Ending', 'Break Deficits', 'Approval Gaps', 'Total Errors']

print(f'Summary rows (including Grand Total): {len(summary_rows)}')
for sr in summary_rows:
    print(f'  {sr}')

# ── 6. Write audit Excel ─────────────────────────────────────────────
wb = Workbook()

# RawData sheet
ws_raw = wb.active
ws_raw.title = 'RawData'
ws_raw.append(entries_headers)
for row in entries_rows:
    ws_raw.append(row)

# Formatted Data sheet
ws_fmt = wb.create_sheet('Formatted Data')
ws_fmt.append(formatted_headers)
for row in formatted_rows:
    ws_fmt.append(row)

# Summary sheet
ws_sum = wb.create_sheet('Summary')
ws_sum.append(summary_headers)
for row in summary_rows:
    ws_sum.append(row)

wb.save('/root/Timesheet_Compliance_Audit.xlsx')
print('Saved /root/Timesheet_Compliance_Audit.xlsx')

# ── 7. Identify top-2 employee IDs with most total errors ───────────
emp_errors = defaultdict(int)
for sr in summary_rows[:-1]:  # exclude Grand Total
    emp_errors[sr[0]] += sr[4]

top_emps = sorted(emp_errors.items(), key=lambda x: -x[1])[:2]
top_emp_ids = [str(e[0]) for e in top_emps]
print(f'Top employees by errors: {top_emps}')

# ── 8. Write Word brief ──────────────────────────────────────────────
doc = Document()
doc.add_heading('Timesheet Compliance Brief', level=1)

p1 = (f'This report summarizes the results of the weekly timesheet compliance audit. '
      f'Two checks were applied to every submission: "Break Deficit" flags entries where the '
      f'recorded break minutes fall below the minimum required for the employee\'s role as '
      f'defined in the BreakRules policy table, and "Approval Missing" flags entries where '
      f'hours worked exceed the overtime threshold for the role yet no Approval Code was provided.')

p2 = (f'Across all submissions, the audit identified {total_bd} Break Deficit(s), '
      f'{total_ag} Approval Gap(s), and {total_te} Total Error(s).')

p3 = (f'Employees {" and ".join(top_emp_ids)} were the highest-priority cases, each showing '
      f'frequent exceptions that warrant immediate supervisory review.')

p4 = (f'We recommend that managers conduct a targeted review of overtime-approval workflows '
      f'and ensure break-scheduling compliance training is completed for all flagged roles '
      f'before the next reporting cycle.')

doc.add_paragraph(p1)
doc.add_paragraph(p2)
doc.add_paragraph(p3)
doc.add_paragraph(p4)

doc.save('/root/Timesheet_Compliance_Brief.docx')
print('Saved /root/Timesheet_Compliance_Brief.docx')
print('Done.')
```

After running the script, verify both output files exist and that the Excel workbook contains sheets named exactly `RawData`, `Formatted Data`, and `Summary`, and that the Word document contains the required summary content.

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