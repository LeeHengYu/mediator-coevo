# Task Instruction

Execute the following Python script in `/root/` to produce both deliverables.

```python
import openpyxl
from openpyxl import Workbook
from docx import Document
from collections import defaultdict

# ── 1. Read source workbook ──────────────────────────────────────────────
src = openpyxl.load_workbook('/root/Timesheet_Submissions.xlsx', data_only=True)

# Read BreakRules sheet → dict keyed by Role
br_sheet = src['BreakRules']
br_headers = [c.value for c in br_sheet[1]]
break_rules = {}
for row in br_sheet.iter_rows(min_row=2, values_only=True):
    d = dict(zip(br_headers, row))
    role = d.get('Role') or d.get('role')
    break_rules[role] = d
print('BreakRules headers:', br_headers)
print('BreakRules data:', break_rules)

# Read Entries sheet
en_sheet = src['Entries']
en_headers = [c.value for c in en_sheet[1]]
en_rows = []
for row in en_sheet.iter_rows(min_row=2, values_only=True):
    en_rows.append(list(row))
print('Entries headers:', en_headers)
print('Entries row count:', len(en_rows))
print('First 3 rows:', en_rows[:3])

# Build column index map for Entries
col = {h: i for i, h in enumerate(en_headers)}
print('Column map:', col)

# ── 2. Identify BreakRules column names dynamically ─────────────────────
# We need 'Min Break Minutes' and 'Overtime Threshold' (or similar names)
min_break_key = None
ot_key = None
for k in br_headers:
    if k and 'min' in k.lower() and 'break' in k.lower():
        min_break_key = k
    if k and ('overtime' in k.lower() or 'threshold' in k.lower()):
        ot_key = k
print(f'Min break key: {min_break_key}, OT key: {ot_key}')

# ── 3. Build output workbook ─────────────────────────────────────────────
wb = Workbook()

# --- RawData sheet ---
ws_raw = wb.active
ws_raw.title = 'RawData'
ws_raw.append(en_headers)
for r in en_rows:
    ws_raw.append(r)

# --- Formatted Data sheet ---
ws_fmt = wb.create_sheet('Formatted Data')
fmt_headers = list(en_headers[:8]) + ['Break Deficit', 'Approval Missing', 'Total Errors', 'Error Summary']
ws_fmt.append(fmt_headers)

formatted_rows = []  # store dicts for summary
for r in en_rows:
    base = list(r[:8])  # first 8 columns preserved exactly
    role = r[col['Role']]
    hours = r[col['Hours Worked']]
    brk = r[col['Break Minutes']]
    approval = r[col['Approval Code']]

    rule = break_rules.get(role, {})
    min_brk = rule.get(min_break_key, 0)
    ot_thresh = rule.get(ot_key, 9999)

    # Convert to float safely
    try:
        hours_f = float(hours) if hours is not None else 0.0
    except (ValueError, TypeError):
        hours_f = 0.0
    try:
        brk_f = float(brk) if brk is not None else 0.0
    except (ValueError, TypeError):
        brk_f = 0.0
    try:
        min_brk_f = float(min_brk) if min_brk is not None else 0.0
    except (ValueError, TypeError):
        min_brk_f = 0.0
    try:
        ot_f = float(ot_thresh) if ot_thresh is not None else 9999.0
    except (ValueError, TypeError):
        ot_f = 9999.0

    break_deficit = 1 if brk_f < min_brk_f else 0

    # Approval Missing: hours > OT threshold AND approval code is blank
    approval_blank = (approval is None or str(approval).strip() == '')
    approval_missing = 1 if (hours_f > ot_f and approval_blank) else 0

    total_errors = break_deficit + approval_missing

    parts = []
    if break_deficit:
        parts.append('Break Deficit')
    if approval_missing:
        parts.append('Approval Missing')
    error_summary = ', '.join(parts) if parts else 'None'

    out_row = base + [break_deficit, approval_missing, total_errors, error_summary]
    ws_fmt.append(out_row)

    formatted_rows.append({
        'Employee ID': r[col['Employee ID']],
        'Week Ending': r[col['Week Ending']],
        'Break Deficit': break_deficit,
        'Approval Missing': approval_missing,
        'Total Errors': total_errors,
    })

# --- Summary sheet ---
ws_sum = wb.create_sheet('Summary')
sum_headers = ['Employee ID', 'Week Ending', 'Break Deficits', 'Approval Gaps', 'Total Errors']
ws_sum.append(sum_headers)

# Aggregate by (Employee ID, Week Ending)
agg = defaultdict(lambda: [0, 0, 0])
for fr in formatted_rows:
    key = (fr['Employee ID'], fr['Week Ending'])
    agg[key][0] += fr['Break Deficit']
    agg[key][1] += fr['Approval Missing']
    agg[key][2] += fr['Total Errors']

# Filter to groups with Total Errors > 0, sort by Employee ID asc then Week Ending asc
filtered = [(k, v) for k, v in agg.items() if v[2] > 0]
filtered.sort(key=lambda x: (str(x[0][0]), str(x[0][1])))

grand_bd = 0
grand_ag = 0
grand_te = 0
for (eid, we), (bd, ag, te) in filtered:
    ws_sum.append([eid, we, bd, ag, te])
    grand_bd += bd
    grand_ag += ag
    grand_te += te

ws_sum.append(['Grand Total', '-', grand_bd, grand_ag, grand_te])

wb.save('/root/Timesheet_Compliance_Audit.xlsx')
print('Saved Timesheet_Compliance_Audit.xlsx')
print(f'Grand totals: Break Deficits={grand_bd}, Approval Gaps={grand_ag}, Total Errors={grand_te}')

# ── 4. Find top-2 employees by total errors ──────────────────────────────
emp_errors = defaultdict(int)
for fr in formatted_rows:
    emp_errors[fr['Employee ID']] += fr['Total Errors']
top2 = sorted(emp_errors.items(), key=lambda x: -x[1])[:2]
print('Top 2 employees:', top2)

# ── 5. Build Word brief ──────────────────────────────────────────────────
doc = Document()
doc.add_heading('Timesheet Compliance Brief', level=1)

lines = []
lines.append(
    'This executive summary presents the results of the weekly timesheet compliance audit. '
    'Two primary checks were applied: "Break Deficit" flags any entry where the recorded break '
    'minutes fall below the minimum break threshold defined for the employee\'s role, and '
    '"Approval Missing" flags any entry where hours worked exceed the overtime threshold for '
    'the role yet no approval code has been recorded.'
)
lines.append(
    f'Across all submissions, the audit identified {grand_bd} Break Deficit(s), '
    f'{grand_ag} Approval Gap(s), and {grand_te} Total Error(s).'
)
lines.append(
    f'Employees {top2[0][0]} ({top2[0][1]} errors) and {top2[1][0]} ({top2[1][1]} errors) '
    'are flagged as high-priority due to frequent exceptions and should receive targeted coaching.'
)
lines.append(
    'We recommend implementing automated pre-submission validation that blocks timesheet entries '
    'with insufficient break time or missing overtime approvals, and scheduling refresher training '
    'for the identified high-priority employees.'
)

doc.add_paragraph(' '.join(lines))
doc.save('/root/Timesheet_Compliance_Brief.docx')
print('Saved Timesheet_Compliance_Brief.docx')
```

After running the script, verify:
1. `/root/Timesheet_Compliance_Audit.xlsx` exists and has sheets `RawData`, `Formatted Data`, `Summary`.
2. `/root/Timesheet_Compliance_Brief.docx` exists.
3. Print the first few rows of each sheet to confirm correctness.

```python
import openpyxl
wb = openpyxl.load_workbook('/root/Timesheet_Compliance_Audit.xlsx')
print('Sheets:', wb.sheetnames)
for sn in wb.sheetnames:
    ws = wb[sn]
    print(f'\n--- {sn} ---')
    for i, row in enumerate(ws.iter_rows(values_only=True)):
        print(row)
        if i >= 5:
            print('...')
            break
    print(f'Total rows: {ws.max_row}')

from docx import Document
doc = Document('/root/Timesheet_Compliance_Brief.docx')
for p in doc.paragraphs:
    print(p.text)
```

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