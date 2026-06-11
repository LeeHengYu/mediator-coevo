# Task Instruction

Execute the following Python script in a single step to produce both deliverables. Read the script carefully — it handles all data processing, Excel generation, and Word generation.

```python
import openpyxl
from openpyxl.utils import get_column_letter
from copy import copy
import os

# ─── Helper: copy a sheet from one workbook to another ───
def copy_sheet(source_ws, target_wb, sheet_name):
    target_ws = target_wb.create_sheet(title=sheet_name)
    for row in source_ws.iter_rows():
        for cell in row:
            new_cell = target_ws.cell(row=cell.row, column=cell.column, value=cell.value)
            if cell.has_style:
                new_cell.font = copy(cell.font)
                new_cell.border = copy(cell.border)
                new_cell.fill = copy(cell.fill)
                new_cell.number_format = copy(cell.number_format)
                new_cell.protection = copy(cell.protection)
                new_cell.alignment = copy(cell.alignment)
    for mc in source_ws.merged_cells.ranges:
        target_ws.merge_cells(str(mc))
    for col_idx, col_dim in source_ws.column_dimensions.items():
        target_ws.column_dimensions[col_idx].width = col_dim.width
    for row_idx, row_dim in source_ws.row_dimensions.items():
        target_ws.row_dimensions[row_idx].height = row_dim.height
    return target_ws

# ─── Load source workbooks ───
template_wb = openpyxl.load_workbook('/root/Cycle_Template.xlsx')
plan_wb = openpyxl.load_workbook('/root/Cycle_Plan.xlsx')
event_wb = openpyxl.load_workbook('/root/Count_Event_Log.xlsx')

# ─── Parse Count_Event_Log to get latest FINAL events ───
event_ws = event_wb.active
event_headers = [cell.value for cell in next(event_ws.iter_rows(min_row=1, max_row=1))]
print('Event log headers:', event_headers)

# Find column indices
def find_col(headers, *candidates):
    for h_idx, h in enumerate(headers):
        if h is None:
            continue
        h_lower = str(h).strip().lower()
        for c in candidates:
            if c.lower() == h_lower:
                return h_idx
    return None

ev_facility_idx = find_col(event_headers, 'Facility')
ev_session_idx = find_col(event_headers, 'Session ID')
ev_bin_idx = find_col(event_headers, 'Bin ID')
ev_event_type_idx = find_col(event_headers, 'Event Type')
ev_count_qty_idx = find_col(event_headers, 'Count Qty')
ev_row_idx_col = None  # We'll use row number for ordering

print(f'Event col indices: facility={ev_facility_idx}, session={ev_session_idx}, bin={ev_bin_idx}, event_type={ev_event_type_idx}, count_qty={ev_count_qty_idx}')

# Build dict: (facility, session_id, bin_id) -> latest FINAL row's Count Qty
final_counts = {}  # key -> (count_qty, row_number)
for row_num, row in enumerate(event_ws.iter_rows(min_row=2, values_only=True), start=2):
    facility = row[ev_facility_idx]
    session_id = row[ev_session_idx]
    bin_id = row[ev_bin_idx]
    event_type = row[ev_event_type_idx]
    count_qty = row[ev_count_qty_idx]
    
    # Skip rows with blank keys or blank Count Qty
    if facility is None or session_id is None or bin_id is None:
        continue
    if event_type is None:
        continue
    if str(event_type).strip().upper() != 'FINAL':
        continue
    if count_qty is None or str(count_qty).strip() == '':
        continue
    
    key = (str(facility).strip(), str(session_id).strip(), str(bin_id).strip())
    # Keep the latest row (highest row number)
    if key not in final_counts or row_num > final_counts[key][1]:
        try:
            final_counts[key] = (float(count_qty), row_num)
        except (ValueError, TypeError):
            continue

print(f'Found {len(final_counts)} unique FINAL count entries')

# ─── Parse Cycle_Plan ───
plan_ws = plan_wb.active
plan_headers = [cell.value for cell in next(plan_ws.iter_rows(min_row=1, max_row=1))]
print('Plan headers:', plan_headers)

plan_rows = []
for row in plan_ws.iter_rows(min_row=2, values_only=True):
    plan_rows.append(list(row))

print(f'Plan has {len(plan_rows)} data rows')

# Identify plan columns
p_facility_idx = find_col(plan_headers, 'Facility')
p_session_idx = find_col(plan_headers, 'Session ID')
p_bin_idx = find_col(plan_headers, 'Bin ID')
p_product_idx = find_col(plan_headers, 'Product ID')
p_expected_idx = find_col(plan_headers, 'Expected Qty')
p_variance_idx = find_col(plan_headers, 'Allowed Variance')
p_approval_idx = find_col(plan_headers, 'Approval Needed')

print(f'Plan col indices: facility={p_facility_idx}, session={p_session_idx}, bin={p_bin_idx}, product={p_product_idx}, expected={p_expected_idx}, variance={p_variance_idx}, approval={p_approval_idx}')

# ─── Build Formatted Data rows ───
formatted_rows = []
for row in plan_rows:
    facility = row[p_facility_idx]
    session_id = row[p_session_idx]
    bin_id = row[p_bin_idx]
    product_id = row[p_product_idx]
    expected_qty = row[p_expected_idx]
    allowed_variance = row[p_variance_idx]
    approval_needed = row[p_approval_idx]
    
    key = (str(facility).strip() if facility else '', 
           str(session_id).strip() if session_id else '', 
           str(bin_id).strip() if bin_id else '')
    
    # Check for FINAL count
    has_final = key in final_counts
    count_qty = final_counts[key][0] if has_final else None
    
    # Missing Final Count
    missing_final = 0 if has_final else 1
    
    # Approval Gap
    approval_gap = 0
    if has_final:
        approval_str = str(approval_needed).strip().upper() if approval_needed else ''
        if approval_str == 'YES':
            try:
                exp = float(expected_qty)
                diff = abs(exp - count_qty)
                var = float(allowed_variance)
                if diff > var:
                    approval_gap = 1
            except (ValueError, TypeError):
                pass
    
    total_errors = missing_final + approval_gap
    
    # Error Summary
    if total_errors == 0:
        error_summary = 'None'
    elif missing_final == 1 and approval_gap == 1:
        error_summary = 'Missing Final Count, Approval Gap'
    elif missing_final == 1:
        error_summary = 'Missing Final Count'
    else:
        error_summary = 'Approval Gap'
    
    formatted_rows.append([
        facility, session_id, bin_id, product_id, expected_qty, allowed_variance, approval_needed,
        missing_final, approval_gap, total_errors, error_summary
    ])

print(f'Formatted data: {len(formatted_rows)} rows')

# ─── Build Summary data ───
from collections import defaultdict
summary_agg = defaultdict(lambda: [0, 0, 0])  # (facility, session) -> [missing, approval, total]
for frow in formatted_rows:
    fac = frow[0]
    sess = frow[1]
    key = (str(fac).strip() if fac else '', str(sess).strip() if sess else '')
    summary_agg[key][0] += frow[7]  # missing
    summary_agg[key][1] += frow[8]  # approval gap
    summary_agg[key][2] += frow[9]  # total errors

# Filter to groups with Total Errors > 0, sort by Facility asc then Session ID asc
summary_rows = []
for (fac, sess), vals in summary_agg.items():
    if vals[2] > 0:
        summary_rows.append([fac, sess, vals[0], vals[1], vals[2]])

summary_rows.sort(key=lambda x: (str(x[0]), str(x[1])))

# Grand totals from entire dataset
total_missing = sum(frow[7] for frow in formatted_rows)
total_approval = sum(frow[8] for frow in formatted_rows)
total_total = sum(frow[9] for frow in formatted_rows)

summary_rows.append(['Grand Total', '-', total_missing, total_approval, total_total])

print(f'Summary rows (incl Grand Total): {len(summary_rows)}')
print('Grand totals: Missing={}, Approval={}, Total={}'.format(total_missing, total_approval, total_total))

# ─── Create output workbook ───
out_wb = openpyxl.Workbook()
# Remove default sheet
out_wb.remove(out_wb.active)

# 1) Overview - copy from template
template_overview = template_wb['Overview']
copy_sheet(template_overview, out_wb, 'Overview')

# 2) RawData - copy plan table exactly
raw_ws = out_wb.create_sheet(title='RawData')
for row in plan_ws.iter_rows():
    for cell in row:
        new_cell = raw_ws.cell(row=cell.row, column=cell.column, value=cell.value)
        if cell.has_style:
            new_cell.font = copy(cell.font)
            new_cell.border = copy(cell.border)
            new_cell.fill = copy(cell.fill)
            new_cell.number_format = copy(cell.number_format)
            new_cell.protection = copy(cell.protection)
            new_cell.alignment = copy(cell.alignment)

# 3) Formatted Data
fd_ws = out_wb.create_sheet(title='Formatted Data')
fd_headers = ['Facility', 'Session ID', 'Bin ID', 'Product ID', 'Expected Qty', 'Allowed Variance', 'Approval Needed',
              'Missing Final Count', 'Approval Gap', 'Total Errors', 'Error Summary']
for col_idx, h in enumerate(fd_headers, 1):
    fd_ws.cell(row=1, column=col_idx, value=h)
for row_idx, frow in enumerate(formatted_rows, 2):
    for col_idx, val in enumerate(frow, 1):
        fd_ws.cell(row=row_idx, column=col_idx, value=val)

# 4) Summary
sum_ws = out_wb.create_sheet(title='Summary')
sum_headers = ['Facility', 'Session ID', 'Missing Final Counts', 'Approval Gaps', 'Total Errors']
for col_idx, h in enumerate(sum_headers, 1):
    sum_ws.cell(row=1, column=col_idx, value=h)
for row_idx, srow in enumerate(summary_rows, 2):
    for col_idx, val in enumerate(srow, 1):
        sum_ws.cell(row=row_idx, column=col_idx, value=val)

out_wb.save('/root/Cycle_Count_Variance_Audit.xlsx')
print('Excel saved.')

# ─── Find top 2 high-priority facility-session combos ───
# Sort summary_rows (excluding Grand Total) by Total Errors descending
error_rows = [r for r in summary_rows if r[0] != 'Grand Total']
error_rows.sort(key=lambda x: x[4], reverse=True)
top2 = error_rows[:2]
print('Top 2 facility-session combos:', top2)

# ─── Create Word document ───
from docx import Document

doc = Document()
doc.add_heading('Cycle Count Variance Audit – Executive Brief', level=1)

# Build the mention strings - include both original form and space-replaced form
top_mentions = []
for r in top2:
    fac = str(r[0]).strip()
    sess = str(r[1]).strip()
    top_mentions.append(f'{fac} / {sess}')

top_str = ' and '.join(top_mentions)

para_text = (
    f'This audit examined cycle count records to identify two key exceptions: '
    f'Missing Final Count (where a bin has no final count event recorded, indicating the count was never completed) '
    f'and Approval Gap (where a final count exists but the absolute variance between expected and counted quantities '
    f'exceeds the allowed threshold for items requiring approval, indicating a potential compliance issue). '
    f'Across all facilities and sessions, the audit identified {total_missing} Missing Final Count errors, '
    f'{total_approval} Approval Gap errors, and {total_total} Total Errors. '
    f'The highest-priority facility-session combinations with frequent exceptions are '
)

# Add each facility and session ID explicitly so the verifier can find them
for i, r in enumerate(top2):
    fac = str(r[0]).strip()
    sess = str(r[1]).strip()
    if i > 0:
        para_text += ' and '
    para_text += f'{fac} (Session {sess} with {r[4]} total errors)'

para_text += '. '
para_text += (
    'We recommend prioritizing recounts for bins with Missing Final Count errors and '
    'implementing a mandatory supervisory sign-off for all Approval Gap exceptions to '
    'strengthen inventory accuracy and compliance.'
)

doc.add_paragraph(para_text)
doc.save('/root/Cycle_Count_Variance_Brief.docx')
print('Word document saved.')

# Verify files exist
for f in ['/root/Cycle_Count_Variance_Audit.xlsx', '/root/Cycle_Count_Variance_Brief.docx']:
    print(f'{f} exists: {os.path.exists(f)}')

print('DONE')
```

After running the script, verify:
1. Both output files exist at the specified paths.
2. The Excel file has exactly 4 sheets named: Overview, RawData, Formatted Data, Summary.
3. The Word document contains facility and session ID identifiers from the top-2 error combinations.
4. Print the content of the Word document paragraphs to confirm the facility-session mentions are present.

Verification step — run this after the main script:
```python
from docx import Document
doc = Document('/root/Cycle_Count_Variance_Brief.docx')
for p in doc.paragraphs:
    print(repr(p.text))

import openpyxl
wb = openpyxl.load_workbook('/root/Cycle_Count_Variance_Audit.xlsx')
print('Sheet names:', wb.sheetnames)
for sheet_name in ['Formatted Data', 'Summary']:
    ws = wb[sheet_name]
    print(f'\n--- {sheet_name} ---')
    for row in ws.iter_rows(max_row=min(ws.max_row, 5), values_only=True):
        print(row)
    if ws.max_row > 5:
        print('...')
        # Print last 3 rows
        for row in ws.iter_rows(min_row=max(ws.max_row-2, 1), values_only=True):
            print(row)
print('Verification complete.')
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
Task metadata: author_email=catpaw@meituan.com, author_name=CatPaw Benchmark Builder, category=spreadsheet-audit, difficulty=expert, tags=[excel, openpyxl, docx, audit, inventory].
Verifier config: timeout_sec=900.0.