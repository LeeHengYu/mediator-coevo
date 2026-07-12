# Task Instruction

## Task: Cycle Count Variance Audit

You must create two deliverable files from three input files. Follow every step carefully.

### Step 0: Inspect Input Files

```bash
cd /root
python3 -c "
import openpyxl
for fname in ['Cycle_Plan.xlsx', 'Count_Event_Log.xlsx', 'Cycle_Template.xlsx']:
    wb = openpyxl.load_workbook(fname)
    print(f'\n=== {fname} ===')
    print(f'Sheets: {wb.sheetnames}')
    for sn in wb.sheetnames:
        ws = wb[sn]
        print(f'  Sheet [{sn}]: {ws.max_row} rows x {ws.max_column} cols')
        for r in range(1, min(ws.max_row+1, 8)):
            print(f'    Row {r}: {[ws.cell(r, c).value for c in range(1, ws.max_column+1)]}')
"
```

Read and understand:
- `Cycle_Plan.xlsx`: the plan table with columns like Facility, Session ID, Bin ID, Product ID, Expected Qty, Allowed Variance, Approval Needed.
- `Count_Event_Log.xlsx`: event log with columns including Facility, Session ID, Bin ID, Event Type, Count Qty, and possibly a timestamp or row order.
- `Cycle_Template.xlsx`: has an `Overview` sheet to preserve exactly.

### Step 1: Build the Python Script

Create and run a Python script `/root/build_audit.py` that does the following:

```python
import openpyxl
from openpyxl.utils import get_column_letter
from copy import copy
from docx import Document
import pandas as pd

# ---- Load inputs ----
plan_wb = openpyxl.load_workbook('Cycle_Plan.xlsx')
log_wb = openpyxl.load_workbook('Count_Event_Log.xlsx')
template_wb = openpyxl.load_workbook('Cycle_Template.xlsx')

# Read plan data
plan_ws = plan_wb.active
plan_headers = [plan_ws.cell(1, c).value for c in range(1, plan_ws.max_column+1)]
print('Plan headers:', plan_headers)
plan_data = []
for r in range(2, plan_ws.max_row+1):
    row = [plan_ws.cell(r, c).value for c in range(1, plan_ws.max_column+1)]
    plan_data.append(row)
print(f'Plan rows: {len(plan_data)}')

# Read event log
log_ws = log_wb.active
log_headers = [log_ws.cell(1, c).value for c in range(1, log_ws.max_column+1)]
print('Log headers:', log_headers)
log_data = []
for r in range(2, log_ws.max_row+1):
    row = {log_headers[c]: log_ws.cell(r, c+1).value for c in range(len(log_headers))}
    row['_row_num'] = r  # preserve original row order for "latest" determination
    log_data.append(row)
print(f'Log rows: {len(log_data)}')

# Identify column names in log (handle possible variations)
# We need: Facility, Session ID, Bin ID, Event Type, Count Qty
# Print a sample to confirm
print('Sample log row:', log_data[0] if log_data else 'EMPTY')

# Build lookup: for each (Facility, Session ID, Bin ID), keep only the latest FINAL event
# "Latest" = last occurrence in file order (highest _row_num)
final_lookup = {}
for row in log_data:
    facility = row.get('Facility')
    session_id = row.get('Session ID')
    bin_id = row.get('Bin ID')
    event_type = row.get('Event Type')
    count_qty = row.get('Count Qty')
    
    # Skip rows with blank keys or blank Count Qty
    if facility is None or session_id is None or bin_id is None:
        continue
    if str(facility).strip() == '' or str(session_id).strip() == '' or str(bin_id).strip() == '':
        continue
    
    # Only FINAL events
    if event_type is None or str(event_type).strip().upper() != 'FINAL':
        continue
    
    # Skip blank Count Qty
    if count_qty is None or str(count_qty).strip() == '':
        continue
    
    key = (str(facility).strip(), str(session_id).strip(), str(bin_id).strip())
    # Keep the latest (highest row number)
    if key not in final_lookup or row['_row_num'] > final_lookup[key]['_row_num']:
        final_lookup[key] = row

print(f'Final lookup entries: {len(final_lookup)}')

# ---- Create output workbook ----
out_wb = openpyxl.Workbook()
# Remove default sheet
out_wb.remove(out_wb.active)

# --- 1) Overview sheet: copy from template exactly ---
template_overview = template_wb['Overview']
out_overview = out_wb.create_sheet('Overview')

# Copy cell values, styles, merged cells, column widths, row heights
for row in template_overview.iter_rows(min_row=1, max_row=template_overview.max_row, max_col=template_overview.max_column):
    for cell in row:
        new_cell = out_overview.cell(row=cell.row, column=cell.column, value=cell.value)
        if cell.has_style:
            new_cell.font = copy(cell.font)
            new_cell.border = copy(cell.border)
            new_cell.fill = copy(cell.fill)
            new_cell.number_format = copy(cell.number_format)
            new_cell.protection = copy(cell.protection)
            new_cell.alignment = copy(cell.alignment)

for merged_range in template_overview.merged_cells.ranges:
    out_overview.merge_cells(str(merged_range))

for col_letter in [get_column_letter(c) for c in range(1, template_overview.max_column+1)]:
    if template_overview.column_dimensions[col_letter].width:
        out_overview.column_dimensions[col_letter].width = template_overview.column_dimensions[col_letter].width

for r in range(1, template_overview.max_row+1):
    if template_overview.row_dimensions[r].height:
        out_overview.row_dimensions[r].height = template_overview.row_dimensions[r].height

# --- 2) RawData sheet: copy plan table exactly ---
out_raw = out_wb.create_sheet('RawData')
for row in plan_ws.iter_rows(min_row=1, max_row=plan_ws.max_row, max_col=plan_ws.max_column):
    for cell in row:
        new_cell = out_raw.cell(row=cell.row, column=cell.column, value=cell.value)
        if cell.has_style:
            new_cell.font = copy(cell.font)
            new_cell.border = copy(cell.border)
            new_cell.fill = copy(cell.fill)
            new_cell.number_format = copy(cell.number_format)
            new_cell.protection = copy(cell.protection)
            new_cell.alignment = copy(cell.alignment)

# --- 3) Formatted Data sheet ---
out_fmt = out_wb.create_sheet('Formatted Data')

# Map plan column names to indices (0-based)
plan_col_map = {h: i for i, h in enumerate(plan_headers) if h is not None}
print('Plan column map:', plan_col_map)

# Expected first 7 columns
fmt_headers = ['Facility', 'Session ID', 'Bin ID', 'Product ID', 'Expected Qty', 'Allowed Variance', 'Approval Needed',
               'Missing Final Count', 'Approval Gap', 'Total Errors', 'Error Summary']

# Write headers
for c, h in enumerate(fmt_headers, 1):
    out_fmt.cell(1, c, h)

# Track data for Summary
summary_data = {}  # (facility, session_id) -> {missing: int, approval: int, total: int}

total_missing = 0
total_approval = 0
total_errors = 0

for r_idx, row_vals in enumerate(plan_data, 2):
    # Write first 7 columns from plan
    facility_val = row_vals[plan_col_map.get('Facility', 0)]
    session_val = row_vals[plan_col_map.get('Session ID', 1)]
    bin_val = row_vals[plan_col_map.get('Bin ID', 2)]
    product_val = row_vals[plan_col_map.get('Product ID', 3)]
    expected_qty = row_vals[plan_col_map.get('Expected Qty', 4)]
    allowed_var = row_vals[plan_col_map.get('Allowed Variance', 5)]
    approval_needed = row_vals[plan_col_map.get('Approval Needed', 6)]
    
    out_fmt.cell(r_idx, 1, facility_val)
    out_fmt.cell(r_idx, 2, session_val)
    out_fmt.cell(r_idx, 3, bin_val)
    out_fmt.cell(r_idx, 4, product_val)
    out_fmt.cell(r_idx, 5, expected_qty)
    out_fmt.cell(r_idx, 6, allowed_var)
    out_fmt.cell(r_idx, 7, approval_needed)
    
    # Lookup final count
    key = (str(facility_val).strip() if facility_val else '',
           str(session_val).strip() if session_val else '',
           str(bin_val).strip() if bin_val else '')
    
    final_event = final_lookup.get(key)
    
    # Missing Final Count
    missing_final = 1 if final_event is None else 0
    
    # Approval Gap
    approval_gap = 0
    if final_event is not None:
        approval_str = str(approval_needed).strip().upper() if approval_needed else ''
        if approval_str == 'YES':
            count_qty = final_event.get('Count Qty')
            try:
                diff = abs(float(expected_qty) - float(count_qty))
                if diff > float(allowed_var):
                    approval_gap = 1
            except (TypeError, ValueError):
                pass
    
    total_err = missing_final + approval_gap
    
    # Error Summary
    errors = []
    if missing_final == 1:
        errors.append('Missing Final Count')
    if approval_gap == 1:
        errors.append('Approval Gap')
    error_summary = ', '.join(errors) if errors else 'None'
    
    out_fmt.cell(r_idx, 8, missing_final)
    out_fmt.cell(r_idx, 9, approval_gap)
    out_fmt.cell(r_idx, 10, total_err)
    out_fmt.cell(r_idx, 11, error_summary)
    
    # Accumulate for summary
    grp_key = (str(facility_val).strip() if facility_val else '', str(session_val).strip() if session_val else '')
    if grp_key not in summary_data:
        summary_data[grp_key] = {'missing': 0, 'approval': 0, 'total': 0}
    summary_data[grp_key]['missing'] += missing_final
    summary_data[grp_key]['approval'] += approval_gap
    summary_data[grp_key]['total'] += total_err
    
    total_missing += missing_final
    total_approval += approval_gap
    total_errors += total_err

# --- 4) Summary sheet ---
out_sum = out_wb.create_sheet('Summary')
sum_headers = ['Facility', 'Session ID', 'Missing Final Counts', 'Approval Gaps', 'Total Errors']
for c, h in enumerate(sum_headers, 1):
    out_sum.cell(1, c, h)

# Filter groups with Total Errors > 0, sort by Facility asc, Session ID asc
filtered = [(k, v) for k, v in summary_data.items() if v['total'] > 0]
filtered.sort(key=lambda x: (x[0][0], x[0][1]))

row_num = 2
for (fac, sess), vals in filtered:
    out_sum.cell(row_num, 1, fac)
    out_sum.cell(row_num, 2, sess)
    out_sum.cell(row_num, 3, vals['missing'])
    out_sum.cell(row_num, 4, vals['approval'])
    out_sum.cell(row_num, 5, vals['total'])
    row_num += 1

# Grand Total row
out_sum.cell(row_num, 1, 'Grand Total')
out_sum.cell(row_num, 2, '-')
out_sum.cell(row_num, 3, total_missing)
out_sum.cell(row_num, 4, total_approval)
out_sum.cell(row_num, 5, total_errors)

out_wb.save('Cycle_Count_Variance_Audit.xlsx')
print('\nSaved Cycle_Count_Variance_Audit.xlsx')
print(f'Totals: Missing={total_missing}, Approval={total_approval}, Errors={total_errors}')

# --- Identify top facility-session combos for the brief ---
top_combos = sorted(filtered, key=lambda x: x[1]['total'], reverse=True)
top_mentions = []
for (fac, sess), vals in top_combos[:3]:
    top_mentions.append(f'{fac} / {sess} ({vals["total"]} errors)')
print('Top combos:', top_mentions)

# ---- Create Word document ----
doc = Document()
doc.add_heading('Cycle Count Variance Audit Brief', level=1)

para_text = (
    f'This audit evaluated cycle-count accuracy across all planned sessions. '
    f'Two checks were applied to every bin in the plan: (1) Missing Final Count, which flags bins '
    f'where no valid FINAL count event was recorded in the event log, indicating the count was never '
    f'completed; and (2) Approval Gap, which flags bins where a FINAL count exists but the absolute '
    f'variance between the expected quantity and the counted quantity exceeds the allowed variance '
    f'threshold for items requiring approval. '
    f'Across the dataset, the audit identified {total_missing} Missing Final Count(s), '
    f'{total_approval} Approval Gap(s), and {total_errors} Total Error(s). '
    f'High-priority facility-session combinations with frequent exceptions include '
    f'{", ".join(top_mentions[:2]) if len(top_mentions) >= 2 else ", ".join(top_mentions)}. '
    f'It is recommended that operations prioritize recounting bins with missing final counts '
    f'immediately and escalate approval-gap exceptions to supervisors for variance reconciliation '
    f'before the next audit cycle.'
)

doc.add_paragraph(para_text)
doc.save('Cycle_Count_Variance_Brief.docx')
print('Saved Cycle_Count_Variance_Brief.docx')
```

### Step 2: Run the Script

```bash
cd /root
pip install openpyxl python-docx pandas 2>/dev/null
python3 build_audit.py
```

Carefully read the output. Check:
- Plan headers match expected column names (Facility, Session ID, Bin ID, Product ID, Expected Qty, Allowed Variance, Approval Needed). If column names differ, adjust the script accordingly and re-run.
- Log headers match expected column names. If they differ, adjust.
- The totals look reasonable.

### Step 3: Handle Column Name Mismatches

If the column names in the input files don't exactly match what the script expects, you MUST update the script to use the actual column names. Common issues:
- Extra spaces, different casing, underscores vs spaces
- The log might have columns like `Count Qty` or `Counted Qty` or `Final Qty`
- There might be a timestamp column to determine "latest" FINAL event

After any fix, re-run and verify.

### Step 4: Validate Output

```bash
python3 -c "
import openpyxl
wb = openpyxl.load_workbook('Cycle_Count_Variance_Audit.xlsx')
print('Sheets:', wb.sheetnames)
assert wb.sheetnames == ['Overview', 'RawData', 'Formatted Data', 'Summary'], f'Sheet names wrong: {wb.sheetnames}'

# Check Formatted Data headers
ws = wb['Formatted Data']
headers = [ws.cell(1, c).value for c in range(1, 12)]
print('Formatted Data headers:', headers)
expected = ['Facility', 'Session ID', 'Bin ID', 'Product ID', 'Expected Qty', 'Allowed Variance', 'Approval Needed',
            'Missing Final Count', 'Approval Gap', 'Total Errors', 'Error Summary']
assert headers == expected, f'Headers mismatch: {headers}'

# Check Summary headers
ws2 = wb['Summary']
sum_h = [ws2.cell(1, c).value for c in range(1, 6)]
print('Summary headers:', sum_h)
expected_sum = ['Facility', 'Session ID', 'Missing Final Counts', 'Approval Gaps', 'Total Errors']
assert sum_h == expected_sum, f'Summary headers mismatch: {sum_h}'

# Check last row is Grand Total
last_row = ws2.max_row
print(f'Summary last row ({last_row}):', [ws2.cell(last_row, c).value for c in range(1, 6)])
assert ws2.cell(last_row, 1).value == 'Grand Total'
assert ws2.cell(last_row, 2).value == '-'

# Spot check a few Formatted Data rows
for r in range(2, min(ws.max_row+1, 7)):
    vals = [ws.cell(r, c).value for c in range(1, 12)]
    print(f'  FD row {r}: {vals}')
    # Verify Total Errors = Missing + Approval
    assert vals[9] == vals[7] + vals[8], f'Total Errors mismatch row {r}'
    # Verify Error Summary consistency
    if vals[7] == 1 and vals[8] == 1:
        assert vals[10] == 'Missing Final Count, Approval Gap'
    elif vals[7] == 1:
        assert vals[10] == 'Missing Final Count'
    elif vals[8] == 1:
        assert vals[10] == 'Approval Gap'
    else:
        assert vals[10] == 'None'

print('\nAll validations passed!')
"
```

Also verify the Word document exists:
```bash
python3 -c "
from docx import Document
doc = Document('Cycle_Count_Variance_Brief.docx')
for p in doc.paragraphs:
    print(p.text[:200] if p.text else '')
print('Word doc OK')
"
```

### Step 5: Fix Any Issues

If any validation fails, diagnose the specific problem, fix the script, re-run, and re-validate. Common issues to watch for:
- Session ID might be numeric (int) in one file and string in another — ensure consistent string comparison in the lookup key
- Bin ID similarly may need type normalization
- The Overview sheet copy must preserve the sheet exactly — if there are images or charts, note they may not copy via openpyxl but cell content should be preserved
- The Summary sort must be ascending by Facility then Session ID — ensure string-based sorting is correct (if Session IDs are numeric, sort numerically)

### Critical Constraints
- Output file: `/root/Cycle_Count_Variance_Audit.xlsx` with sheets exactly named: `Overview`, `RawData`, `Formatted Data`, `Summary`
- Output file: `/root/Cycle_Count_Variance_Brief.docx`
- All computed columns in `Formatted Data` must be concrete values (not Excel formulas)
- The `Overview` sheet must be preserved from the template unchanged
- The `Error Summary` text must be exactly one of: `None`, `Missing Final Count`, `Approval Gap`, `Missing Final Count, Approval Gap`

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