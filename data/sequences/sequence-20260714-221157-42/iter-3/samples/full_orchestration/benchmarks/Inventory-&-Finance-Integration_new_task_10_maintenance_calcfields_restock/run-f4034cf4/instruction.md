# Task Instruction

You must build a single Excel workbook at /root/maintenance_resupply_actions_sep_2025.xlsx using Python (openpyxl). Follow every step below precisely.

## Step 0 – Inspect the source workbook

```bash
cd /root
python3 -c "
import openpyxl, json
wb = openpyxl.load_workbook('Maintenance_Parts_and_Deliveries_Latest.xlsx', data_only=True)
for s in wb.sheetnames:
    ws = wb[s]
    print(f'\n=== {s} === (rows={ws.max_row}, cols={ws.max_column})')
    for r in ws.iter_rows(min_row=1, max_row=min(ws.max_row, 25), values_only=False):
        print([(c.coordinate, c.value) for c in r])
"
```

Read and understand:
- **Current Parts**: find AsOfDate (B1), PlanningHorizonEnd (D1), and the data rows (part codes, current units, daily consumption). Note the exact column positions.
- **Scheduled Deliveries**: find part code, inbound date, inbound quantity columns.
- **Ratio**: find the conversion ratio (units per crate).

Record what you find before writing any code.

## Step 1 – Write the generation script

Create `/root/build_workbook.py` with the full logic below. Adapt column indices to match what you observed in Step 0.

```python
import openpyxl
from openpyxl import Workbook
from datetime import datetime, timedelta
import math

# ---- Load source ----
src = openpyxl.load_workbook('Maintenance_Parts_and_Deliveries_Latest.xlsx', data_only=True)

# Read Current Parts
cp = src['Current Parts']
as_of_date = cp['B1'].value          # should be a date or string
planning_end = cp['D1'].value         # should be a date or string

# Convert to date objects if needed
if isinstance(as_of_date, str):
    as_of_date = datetime.strptime(as_of_date, '%Y-%m-%d').date()
elif isinstance(as_of_date, datetime):
    as_of_date = as_of_date.date()

if isinstance(planning_end, str):
    planning_end = datetime.strptime(planning_end, '%Y-%m-%d').date()
elif isinstance(planning_end, datetime):
    planning_end = planning_end.date()

remaining_days = (planning_end - as_of_date).days

# Read parts data – ADAPT header row and column indices after inspection
# Find header row in Current Parts (likely row 2 or row 3)
# Read all part rows below header
# Store as list of dicts: {part_code, current_units, daily_consumption}

# Read Scheduled Deliveries
# Store as dict: part_code -> list of (date, quantity)

# Read Ratio sheet – find the units-per-crate conversion ratio

# ---- Calculations per part ----
# For each part (preserving source order):
#   current_doh = current_units / daily_consumption if daily_consumption > 0 else None
#   projected_stockout = as_of_date + timedelta(days=math.floor(current_doh)) if rate > 0 else None
#   inbound_units = sum of qty where delivery_date <= planning_end for this part
#   delivered_doh = (current_units + inbound_units) / daily_consumption if rate > 0 else None
#   remaining_demand = daily_consumption * remaining_days
#   additional_needed = max(0, remaining_demand - current_units - inbound_units)
#   raw_crates = additional_needed / ratio
#   crates_rounded = math.ceil(raw_crates) if additional_needed > 0 else 0
#   earliest_scheduled = min of delivery dates for this part, or None
#   required_delivery_date:
#       if crates_rounded == 0: None
#       elif earliest_scheduled is not None and earliest_scheduled <= projected_stockout:
#           as_of_date + timedelta(days=math.floor(delivered_doh))
#       else: projected_stockout
#   rounding_applied = True if additional_needed > 0 and crates_rounded != raw_crates else False
#       (i.e., raw_crates is not already an integer)
#   earlier_delivery_required = True if crates_rounded > 0 and
#       (earliest_scheduled is None or required_delivery_date < earliest_scheduled) else False

# ---- Write output workbook ----
wb = Workbook()

# Sheet 1: Part_Results
ws1 = wb.active
ws1.title = 'Part_Results'

# Metadata
ws1['A1'] = 'Field';  ws1['B1'] = 'Value'
ws1['A2'] = 'AsOfDate';  ws1['B2'] = as_of_date.isoformat()
ws1['A3'] = 'PlanningHorizonEnd';  ws1['B3'] = planning_end.isoformat()
ws1['A4'] = 'RemainingDaysInSeptember';  ws1['B4'] = remaining_days

# Header at row 6
headers = ['Part_Code','Current_Units','Daily_Consumption_Units','Current_DOH',
           'Projected_Stockout_Date','Inbound_Units_By_Sep30','Delivered_DOH_To_Sep30',
           'Remaining_September_Demand_Units','Additional_Units_Needed',
           'Crates_Required_Rounded_Up','Required_Delivery_Date','Rounding_Applied',
           'Earlier_Delivery_Required','Earliest_Scheduled_Delivery_Date']
for ci, h in enumerate(headers, 1):
    ws1.cell(row=6, column=ci, value=h)

# Data rows starting at row 7
# For each part write the 14 columns.
# Date columns (Projected_Stockout_Date, Required_Delivery_Date, Earliest_Scheduled_Delivery_Date)
#   must be ISO strings (YYYY-MM-DD) or blank (None).
# Numeric fields must be numeric (int or float), not strings.
# Boolean fields (Rounding_Applied, Earlier_Delivery_Required) must be Python True/False
#   so openpyxl writes them as Excel booleans.

# Sheet 2: Additional_Resupply_Needed
ws2 = wb.create_sheet('Additional_Resupply_Needed')
headers2 = ['Part_Code','Required_Delivery_Date','Crates_Required_Rounded_Up',
            'Additional_Units_Needed','Rounding_Applied','Earlier_Delivery_Required']
for ci, h in enumerate(headers2, 1):
    ws2.cell(row=1, column=ci, value=h)

# Only rows where Crates_Required_Rounded_Up > 0, same order as Part_Results

wb.save('/root/maintenance_resupply_actions_sep_2025.xlsx')
print('Workbook saved successfully.')
```

**CRITICAL**: The skeleton above is a template. You MUST adapt it after inspecting the source workbook in Step 0. Specifically:
- Identify the exact row where part data starts in Current Parts (could be row 2, 3, etc.).
- Identify exact column letters/indices for Part_Code, Current_Units, Daily_Consumption_Units.
- Identify exact column layout of Scheduled Deliveries (part code column, date column, quantity column).
- Identify how the Ratio sheet stores the conversion ratio (single cell? column?).
- Handle date parsing for delivery dates (they may be datetime objects or strings).

## Step 2 – Run and verify

```bash
python3 /root/build_workbook.py
```

If errors occur, fix them and re-run.

## Step 3 – Validate the output

```bash
python3 -c "
import openpyxl
wb = openpyxl.load_workbook('/root/maintenance_resupply_actions_sep_2025.xlsx', data_only=True)
print('Sheets:', wb.sheetnames)
assert wb.sheetnames == ['Part_Results', 'Additional_Resupply_Needed'], f'Wrong sheets: {wb.sheetnames}'

ws1 = wb['Part_Results']
print('A1:', ws1['A1'].value, 'B1:', ws1['B1'].value)
print('A2:', ws1['A2'].value, 'B2:', ws1['B2'].value)
print('A3:', ws1['A3'].value, 'B3:', ws1['B3'].value)
print('A4:', ws1['A4'].value, 'B4:', ws1['B4'].value)
assert ws1['A1'].value == 'Field' and ws1['B1'].value == 'Value'
assert ws1['A2'].value == 'AsOfDate'
assert ws1['A3'].value == 'PlanningHorizonEnd'
assert ws1['A4'].value == 'RemainingDaysInSeptember'
assert isinstance(ws1['B4'].value, (int, float))

# Check header row 6
expected_h = ['Part_Code','Current_Units','Daily_Consumption_Units','Current_DOH',
              'Projected_Stockout_Date','Inbound_Units_By_Sep30','Delivered_DOH_To_Sep30',
              'Remaining_September_Demand_Units','Additional_Units_Needed',
              'Crates_Required_Rounded_Up','Required_Delivery_Date','Rounding_Applied',
              'Earlier_Delivery_Required','Earliest_Scheduled_Delivery_Date']
actual_h = [ws1.cell(row=6, column=c).value for c in range(1,15)]
assert actual_h == expected_h, f'Header mismatch: {actual_h}'

# Print first few data rows
for r in range(7, min(ws1.max_row+1, 12)):
    vals = [ws1.cell(row=r, column=c).value for c in range(1,15)]
    print(f'Row {r}: {vals}')

# Check data types: numeric fields should be numbers, booleans should be bool
for r in range(7, ws1.max_row+1):
    cu = ws1.cell(row=r, column=2).value
    if cu is not None:
        assert isinstance(cu, (int, float)), f'Row {r} Current_Units not numeric: {type(cu)}'
    ra = ws1.cell(row=r, column=12).value
    assert isinstance(ra, bool), f'Row {r} Rounding_Applied not bool: {type(ra)} = {ra}'
    ed = ws1.cell(row=r, column=13).value
    assert isinstance(ed, bool), f'Row {r} Earlier_Delivery_Required not bool: {type(ed)} = {ed}'

# Check Sheet 2
ws2 = wb['Additional_Resupply_Needed']
expected_h2 = ['Part_Code','Required_Delivery_Date','Crates_Required_Rounded_Up',
               'Additional_Units_Needed','Rounding_Applied','Earlier_Delivery_Required']
actual_h2 = [ws2.cell(row=1, column=c).value for c in range(1,7)]
assert actual_h2 == expected_h2, f'Sheet2 header mismatch: {actual_h2}'

# All rows in sheet 2 should have Crates > 0
for r in range(2, ws2.max_row+1):
    crates = ws2.cell(row=r, column=3).value
    if crates is not None:
        assert crates > 0, f'Sheet2 row {r} has crates={crates}'

print('All validation checks passed.')
"
```

If any assertion fails, diagnose and fix the generation script, then re-run.

## Key Reminders

- **Do NOT modify the source file** `/root/Maintenance_Parts_and_Deliveries_Latest.xlsx`.
- **Booleans** must be Python `True`/`False` (not strings "TRUE"/"FALSE") so openpyxl writes Excel boolean type.
- **Date columns** (Projected_Stockout_Date, Required_Delivery_Date, Earliest_Scheduled_Delivery_Date) must be **ISO format strings** (`str` type, e.g. `'2025-09-15'`), not datetime objects.
- **Numeric fields** must be `int` or `float`, not strings.
- **Rounding_Applied**: TRUE only when additional_needed > 0 AND `math.ceil(additional_needed/ratio) != additional_needed/ratio` (i.e., the ceiling operation actually rounded up).
- **Earlier_Delivery_Required**: TRUE when crates > 0 AND (earliest_scheduled is None OR required_delivery_date < earliest_scheduled).
- When daily_consumption is 0 or missing, Current_DOH, Projected_Stockout_Date, and Delivered_DOH_To_Sep30 should be blank (None).
- The output file must exist at exactly `/root/maintenance_resupply_actions_sep_2025.xlsx`.

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

Task-local resources are available under `environment/skills`: Inventory Turnover Analyzer, bc-calculated-fields-manufacturing.

# Verifier Contract

Success is judged by the SkillFlow verifier for this task.
Do not bypass, remove, or weaken verifier scripts, tests, fixtures, or expected-output checks.
Run the provided tests or verifier command when practical before finalizing.
Task metadata: author_email=codex@example.com, author_name=Codex, category=manufacturing-maintenance, difficulty=medium, tags=[excel, manufacturing, maintenance, calculated-fields, restock].
Verifier config: timeout_sec=900.0.