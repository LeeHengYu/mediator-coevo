# Task Instruction

Execute the following Python script to produce the required Excel workbook. Before running, inspect the source workbook to understand its structure, then generate the output.

```bash
cd /root && python3 << 'PYEOF'
import openpyxl
from openpyxl import Workbook
from datetime import datetime, timedelta
import math

# ── 1. Read source workbook ──────────────────────────────────────────────
src = openpyxl.load_workbook('/root/MealKits_Inventory_and_Inbound_Latest.xlsx', data_only=True)

# ── 2. Parse Current Inventory ───────────────────────────────────────────
ws_inv = src['Current Inventory']

# AsOfDate from B1, PlanningHorizonEnd from D1
raw_as_of = ws_inv['B1'].value
raw_horizon = ws_inv['D1'].value

def to_date(v):
    if isinstance(v, datetime):
        return v.date() if hasattr(v, 'date') else v
    if isinstance(v, str):
        for fmt in ('%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y'):
            try:
                return datetime.strptime(v.strip(), fmt).date()
            except ValueError:
                pass
    # maybe it's already a date
    return v

as_of_date = to_date(raw_as_of)
horizon_date = to_date(raw_horizon)
remaining_days = (horizon_date - as_of_date).days

print(f'AsOfDate={as_of_date}, PlanningHorizonEnd={horizon_date}, RemainingDays={remaining_days}')

# Find header row in Current Inventory (row 2 or 3 typically)
inv_headers = {}
for row in ws_inv.iter_rows(min_row=1, max_row=10):
    vals = [c.value for c in row]
    # look for a row that has 'Meal_Kit_ID' or similar
    lower_vals = [str(v).strip().lower().replace(' ','_') if v else '' for v in vals]
    if any('meal' in lv and 'kit' in lv for lv in lower_vals) or any('meal_kit_id' in lv for lv in lower_vals):
        for i, v in enumerate(vals):
            if v is not None:
                inv_headers[str(v).strip()] = i
        header_row_num = row[0].row
        break

print(f'Inventory headers at row {header_row_num}: {inv_headers}')

# Read inventory data rows
inv_data = []
for row in ws_inv.iter_rows(min_row=header_row_num+1, values_only=False):
    vals = [c.value for c in row]
    if vals[0] is None and all(v is None for v in vals[:3]):
        continue
    inv_data.append(vals)

print(f'Inventory rows: {len(inv_data)}')

# Map header names
def find_col(headers, *candidates):
    for c in candidates:
        for h, idx in headers.items():
            if c.lower().replace(' ','_') == h.lower().replace(' ','_'):
                return idx
    # fuzzy
    for c in candidates:
        for h, idx in headers.items():
            if c.lower() in h.lower():
                return idx
    return None

col_kit_id = find_col(inv_headers, 'Meal_Kit_ID', 'Meal Kit ID', 'MealKitID')
col_current_boxes = find_col(inv_headers, 'Current_Boxes', 'Current Boxes', 'Boxes')
col_daily_rate = find_col(inv_headers, 'Daily_Order_Rate_Boxes', 'Daily Order Rate', 'Daily_Order_Rate')
col_expiring = find_col(inv_headers, 'Boxes_Expiring_By_Nov30', 'Boxes Expiring', 'Expiring')

print(f'Columns: kit_id={col_kit_id}, current_boxes={col_current_boxes}, daily_rate={col_daily_rate}, expiring={col_expiring}')

# ── 3. Parse Incoming Deliveries ─────────────────────────────────────────
ws_del = src['Incoming Deliveries']
del_headers = {}
for row in ws_del.iter_rows(min_row=1, max_row=10):
    vals = [c.value for c in row]
    lower_vals = [str(v).strip().lower().replace(' ','_') if v else '' for v in vals]
    if any('meal' in lv and 'kit' in lv for lv in lower_vals) or any('delivery' in lv or 'date' in lv for lv in lower_vals):
        for i, v in enumerate(vals):
            if v is not None:
                del_headers[str(v).strip()] = i
        del_header_row = row[0].row
        break

print(f'Delivery headers at row {del_header_row}: {del_headers}')

col_del_kit = find_col(del_headers, 'Meal_Kit_ID', 'Meal Kit ID')
col_del_qty = find_col(del_headers, 'Boxes', 'Quantity', 'Inbound_Boxes', 'Inbound Boxes', 'Delivery_Boxes')
col_del_date = find_col(del_headers, 'Delivery_Date', 'Delivery Date', 'Inbound_Date', 'Date')

print(f'Delivery columns: kit={col_del_kit}, qty={col_del_qty}, date={col_del_date}')

# Build delivery lookup: kit_id -> list of (date, qty)
from collections import defaultdict
deliveries = defaultdict(list)
for row in ws_del.iter_rows(min_row=del_header_row+1, values_only=True):
    if row[col_del_kit] is None:
        continue
    kit = str(row[col_del_kit]).strip()
    qty = row[col_del_qty]
    d = to_date(row[col_del_date])
    if qty is not None and d is not None:
        deliveries[kit].append((d, float(qty)))

print(f'Deliveries parsed for {len(deliveries)} kits')

# ── 4. Parse Shelf_Life ──────────────────────────────────────────────────
ws_sl = src['Shelf_Life']
sl_headers = {}
for row in ws_sl.iter_rows(min_row=1, max_row=10):
    vals = [c.value for c in row]
    lower_vals = [str(v).strip().lower().replace(' ','_') if v else '' for v in vals]
    if any('meal' in lv and 'kit' in lv for lv in lower_vals) or any('pallet' in lv or 'box' in lv or 'conversion' in lv for lv in lower_vals):
        for i, v in enumerate(vals):
            if v is not None:
                sl_headers[str(v).strip()] = i
        sl_header_row = row[0].row
        break

print(f'Shelf_Life headers at row {sl_header_row}: {sl_headers}')

col_sl_kit = find_col(sl_headers, 'Meal_Kit_ID', 'Meal Kit ID')
col_sl_conv = find_col(sl_headers, 'Boxes_Per_Pallet', 'Boxes Per Pallet', 'Conversion', 'Pallet_Size', 'BoxesPerPallet')

print(f'Shelf_Life columns: kit={col_sl_kit}, conv={col_sl_conv}')

conversion = {}
for row in ws_sl.iter_rows(min_row=sl_header_row+1, values_only=True):
    if row[col_sl_kit] is None:
        continue
    kit = str(row[col_sl_kit]).strip()
    conv = float(row[col_sl_conv]) if row[col_sl_conv] else 1
    conversion[kit] = conv

print(f'Conversion ratios: {conversion}')

# ── 5. Build output ──────────────────────────────────────────────────────
wb = Workbook()

# Sheet 1: Freshness_Results
ws1 = wb.active
ws1.title = 'Freshness_Results'

# Metadata
ws1['A1'] = 'Field'
ws1['B1'] = 'Value'
ws1['A2'] = 'AsOfDate'
ws1['B2'] = as_of_date.strftime('%Y-%m-%d')
ws1['A3'] = 'PlanningHorizonEnd'
ws1['B3'] = horizon_date.strftime('%Y-%m-%d')
ws1['A4'] = 'RemainingDaysInNovember'
ws1['B4'] = remaining_days

# Header at row 6
headers_out = [
    'Meal_Kit_ID', 'Current_Boxes', 'Boxes_Expiring_By_Nov30',
    'Usable_Current_Boxes', 'Daily_Order_Rate_Boxes', 'Current_DOH',
    'Projected_OOS_Date', 'Inbound_Boxes_By_Nov30', 'Delivered_DOH_To_Nov30',
    'Remaining_November_Demand_Boxes', 'Additional_Boxes_Needed',
    'Pallets_Required_Rounded_Up', 'Required_Delivery_Date',
    'Rounding_Applied', 'Earlier_Delivery_Required',
    'Earliest_Scheduled_Inbound_Date'
]
for ci, h in enumerate(headers_out, 1):
    ws1.cell(row=6, column=ci, value=h)

sheet2_rows = []

for ri, inv_row in enumerate(inv_data):
    kit_id = str(inv_row[col_kit_id]).strip() if inv_row[col_kit_id] else None
    if kit_id is None:
        continue
    current_boxes = float(inv_row[col_current_boxes]) if inv_row[col_current_boxes] is not None else 0
    daily_rate = float(inv_row[col_daily_rate]) if inv_row[col_daily_rate] is not None else 0
    expiring = float(inv_row[col_expiring]) if inv_row[col_expiring] is not None else 0

    usable = max(0, current_boxes - expiring)

    # Current_DOH based on usable (the later definition overrides the earlier one)
    if daily_rate > 0:
        current_doh = usable / daily_rate
    else:
        current_doh = None

    # Projected_OOS_Date
    if daily_rate > 0 and current_doh is not None:
        proj_oos = as_of_date + timedelta(days=math.floor(current_doh))
        proj_oos_str = proj_oos.strftime('%Y-%m-%d')
    else:
        proj_oos = None
        proj_oos_str = None

    # Inbound_Boxes_By_Nov30
    kit_deliveries = deliveries.get(kit_id, [])
    inbound_by_nov30 = sum(qty for d, qty in kit_deliveries if d <= horizon_date)

    # Delivered_DOH_To_Nov30 (using usable, per later definition)
    if daily_rate > 0:
        delivered_doh = (usable + inbound_by_nov30) / daily_rate
    else:
        delivered_doh = None

    # Remaining_November_Demand_Boxes
    remaining_demand = daily_rate * remaining_days

    # Additional_Boxes_Needed
    additional = max(0, remaining_demand - usable - inbound_by_nov30)

    # Pallets_Required_Rounded_Up
    conv_ratio = conversion.get(kit_id, 1)
    if additional > 0:
        raw_pallets = additional / conv_ratio
        pallets = math.ceil(raw_pallets)
        rounding_applied = (pallets != raw_pallets)  # True if ceil changed the value
    else:
        pallets = 0
        raw_pallets = 0
        rounding_applied = False

    # Earliest_Scheduled_Inbound_Date
    if kit_deliveries:
        earliest_inbound = min(d for d, q in kit_deliveries)
        earliest_inbound_str = earliest_inbound.strftime('%Y-%m-%d')
    else:
        earliest_inbound = None
        earliest_inbound_str = None

    # Required_Delivery_Date
    if pallets == 0:
        req_del_date_str = None
    else:
        if earliest_inbound is not None and proj_oos is not None and earliest_inbound <= proj_oos:
            # use AsOfDate + floor(Delivered_DOH_To_Nov30)
            if delivered_doh is not None:
                req_del = as_of_date + timedelta(days=math.floor(delivered_doh))
                req_del_date_str = req_del.strftime('%Y-%m-%d')
            else:
                req_del_date_str = None
        else:
            # use Projected_OOS_Date
            req_del_date_str = proj_oos_str

    # Earlier_Delivery_Required
    if pallets > 0:
        if earliest_inbound is None:
            earlier_delivery = True
        else:
            if req_del_date_str is not None:
                req_del_parsed = datetime.strptime(req_del_date_str, '%Y-%m-%d').date()
                earlier_delivery = req_del_parsed < earliest_inbound
            else:
                earlier_delivery = False
    else:
        earlier_delivery = False

    out_row = [
        kit_id,
        current_boxes,
        expiring,
        usable,
        daily_rate,
        round(current_doh, 6) if current_doh is not None else None,
        proj_oos_str,
        inbound_by_nov30,
        round(delivered_doh, 6) if delivered_doh is not None else None,
        remaining_demand,
        additional,
        pallets,
        req_del_date_str,
        rounding_applied,
        earlier_delivery,
        earliest_inbound_str
    ]

    row_num = 7 + ri
    for ci, val in enumerate(out_row, 1):
        ws1.cell(row=row_num, column=ci, value=val)

    if pallets > 0:
        sheet2_rows.append([
            kit_id,
            req_del_date_str,
            pallets,
            additional,
            rounding_applied,
            earlier_delivery
        ])

# Sheet 2: Additional_Freshness_Needed
ws2 = wb.create_sheet('Additional_Freshness_Needed')
headers2 = [
    'Meal_Kit_ID', 'Required_Delivery_Date', 'Pallets_Required_Rounded_Up',
    'Additional_Boxes_Needed', 'Rounding_Applied', 'Earlier_Delivery_Required'
]
for ci, h in enumerate(headers2, 1):
    ws2.cell(row=1, column=ci, value=h)

for ri, s2row in enumerate(sheet2_rows, 2):
    for ci, val in enumerate(s2row, 1):
        ws2.cell(row=ri, column=ci, value=val)

# Save
outpath = '/root/freshness_replenishment_plan_november_2025.xlsx'
wb.save(outpath)
print(f'Saved to {outpath}')
print(f'Sheet1 data rows: {len(inv_data)}, Sheet2 rows: {len(sheet2_rows)}')
PYEOF
```

After the script runs, verify the output:
```bash
python3 -c "
import openpyxl
wb = openpyxl.load_workbook('/root/freshness_replenishment_plan_november_2025.xlsx')
print('Sheets:', wb.sheetnames)
ws1 = wb['Freshness_Results']
print('Metadata:')
for r in range(1,5):
    print(f'  {ws1.cell(r,1).value} = {ws1.cell(r,2).value}')
print('Headers row 6:', [ws1.cell(6,c).value for c in range(1,17)])
print('First data row:', [ws1.cell(7,c).value for c in range(1,17)])
ws2 = wb['Additional_Freshness_Needed']
print('Sheet2 headers:', [ws2.cell(1,c).value for c in range(1,7)])
print('Sheet2 row count:', ws2.max_row - 1)
if ws2.max_row > 1:
    print('Sheet2 first row:', [ws2.cell(2,c).value for c in range(1,7)])
"
```

If the script encounters column-finding issues (prints None for column indices), inspect the source sheets manually:
```bash
python3 -c "
import openpyxl
wb = openpyxl.load_workbook('/root/MealKits_Inventory_and_Inbound_Latest.xlsx', data_only=True)
for name in wb.sheetnames:
    ws = wb[name]
    print(f'\n=== {name} ===')
    for row in ws.iter_rows(min_row=1, max_row=5, values_only=False):
        print([f'{c.coordinate}={c.value}' for c in row])
"
```
Then adjust column mappings and re-run.

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

Task-local resources are available under `environment/skills`: inventory-manager, shelf-life-management.

# Verifier Contract

Success is judged by the SkillFlow verifier for this task.
Do not bypass, remove, or weaken verifier scripts, tests, fixtures, or expected-output checks.
Run the provided tests or verifier command when practical before finalizing.
Task metadata: author_email=codex@example.com, author_name=Codex, category=fresh-food-operations, difficulty=medium, tags=[excel, shelf-life, freshness, replenishment, operations].
Verifier config: timeout_sec=900.0.