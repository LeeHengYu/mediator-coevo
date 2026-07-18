# Task Instruction

Execute the following steps to produce the required workbook.

## 1. Inspect the source workbook

```python
import openpyxl, os
wb = openpyxl.load_workbook('/root/Inventory_and_Shipments_Latest.xlsx', data_only=True)
for name in wb.sheetnames:
    ws = wb[name]
    print(f'\n=== {name} === (rows={ws.max_row}, cols={ws.max_column})')
    for r in ws.iter_rows(min_row=1, max_row=min(ws.max_row, 15), values_only=False):
        print([(c.coordinate, c.value) for c in r])
```

Run this first and read the output carefully. Identify:
- The exact cell for AsOfDate (expected `Current Inventory!B1`) and PlanningHorizonEnd (`Current Inventory!D1`).
- Column layout of `Current Inventory` (SKU names, inventory amounts, daily rates).
- Column layout of `Incoming Shipments` (SKU, delivery date, number of cases left).
- Column layout of `Ratio` sheet (SKU, cases per pallet).

## 2. Build the output workbook

After inspecting, write and run a single Python script that:

### 2a. Parse source data
- Read `AsOfDate` from `Current Inventory!B1`. If it is a datetime, convert to `date`. Store as `as_of_date`.
- Read `PlanningHorizonEnd` from `Current Inventory!D1`. Store as `horizon_end`.
- `remaining_days = (horizon_end - as_of_date).days`  — expect 27 for Jul 4→Jul 31.
- Parse the SKU data rows from `Current Inventory`: for each SKU row, capture `Product_SKU`, `Current_Cases`, `Daily_Rate_Cases_Per_Day`. Preserve the row order exactly.
- Parse `Incoming Shipments`: for each row capture SKU, Delivery Date (as date), Number of Cases Left (as numeric). Build a dict: SKU → list of (delivery_date, cases).
- Parse `Ratio`: build a dict SKU → Cases_Per_Pallet.

### 2b. Compute per-SKU values

For each SKU (in source order):

```
rate = Daily_Rate_Cases_Per_Day
current = Current_Cases

if rate > 0:
    current_doh = current / rate
    projected_oos = as_of_date + timedelta(days=int(current_doh))   # floor
else:
    current_doh = None
    projected_oos = None

# Inbound cases: sum of cases for shipments with delivery_date <= horizon_end
inbound = sum(cases for (dd, cases) in shipments.get(sku, []) if dd <= horizon_end)

if rate > 0:
    delivered_doh = (current + inbound) / rate
else:
    delivered_doh = None

remaining_demand = rate * remaining_days
additional = max(0, remaining_demand - current - inbound)

cases_per_pallet = ratio_dict.get(sku, None)
if additional > 0 and cases_per_pallet:
    import math
    pallets = math.ceil(additional / cases_per_pallet)
    rounding_applied = (pallets != additional / cases_per_pallet)  # TRUE if ceil changed value
else:
    pallets = 0
    rounding_applied = False

# Earliest scheduled inbound date
sku_dates = [dd for (dd, _) in shipments.get(sku, [])]
earliest_inbound = min(sku_dates) if sku_dates else None

# Required_Delivery_Date
if pallets == 0:
    req_del_date = None
else:
    if earliest_inbound is not None and projected_oos is not None and earliest_inbound <= projected_oos:
        req_del_date = as_of_date + timedelta(days=int(delivered_doh))  # floor
    else:
        req_del_date = projected_oos

# Earlier_Delivery_Required
if pallets > 0 and (earliest_inbound is None or (req_del_date is not None and req_del_date < earliest_inbound)):
    earlier_delivery = True
else:
    earlier_delivery = False
```

### 2c. Write the output workbook

Use `openpyxl` to create `/root/additional_shipments_needed_updated_july_2025.xlsx`.

**Sheet 1: `SKU_Results`**
- A1="Field", B1="Value"
- A2="AsOfDate", B2=as_of_date formatted as "YYYY-MM-DD" string
- A3="PlanningHorizonEnd", B3=horizon_end formatted as "YYYY-MM-DD" string
- A4="RemainingDaysInJuly", B4=remaining_days (integer)
- Row 6 (A6:N6): exact header names listed in the task.
- Data rows starting at row 7, one per SKU in source order.
- Columns E (Projected_OOS_Date), K (Required_Delivery_Date), N (Earliest_Scheduled_Inbound_Date): write as ISO date strings "YYYY-MM-DD" or None/blank.
- Boolean columns L (Rounding_Applied) and M (Earlier_Delivery_Required): write Python `True`/`False` booleans so openpyxl stores them as Excel TRUE/FALSE.
- Numeric columns (B through J excluding E): write as numbers (int or float).
- For blank values (rate=0 cases), write `None`.

**Sheet 2: `Additional_Shipments_Needed`**
- Row 1 headers: Product_SKU, Required_Delivery_Date, Pallets_Required_Rounded_Up, Additional_Cases_Needed, Rounding_Applied, Earlier_Delivery_Required
- Include only SKUs where pallets > 0, same order as SKU_Results.
- Required_Delivery_Date as ISO string.
- Rounding_Applied and Earlier_Delivery_Required as booleans.
- Numeric fields as numbers.

Save the workbook.

## 3. Validate

After writing, re-open the output file and print:
- Sheet names (must be exactly `['SKU_Results', 'Additional_Shipments_Needed']`)
- Metadata cells A1:B4 of SKU_Results
- Header row 6 of SKU_Results (all 14 names)
- First 5 data rows of SKU_Results with all column values
- All rows of Additional_Shipments_Needed
- Confirm date columns contain strings in YYYY-MM-DD format
- Confirm boolean columns contain True/False
- Confirm numeric columns contain numbers
- Confirm Additional_Shipments_Needed only has rows with Pallets > 0
- Confirm SKU order matches source

If any discrepancy is found, fix and re-save.

## Important notes
- Do NOT modify the source file.
- Adapt column indices based on what you actually see when inspecting the source. The instructions above assume typical layouts; adjust if the actual layout differs.
- When computing `Inbound_Cases_By_July31`, compare dates properly (convert any datetime to date if needed).
- `floor` for DOH means `int()` truncation toward zero (Python `int()` on a positive float).
- `ceil` is `math.ceil`.
- The output path must be exactly `/root/additional_shipments_needed_updated_july_2025.xlsx`.

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

Task-local resources are available under `environment/skills`: Inventory Turnover Analyzer, bc-calculated-fields-manufacturing, inventory-manager, shelf-life-management, stochastic-inventory-models.

# Verifier Contract

Success is judged by the SkillFlow verifier for this task.
Do not bypass, remove, or weaken verifier scripts, tests, fixtures, or expected-output checks.
Run the provided tests or verifier command when practical before finalizing.
Task metadata: author_email=codex@example.com, author_name=Codex, category=supply-chain, difficulty=medium, tags=[inventory, excel, replenishment, logistics, forecasting].
Verifier config: timeout_sec=900.0.