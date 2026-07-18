# Task Instruction

Produce the required Excel workbook at `/root/additional_shipments_needed_updated_july_2025.xlsx` by following these steps precisely:

## Step 0 – Inspect the source workbook

```python
import openpyxl, os
wb = openpyxl.load_workbook('/root/Inventory_and_Shipments_Latest.xlsx', data_only=True)
for s in wb.sheetnames:
    ws = wb[s]
    print(f'\n=== {s} (rows={ws.max_row}, cols={ws.max_column}) ===')
    for r in ws.iter_rows(min_row=1, max_row=min(ws.max_row, 15), values_only=False):
        print([(c.coordinate, c.value) for c in r])
```

Run this first and read every printed line carefully before writing any logic. You need to understand:
- The exact layout of `Current Inventory` (where AsOfDate is in B1, PlanningHorizonEnd is in D1, where the SKU data rows start, which columns hold SKU name, current cases, daily rate).
- The layout of `Incoming Shipments` (SKU column, delivery date column, number-of-cases column — find the exact header names).
- The layout of `Ratio` (SKU column, cases-per-pallet column — find the exact header names).

Print ALL rows of `Incoming Shipments` and `Ratio` so you have complete data.

## Step 1 – Parse source data into Python structures

Using what you learned from the inspection:

1. Extract `AsOfDate` and `PlanningHorizonEnd` as `datetime.date` objects. If they are datetime objects, call `.date()`. Compute `RemainingDaysInJuly = (PlanningHorizonEnd - AsOfDate).days`.

2. Build an ordered list of SKU dicts from `Current Inventory` data rows, each with keys: `sku` (string), `current_cases` (numeric), `daily_rate` (numeric). Preserve the row order from the source sheet.

3. Build a dict mapping each SKU to a list of `(delivery_date, cases_left)` tuples from `Incoming Shipments`. Be careful with the column name — it may be "Number of Cases Left" or similar. Parse dates properly.

4. Build a dict mapping each SKU to its `cases_per_pallet` from the `Ratio` sheet.

## Step 2 – Compute SKU_Results rows

For each SKU (in source order), compute every field per these rules:

- `Current_DOH` = `current_cases / daily_rate` if `daily_rate > 0`, else `None`
- `Projected_OOS_Date` = `AsOfDate + timedelta(days=floor(Current_DOH))` if rate > 0, else `None`
- `Inbound_Cases_By_July31` = sum of cases for that SKU where `delivery_date <= PlanningHorizonEnd`
- `Delivered_DOH_To_July31` = `(current_cases + inbound_cases) / daily_rate` if rate > 0, else `None`
- `Remaining_July_Demand_Cases` = `daily_rate * RemainingDaysInJuly`
- `Additional_Cases_Needed` = `max(0, remaining_demand - current_cases - inbound_cases)`
- `cases_per_pallet` from Ratio sheet for this SKU
- `Pallets_Required_Rounded_Up` = `math.ceil(additional_cases / cases_per_pallet)` if `additional_cases > 0`, else `0`
- `Rounding_Applied` = `True` if `additional_cases > 0` and `additional_cases / cases_per_pallet != math.ceil(additional_cases / cases_per_pallet)` (i.e., there was a fractional remainder), else `False`
- `Earliest_Scheduled_Inbound_Date` = earliest delivery date for this SKU from Incoming Shipments, or `None` if no shipments exist
- `Required_Delivery_Date`:
  - `None` if `pallets_required == 0`
  - else if `Earliest_Scheduled_Inbound_Date` is not None AND `Earliest_Scheduled_Inbound_Date <= Projected_OOS_Date`: use `AsOfDate + timedelta(days=floor(Delivered_DOH_To_July31))`
  - else: use `Projected_OOS_Date`
- `Earlier_Delivery_Required` = `True` if `pallets_required > 0` AND (`Earliest_Scheduled_Inbound_Date is None` OR `Required_Delivery_Date < Earliest_Scheduled_Inbound_Date`), else `False`

**Important**: All date output fields (Projected_OOS_Date, Required_Delivery_Date, Earliest_Scheduled_Inbound_Date) must be stored as ISO format strings `YYYY-MM-DD` in the Excel cells, not as date objects.

**Important**: Boolean fields (`Rounding_Applied`, `Earlier_Delivery_Required`) must be Python `True`/`False` booleans written to the cells.

## Step 3 – Write the output workbook

Create a new workbook with exactly two sheets in this order: `SKU_Results`, `Additional_Shipments_Needed`.

### SKU_Results sheet:
- A1='Field', B1='Value'
- A2='AsOfDate', B2=AsOfDate as 'YYYY-MM-DD' string
- A3='PlanningHorizonEnd', B3=PlanningHorizonEnd as 'YYYY-MM-DD' string
- A4='RemainingDaysInJuly', B4=integer
- Row 6 (A6:N6): the 14 headers exactly as specified: `Product_SKU`, `Current_Cases`, `Daily_Rate_Cases_Per_Day`, `Current_DOH`, `Projected_OOS_Date`, `Inbound_Cases_By_July31`, `Delivered_DOH_To_July31`, `Remaining_July_Demand_Cases`, `Additional_Cases_Needed`, `Pallets_Required_Rounded_Up`, `Required_Delivery_Date`, `Rounding_Applied`, `Earlier_Delivery_Required`, `Earliest_Scheduled_Inbound_Date`
- Data rows start at row 7, one per SKU in source order.
- Numeric fields must be numbers (int or float), not strings.
- Blank/None cells should be left empty (write `None` which openpyxl treats as empty).

### Additional_Shipments_Needed sheet:
- Row 1 (A1:F1): `Product_SKU`, `Required_Delivery_Date`, `Pallets_Required_Rounded_Up`, `Additional_Cases_Needed`, `Rounding_Applied`, `Earlier_Delivery_Required`
- Include only SKUs where `Pallets_Required_Rounded_Up > 0`, same order as SKU_Results.
- Same data types as in SKU_Results.

Remove any default sheets (like 'Sheet') that openpyxl creates.

Save to `/root/additional_shipments_needed_updated_july_2025.xlsx`.

## Step 4 – Validate

Reload the saved workbook and print:
1. All sheet names
2. SKU_Results cells A1:B4
3. SKU_Results row 6 headers
4. All SKU_Results data rows (row 7+)
5. All Additional_Shipments_Needed rows

Verify:
- Sheet names are exactly `['SKU_Results', 'Additional_Shipments_Needed']`
- Metadata cells match expected values
- Headers match exactly (spelling, underscores)
- Date strings are in YYYY-MM-DD format
- Boolean fields are actual booleans
- Numeric fields are numbers
- No extra/missing rows

If anything is wrong, fix and re-save before finishing.

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