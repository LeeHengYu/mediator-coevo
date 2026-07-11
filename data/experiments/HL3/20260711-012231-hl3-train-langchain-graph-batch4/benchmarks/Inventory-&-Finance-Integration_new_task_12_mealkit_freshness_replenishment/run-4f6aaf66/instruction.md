# Task Instruction

Execute the following steps to produce /root/freshness_replenishment_plan_november_2025.xlsx.

## Step 1 – Inspect the source workbook

```python
import openpyxl
wb = openpyxl.load_workbook('/root/MealKits_Inventory_and_Inbound_Latest.xlsx', data_only=True)
for s in wb.sheetnames:
    ws = wb[s]
    print(f'\n=== {s} === (rows={ws.max_row}, cols={ws.max_column})')
    for r in range(1, min(ws.max_row+1, 12)):
        print([ws.cell(r,c).value for c in range(1, ws.max_column+1)])
```

Read and understand:
- Current Inventory: cell B1 = AsOfDate, D1 = PlanningHorizonEnd. Identify the header row and data rows. Note columns for Meal_Kit_ID, Current_Boxes, Daily_Order_Rate_Boxes.
- Incoming Deliveries: identify columns for Meal_Kit_ID, inbound quantity, inbound/delivery date.
- Shelf_Life: identify columns for Meal_Kit_ID, shelf-life days, Boxes_Per_Pallet (conversion ratio), and any expiry-related info.

Print all rows of Shelf_Life and Incoming Deliveries so you have the full picture.

## Step 2 – Build the output workbook

Write a single Python script that:

1. Reads the source workbook (openpyxl, data_only=True).
2. Extracts AsOfDate from Current Inventory B1 and PlanningHorizonEnd from D1. Convert to datetime.date if needed.
3. Computes RemainingDaysInNovember = (PlanningHorizonEnd - AsOfDate).days
4. Parses Current Inventory data rows into a list of dicts preserving source order, with keys: Meal_Kit_ID, Current_Boxes, Daily_Order_Rate_Boxes.
5. Parses Shelf_Life into a dict keyed by Meal_Kit_ID with shelf_life_days and boxes_per_pallet.
6. Computes Boxes_Expiring_By_Nov30 for each entity:
   - Determine the production/receipt date or expiry date from the data. If the sheet gives shelf-life in days, compute expiry = AsOfDate + shelf_life_days (or however the source encodes it). Boxes whose expiry <= PlanningHorizonEnd count as expiring. If shelf life info implies all current boxes expire by Nov 30, use Current_Boxes; if none expire, use 0. Inspect the data carefully to determine the correct interpretation.
7. Parses Incoming Deliveries; for each entity sums inbound boxes where delivery_date <= PlanningHorizonEnd.
8. Finds Earliest_Scheduled_Inbound_Date per entity (the minimum delivery date across all inbound records for that entity; blank/None if none).
9. For each entity computes all 16 columns per the rules below (use the CORRECTED definitions where Usable_Current_Boxes feeds into Current_DOH and Delivered_DOH_To_Nov30):

   - Usable_Current_Boxes = max(0, Current_Boxes - Boxes_Expiring_By_Nov30)
   - Current_DOH = Usable_Current_Boxes / Daily_Order_Rate_Boxes  (if rate > 0, else None)
   - Projected_OOS_Date = AsOfDate + timedelta(days=floor(Current_DOH))  (if rate > 0, else None)
   - Inbound_Boxes_By_Nov30 = sum of qualifying inbound
   - Delivered_DOH_To_Nov30 = (Usable_Current_Boxes + Inbound_Boxes_By_Nov30) / Daily_Order_Rate_Boxes  (if rate > 0, else None)
   - Remaining_November_Demand_Boxes = Daily_Order_Rate_Boxes * RemainingDaysInNovember
   - Additional_Boxes_Needed = max(0, Remaining_November_Demand_Boxes - Usable_Current_Boxes - Inbound_Boxes_By_Nov30)
   - boxes_per_pallet from Shelf_Life
   - raw_pallets = Additional_Boxes_Needed / boxes_per_pallet
   - Pallets_Required_Rounded_Up = math.ceil(raw_pallets) if Additional_Boxes_Needed > 0 else 0
   - Rounding_Applied = True if Additional_Boxes_Needed > 0 and math.ceil(raw_pallets) != raw_pallets else False
   - Required_Delivery_Date:
     * None if Pallets_Required_Rounded_Up == 0
     * else if Earliest_Scheduled_Inbound_Date is not None and Earliest_Scheduled_Inbound_Date <= Projected_OOS_Date: AsOfDate + timedelta(days=floor(Delivered_DOH_To_Nov30))
     * else: Projected_OOS_Date
   - Earlier_Delivery_Required = True if Pallets_Required_Rounded_Up > 0 and (Earliest_Scheduled_Inbound_Date is None or Required_Delivery_Date < Earliest_Scheduled_Inbound_Date) else False

10. All date columns (Projected_OOS_Date, Required_Delivery_Date, Earliest_Scheduled_Inbound_Date) must be written as ISO strings (str, 'YYYY-MM-DD') or None/blank.
11. Boolean columns (Rounding_Applied, Earlier_Delivery_Required) must be native Python bool (True/False).
12. Numeric columns must remain numeric (int or float), not strings.

## Step 3 – Write Sheet 1: Freshness_Results

Create the output workbook with openpyxl. In the first sheet named 'Freshness_Results':
- A1='Field', B1='Value'
- A2='AsOfDate', B2=AsOfDate as 'YYYY-MM-DD' string
- A3='PlanningHorizonEnd', B3=PlanningHorizonEnd as 'YYYY-MM-DD' string
- A4='RemainingDaysInNovember', B4=integer
- Row 6: the 16 column headers exactly as specified.
- Data rows starting at row 7, one per entity in source order.

## Step 4 – Write Sheet 2: Additional_Freshness_Needed

Second sheet named 'Additional_Freshness_Needed':
- Row 1: 6 column headers exactly as specified.
- Include only rows where Pallets_Required_Rounded_Up > 0, same entity order as Sheet 1.

## Step 5 – Save and verify

Save to /root/freshness_replenishment_plan_november_2025.xlsx.

Then re-open and verify:
- Print sheet names (must be exactly ['Freshness_Results', 'Additional_Freshness_Needed']).
- Print metadata cells A1:B4.
- Print header row 6.
- Print all data rows from both sheets.
- Confirm date columns contain strings, boolean columns contain bools, numeric columns contain numbers.
- Confirm Sheet 2 only has rows with Pallets_Required_Rounded_Up > 0.
- Confirm no source files were modified.

If any issue is found during verification, fix it and re-save before finishing.

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