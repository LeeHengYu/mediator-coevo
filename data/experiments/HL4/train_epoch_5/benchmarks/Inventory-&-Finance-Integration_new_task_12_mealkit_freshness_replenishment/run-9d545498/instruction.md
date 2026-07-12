# Task Instruction

Execute the following steps to produce /root/freshness_replenishment_plan_november_2025.xlsx.

## Step 0: Inspect the source workbook

```python
import openpyxl
wb = openpyxl.load_workbook('/root/MealKits_Inventory_and_Inbound_Latest.xlsx', data_only=True)
for s in wb.sheetnames:
    ws = wb[s]
    print(f'\n=== {s} ===')
    for row in ws.iter_rows(min_row=1, max_row=min(ws.max_row, 15), values_only=False):
        print([(c.coordinate, c.value) for c in row])
    if ws.max_row > 15:
        print(f'  ... total rows: {ws.max_row}')
```

Read and understand every column name, the date cells (Current Inventory!B1 = AsOfDate, Current Inventory!D1 = PlanningHorizonEnd), the Shelf_Life sheet (especially the boxes-per-pallet conversion ratio column), and the Incoming Deliveries sheet structure. Print all rows so nothing is missed.

## Step 1: Build the output workbook

Write a single Python script that:

1. Reads the source workbook with openpyxl (data_only=True).
2. Extracts:
   - AsOfDate from Current Inventory!B1 (convert to datetime.date if needed).
   - PlanningHorizonEnd from Current Inventory!D1 (convert to datetime.date).
   - RemainingDaysInNovember = (PlanningHorizonEnd - AsOfDate).days
   - For each entity row in Current Inventory (starting after the header row — determine where the header row is by inspecting Step 0 output): Meal_Kit_ID, Current_Boxes, Daily_Order_Rate_Boxes, and any expiry-related columns.
   - From Shelf_Life sheet: for each Meal_Kit_ID, get the shelf life in days AND the boxes-per-pallet conversion ratio.
   - From Incoming Deliveries: for each Meal_Kit_ID, collect all (date, quantity) pairs.

3. For Boxes_Expiring_By_Nov30: Using the Shelf_Life sheet's shelf life days, compute the expiration date of current stock. If the current stock's expiration date (AsOfDate + shelf_life_days, or if there's a production/receipt date use that + shelf_life_days — inspect the data to determine the right approach) is <= PlanningHorizonEnd, then Boxes_Expiring_By_Nov30 = Current_Boxes, else 0. IMPORTANT: Carefully inspect what columns exist in Current Inventory and Shelf_Life to determine the correct expiry calculation. If there is an explicit expiry date column, use it directly.

4. Computes all columns per the rules:
   - Usable_Current_Boxes = max(0, Current_Boxes - Boxes_Expiring_By_Nov30)
   - Current_DOH = Usable_Current_Boxes / Daily_Order_Rate_Boxes (if rate > 0, else None)
   - Projected_OOS_Date = AsOfDate + timedelta(days=floor(Current_DOH)) (if rate > 0, else None) — store as ISO string YYYY-MM-DD
   - Inbound_Boxes_By_Nov30 = sum of inbound quantities where delivery_date <= PlanningHorizonEnd
   - Delivered_DOH_To_Nov30 = (Usable_Current_Boxes + Inbound_Boxes_By_Nov30) / Daily_Order_Rate_Boxes (if rate > 0, else None)
   - Remaining_November_Demand_Boxes = Daily_Order_Rate_Boxes * RemainingDaysInNovember
   - Additional_Boxes_Needed = max(0, Remaining_November_Demand_Boxes - Usable_Current_Boxes - Inbound_Boxes_By_Nov30)
   - Pallets_Required_Rounded_Up = math.ceil(Additional_Boxes_Needed / boxes_per_pallet) if Additional_Boxes_Needed > 0 else 0
   - Rounding_Applied = True if Additional_Boxes_Needed > 0 and (Additional_Boxes_Needed % boxes_per_pallet != 0) else False
   - Earliest_Scheduled_Inbound_Date = min of inbound dates for that entity, or None — store as ISO string
   - Required_Delivery_Date:
     - None if Pallets_Required_Rounded_Up == 0
     - else if Earliest_Scheduled_Inbound_Date is not None and Earliest_Scheduled_Inbound_Date <= Projected_OOS_Date: AsOfDate + timedelta(days=floor(Delivered_DOH_To_Nov30)) — as ISO string
     - else: Projected_OOS_Date (already ISO string)
   - Earlier_Delivery_Required = True if Pallets_Required_Rounded_Up > 0 and (Earliest_Scheduled_Inbound_Date is None or Required_Delivery_Date < Earliest_Scheduled_Inbound_Date) else False

5. Creates the output workbook with exactly two sheets in order: Freshness_Results, Additional_Freshness_Needed.

6. Freshness_Results sheet:
   - A1='Field', B1='Value'
   - A2='AsOfDate', B2=AsOfDate as YYYY-MM-DD string
   - A3='PlanningHorizonEnd', B3=PlanningHorizonEnd as YYYY-MM-DD string
   - A4='RemainingDaysInNovember', B4=integer
   - Row 6: the 16 column headers exactly as specified
   - Data rows starting at row 7, one per entity in source order
   - Numeric fields as numbers (int or float), date fields as ISO strings, boolean fields as Python True/False (which openpyxl writes as Excel TRUE/FALSE)

7. Additional_Freshness_Needed sheet:
   - Row 1: the 6 column headers exactly as specified
   - Only rows where Pallets_Required_Rounded_Up > 0, same order as Freshness_Results

8. Saves to /root/freshness_replenishment_plan_november_2025.xlsx

## Step 2: Validate

After creating the file, re-open it and print:
- Sheet names
- Freshness_Results rows 1-4 (metadata)
- Freshness_Results row 6 (headers)
- First 5 data rows
- Additional_Freshness_Needed headers and all rows
- Verify all date columns contain strings (not datetime objects)
- Verify boolean columns contain True/False
- Verify numeric columns contain numbers

## Critical Notes
- Do NOT modify the source file.
- Inspect the source thoroughly in Step 0 before writing any computation code. Column names, date formats, and structure may differ from assumptions.
- The boxes-per-pallet conversion ratio is in the Shelf_Life sheet — find the correct column name by inspection.
- When comparing dates for Required_Delivery_Date logic, convert ISO strings back to dates for comparison.
- Ensure the output workbook has exactly 2 sheets (remove the default sheet if openpyxl creates one).
- All date output cells must be ISO format strings, not Excel date serial numbers or datetime objects.

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