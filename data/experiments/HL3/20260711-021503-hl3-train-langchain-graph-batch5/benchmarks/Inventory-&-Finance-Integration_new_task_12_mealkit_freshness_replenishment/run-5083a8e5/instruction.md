# Task Instruction

## Task: Build Freshness Replenishment Plan for November 2025

Create a single Excel workbook at `/root/freshness_replenishment_plan_november_2025.xlsx` using data from `/root/MealKits_Inventory_and_Inbound_Latest.xlsx`.

### Step 1: Inspect the source workbook

Read all three sheets from `/root/MealKits_Inventory_and_Inbound_Latest.xlsx`:
- **Current Inventory** — note the structure carefully: B1 should contain the AsOfDate, D1 should contain PlanningHorizonEnd. Identify all entity rows, their Meal_Kit_ID, Current_Boxes, Daily_Order_Rate_Boxes, and any other columns.
- **Incoming Deliveries** — note columns for Meal_Kit_ID, delivery date, and quantity.
- **Shelf_Life** — note columns for Meal_Kit_ID, shelf life days (or expiry info), boxes-per-pallet conversion ratio, and any expiration-related data.

Print the first ~20 rows and all column names of each sheet. Also print the exact values in cells B1 and D1 of Current Inventory (these are the dates). Print the dtypes of all columns.

### Step 2: Understand the data model

Before computing anything:
- Determine how to compute **Boxes_Expiring_By_Nov30**: Using shelf life data from the Shelf_Life sheet combined with inventory receipt/production dates. If Current Inventory has a "Date_Received" or similar column, boxes expire when `Date_Received + Shelf_Life_Days <= PlanningHorizonEnd`. If the expiration logic is different (e.g., an explicit expiry date column), adapt accordingly. Print your understanding before proceeding.
- Determine the **boxes-per-pallet conversion ratio** from Shelf_Life for each Meal_Kit_ID.
- Determine how to find **Earliest_Scheduled_Inbound_Date** per entity from Incoming Deliveries.

### Step 3: Compute all values

For each Meal_Kit_ID (preserving source order from Current Inventory):

```
AsOfDate = Current Inventory B1 (parse as date)
PlanningHorizonEnd = Current Inventory D1 (parse as date)
RemainingDaysInNovember = (PlanningHorizonEnd - AsOfDate).days

Boxes_Expiring_By_Nov30: boxes from current inventory that expire on or before PlanningHorizonEnd
Usable_Current_Boxes = max(0, Current_Boxes - Boxes_Expiring_By_Nov30)

Daily_Order_Rate_Boxes: from Current Inventory

Current_DOH = Usable_Current_Boxes / Daily_Order_Rate_Boxes  (if rate > 0, else blank/None)
Projected_OOS_Date = AsOfDate + timedelta(days=floor(Current_DOH))  (if rate > 0, else blank)

Inbound_Boxes_By_Nov30 = sum of inbound quantity where delivery_date <= PlanningHorizonEnd for this entity

Delivered_DOH_To_Nov30 = (Usable_Current_Boxes + Inbound_Boxes_By_Nov30) / Daily_Order_Rate_Boxes  (if rate > 0, else blank)

Remaining_November_Demand_Boxes = Daily_Order_Rate_Boxes * RemainingDaysInNovember

Additional_Boxes_Needed = max(0, Remaining_November_Demand_Boxes - Usable_Current_Boxes - Inbound_Boxes_By_Nov30)

boxes_per_pallet = from Shelf_Life for this entity
Pallets_Required_Rounded_Up = ceil(Additional_Boxes_Needed / boxes_per_pallet) if Additional_Boxes_Needed > 0, else 0

Rounding_Applied = TRUE if Additional_Boxes_Needed > 0 AND (Additional_Boxes_Needed % boxes_per_pallet != 0), else FALSE
  (i.e., rounding changed the container count)

Earliest_Scheduled_Inbound_Date = earliest delivery date for this entity from Incoming Deliveries, else blank

Required_Delivery_Date:
  - blank (None) if Pallets_Required_Rounded_Up == 0
  - else if Earliest_Scheduled_Inbound_Date is not blank AND Earliest_Scheduled_Inbound_Date <= Projected_OOS_Date:
      AsOfDate + timedelta(days=floor(Delivered_DOH_To_Nov30))
  - else: Projected_OOS_Date

Earlier_Delivery_Required = TRUE if Pallets_Required_Rounded_Up > 0 AND
  (Earliest_Scheduled_Inbound_Date is blank OR Required_Delivery_Date < Earliest_Scheduled_Inbound_Date)
  else FALSE
```

**IMPORTANT**: Date columns (Projected_OOS_Date, Required_Delivery_Date, Earliest_Scheduled_Inbound_Date) must be stored as ISO format strings "YYYY-MM-DD" (not datetime objects). Blank values should be None/empty.

Boolean fields (Rounding_Applied, Earlier_Delivery_Required) must be actual Python booleans (True/False), stored as explicit boolean values in the Excel file.

Numeric fields must remain numeric (int or float), not strings.

### Step 4: Build Sheet 1 — Freshness_Results

Metadata block:
- A1="Field", B1="Value"
- A2="AsOfDate", B2=AsOfDate as YYYY-MM-DD string
- A3="PlanningHorizonEnd", B3=PlanningHorizonEnd as YYYY-MM-DD string  
- A4="RemainingDaysInNovember", B4=integer

Header row at row 6 (0-indexed row 5) with exactly these 16 columns in order:
1. Meal_Kit_ID
2. Current_Boxes
3. Boxes_Expiring_By_Nov30
4. Usable_Current_Boxes
5. Daily_Order_Rate_Boxes
6. Current_DOH
7. Projected_OOS_Date
8. Inbound_Boxes_By_Nov30
9. Delivered_DOH_To_Nov30
10. Remaining_November_Demand_Boxes
11. Additional_Boxes_Needed
12. Pallets_Required_Rounded_Up
13. Required_Delivery_Date
14. Rounding_Applied
15. Earlier_Delivery_Required
16. Earliest_Scheduled_Inbound_Date

Data rows start at row 7 (0-indexed row 6).

### Step 5: Build Sheet 2 — Additional_Freshness_Needed

Header at row 1 with exactly these 6 columns:
1. Meal_Kit_ID
2. Required_Delivery_Date
3. Pallets_Required_Rounded_Up
4. Additional_Boxes_Needed
5. Rounding_Applied
6. Earlier_Delivery_Required

Include only rows where Pallets_Required_Rounded_Up > 0, preserving the same entity order as Freshness_Results. Each entity appears at most once.

### Step 6: Write the workbook

Use openpyxl or xlsxwriter (via pandas ExcelWriter) to write exactly two sheets in order: Freshness_Results, Additional_Freshness_Needed.

Do NOT modify the source file.

### Step 7: Validate

1. Re-read the output file and verify:
   - Exactly 2 sheets with correct names in correct order
   - Sheet 1 metadata cells A1:B4 are correct
   - Sheet 1 header at row 6 has all 16 columns
   - Sheet 1 data rows match entity count from source
   - Sheet 2 header has all 6 columns
   - Sheet 2 only contains rows with Pallets_Required_Rounded_Up > 0
   - Date columns contain strings in YYYY-MM-DD format (not datetime)
   - Boolean columns contain True/False
   - Numeric columns are numeric
2. Print a summary of the validation results.
3. Print a few sample rows from each sheet for manual inspection.

If any validation fails, fix and re-validate before finishing.

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