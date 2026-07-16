# Task Instruction

You must build a single Excel workbook at /root/freshness_replenishment_plan_november_2025.xlsx using the source workbook /root/MealKits_Inventory_and_Inbound_Latest.xlsx.

## Step 0 — Inspect the source workbook

Open /root/MealKits_Inventory_and_Inbound_Latest.xlsx and inspect all three sheets:
- "Current Inventory" — print all rows including headers. Pay special attention to cells B1 and D1 (dates), and identify the columns for Meal_Kit_ID, Current_Boxes, Daily_Order_Rate_Boxes, Boxes_Expiring_By_Nov30. Note the exact column names and positions.
- "Incoming Deliveries" — print all rows. Identify columns for Meal_Kit_ID, delivery date, and quantity.
- "Shelf_Life" — print all rows. Identify the column that gives the boxes-per-pallet conversion ratio per Meal_Kit_ID.

Print everything so you can see the exact data before writing any code.

## Step 1 — Write a Python script to produce the output workbook

Use openpyxl (install if needed: `pip install openpyxl`). The script must:

### 1a. Read source data
- Read AsOfDate from Current Inventory cell B1 (the value, which is a date).
- Read PlanningHorizonEnd from Current Inventory cell D1 (the value, which is a date).
- Compute RemainingDaysInNovember = (PlanningHorizonEnd - AsOfDate).days  (calendar day difference, integer).
- Read all entity rows from Current Inventory starting from the data region (identify the header row and data rows from your inspection). Extract: Meal_Kit_ID, Current_Boxes, Daily_Order_Rate_Boxes, Boxes_Expiring_By_Nov30.
- Read Incoming Deliveries: for each entity, collect (date, quantity) pairs.
- Read Shelf_Life: for each entity, get boxes_per_pallet conversion ratio.

### 1b. Compute per-entity values (one row per entity, preserving source order)

For each entity:
1. Usable_Current_Boxes = max(0, Current_Boxes - Boxes_Expiring_By_Nov30)
2. Daily_Order_Rate_Boxes = from source
3. Current_DOH = Usable_Current_Boxes / Daily_Order_Rate_Boxes if rate > 0, else None (blank)
4. Projected_OOS_Date = AsOfDate + timedelta(days=floor(Current_DOH)) if rate > 0, else None
5. Inbound_Boxes_By_Nov30 = sum of inbound quantity for this entity where inbound_date <= PlanningHorizonEnd
6. Delivered_DOH_To_Nov30 = (Usable_Current_Boxes + Inbound_Boxes_By_Nov30) / Daily_Order_Rate_Boxes if rate > 0, else None
7. Remaining_November_Demand_Boxes = Daily_Order_Rate_Boxes * RemainingDaysInNovember
8. Additional_Boxes_Needed = max(0, Remaining_November_Demand_Boxes - Usable_Current_Boxes - Inbound_Boxes_By_Nov30)
9. boxes_per_pallet = from Shelf_Life for this entity
10. Pallets_Required_Rounded_Up = math.ceil(Additional_Boxes_Needed / boxes_per_pallet) if Additional_Boxes_Needed > 0, else 0
11. Rounding_Applied: if Additional_Boxes_Needed > 0 and (Additional_Boxes_Needed % boxes_per_pallet != 0) then TRUE, else FALSE. (Use Python booleans True/False.)
12. Earliest_Scheduled_Inbound_Date = earliest delivery date for this entity from Incoming Deliveries, else None
13. Required_Delivery_Date:
    - None (blank) if Pallets_Required_Rounded_Up == 0
    - else if Earliest_Scheduled_Inbound_Date is not None and Earliest_Scheduled_Inbound_Date <= Projected_OOS_Date: use AsOfDate + timedelta(days=floor(Delivered_DOH_To_Nov30))
    - else: use Projected_OOS_Date
14. Earlier_Delivery_Required: TRUE if Pallets_Required_Rounded_Up > 0 and (Earliest_Scheduled_Inbound_Date is None or Required_Delivery_Date < Earliest_Scheduled_Inbound_Date); else FALSE. (Use Python booleans.)

### 1c. Write Sheet 1: "Freshness_Results"

Create the workbook. First sheet named exactly "Freshness_Results".

Metadata:
- A1="Field", B1="Value"
- A2="AsOfDate", B2=AsOfDate as ISO string "YYYY-MM-DD"
- A3="PlanningHorizonEnd", B3=PlanningHorizonEnd as ISO string
- A4="RemainingDaysInNovember", B4=integer

Row 6 is the header row with exactly these 16 columns in order:
Meal_Kit_ID, Current_Boxes, Boxes_Expiring_By_Nov30, Usable_Current_Boxes, Daily_Order_Rate_Boxes, Current_DOH, Projected_OOS_Date, Inbound_Boxes_By_Nov30, Delivered_DOH_To_Nov30, Remaining_November_Demand_Boxes, Additional_Boxes_Needed, Pallets_Required_Rounded_Up, Required_Delivery_Date, Rounding_Applied, Earlier_Delivery_Required, Earliest_Scheduled_Inbound_Date

Data rows start at row 7. One row per entity in source order.

IMPORTANT formatting rules:
- Numeric fields (Current_Boxes, Boxes_Expiring_By_Nov30, Usable_Current_Boxes, Daily_Order_Rate_Boxes, Current_DOH, Inbound_Boxes_By_Nov30, Delivered_DOH_To_Nov30, Remaining_November_Demand_Boxes, Additional_Boxes_Needed, Pallets_Required_Rounded_Up) must be written as numbers, not strings.
- Date fields (Projected_OOS_Date, Required_Delivery_Date, Earliest_Scheduled_Inbound_Date) must be written as ISO strings "YYYY-MM-DD" (use str with .strftime or .isoformat()). If blank, write None (leave cell empty).
- Boolean fields (Rounding_Applied, Earlier_Delivery_Required) must be written as Python True/False booleans so openpyxl stores them as Excel booleans.

### 1d. Write Sheet 2: "Additional_Freshness_Needed"

Second sheet named exactly "Additional_Freshness_Needed".

Header at row 1 with exactly these 6 columns:
Meal_Kit_ID, Required_Delivery_Date, Pallets_Required_Rounded_Up, Additional_Boxes_Needed, Rounding_Applied, Earlier_Delivery_Required

Include only rows where Pallets_Required_Rounded_Up > 0. Preserve the same entity order as in Freshness_Results. Same formatting rules apply (numbers as numbers, dates as ISO strings, booleans as booleans).

### 1e. Save and verify

Save to /root/freshness_replenishment_plan_november_2025.xlsx. Do NOT modify any source files.

After saving, re-open the output file and print:
- Sheet names
- Metadata cells (A1:B4) from Freshness_Results
- All header and data rows from both sheets
- Types of a sample boolean cell and a sample date cell to confirm booleans are bool and dates are str.

## Step 2 — Review and fix

If anything looks wrong (wrong types, wrong values, missing rows), fix and re-run. Pay special attention to:
- The Current_DOH formula uses Usable_Current_Boxes (not raw Current_Boxes)
- The Delivered_DOH_To_Nov30 formula uses Usable_Current_Boxes (not raw Current_Boxes)
- Rounding_Applied is about whether ceil changed the pallet count vs exact division
- Earlier_Delivery_Required uses Pallets_Required_Rounded_Up > 0 (not Additional_Boxes_Needed > 0) as the gate condition
- Date columns contain ISO strings, not datetime objects
- Boolean columns contain actual booleans, not strings "TRUE"/"FALSE"

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