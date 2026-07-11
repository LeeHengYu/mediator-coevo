# Task Instruction

## Task: Build Freshness Replenishment Plan Workbook

Create `/root/freshness_replenishment_plan_november_2025.xlsx` from source `/root/MealKits_Inventory_and_Inbound_Latest.xlsx`.

### Step 0: Inspect Source Data

1. Open `/root/MealKits_Inventory_and_Inbound_Latest.xlsx` and list all sheet names.
2. For each sheet (`Current Inventory`, `Incoming Deliveries`, `Shelf_Life`), print the first 15 rows to understand column names, data types, and layout.
3. Pay special attention to:
   - `Current Inventory!B1` (AsOfDate) and `Current Inventory!D1` (PlanningHorizonEnd) — note their exact values and types (datetime vs string).
   - The column that holds the entity identifier (likely `Meal_Kit_ID` or similar) and its exact spelling.
   - `Daily_Order_Rate_Boxes` or equivalent column name.
   - `Boxes_Expiring_By_Nov30` or equivalent — if not present, check `Shelf_Life` sheet for expiration/shelf-life data that lets you compute which boxes expire by Nov 30.
   - `Shelf_Life` sheet: look for a boxes-per-pallet conversion ratio column.
   - `Incoming Deliveries` sheet: identify the date column and quantity column, and the entity ID column.
4. Print all unique entity IDs from Current Inventory to confirm count and order.

### Step 1: Compute All Values in Python (using openpyxl + pandas)

Do all calculations in Python, writing final computed values (not Excel formulas) into the output workbook.

#### Metadata
- `AsOfDate` = value from `Current Inventory!B1` (convert to date if datetime)
- `PlanningHorizonEnd` = value from `Current Inventory!D1`
- `RemainingDaysInNovember` = (PlanningHorizonEnd - AsOfDate).days  (calendar day difference)

#### Per-entity calculations (preserve source order from Current Inventory):

1. **Current_Boxes**: from Current Inventory
2. **Boxes_Expiring_By_Nov30**: Determine from Shelf_Life data. If the source has an expiration date per entity or shelf life in days from a production/receipt date, compute how many boxes expire on or before PlanningHorizonEnd. If the source directly provides this field, use it.
3. **Usable_Current_Boxes** = max(0, Current_Boxes - Boxes_Expiring_By_Nov30)
4. **Daily_Order_Rate_Boxes**: from Current Inventory
5. **Current_DOH** = Usable_Current_Boxes / Daily_Order_Rate_Boxes when rate > 0, else leave blank (None)
6. **Projected_OOS_Date** = AsOfDate + timedelta(days=floor(Current_DOH)) when rate > 0, else blank. Store as ISO string YYYY-MM-DD.
7. **Inbound_Boxes_By_Nov30** = sum of inbound quantity for that entity where inbound date <= PlanningHorizonEnd
8. **Delivered_DOH_To_Nov30** = (Usable_Current_Boxes + Inbound_Boxes_By_Nov30) / Daily_Order_Rate_Boxes when rate > 0, else blank
9. **Remaining_November_Demand_Boxes** = Daily_Order_Rate_Boxes * RemainingDaysInNovember
10. **Additional_Boxes_Needed** = max(0, Remaining_November_Demand_Boxes - Usable_Current_Boxes - Inbound_Boxes_By_Nov30)
11. **Pallets_Required_Rounded_Up** = ceil(Additional_Boxes_Needed / boxes_per_pallet) when Additional_Boxes_Needed > 0, else 0. Get boxes_per_pallet from Shelf_Life sheet per entity.
12. **Earliest_Scheduled_Inbound_Date** = earliest inbound date for the entity from Incoming Deliveries (any date, not just ≤ Nov30). If none, blank. Store as ISO string.
13. **Required_Delivery_Date**:
    - blank when Pallets_Required_Rounded_Up == 0
    - else if Earliest_Scheduled_Inbound_Date is not blank AND Earliest_Scheduled_Inbound_Date <= Projected_OOS_Date: use AsOfDate + timedelta(days=floor(Delivered_DOH_To_Nov30))
    - else: use Projected_OOS_Date
    - Store as ISO string YYYY-MM-DD.
14. **Rounding_Applied** = Python boolean True when Additional_Boxes_Needed > 0 AND (Pallets_Required_Rounded_Up * boxes_per_pallet) != Additional_Boxes_Needed; else False
15. **Earlier_Delivery_Required** = Python boolean True when Pallets_Required_Rounded_Up > 0 AND (Earliest_Scheduled_Inbound_Date is blank OR Required_Delivery_Date < Earliest_Scheduled_Inbound_Date); else False

### Step 2: Write Output Workbook

Use openpyxl to create the workbook with exactly two sheets in order: `Freshness_Results`, `Additional_Freshness_Needed`.

#### Sheet 1: Freshness_Results

- A1="Field", B1="Value"
- A2="AsOfDate", B2=AsOfDate as YYYY-MM-DD string
- A3="PlanningHorizonEnd", B3=PlanningHorizonEnd as YYYY-MM-DD string
- A4="RemainingDaysInNovember", B4=integer
- Row 6: Header row with exactly these 16 columns in order:
  Meal_Kit_ID, Current_Boxes, Boxes_Expiring_By_Nov30, Usable_Current_Boxes, Daily_Order_Rate_Boxes, Current_DOH, Projected_OOS_Date, Inbound_Boxes_By_Nov30, Delivered_DOH_To_Nov30, Remaining_November_Demand_Boxes, Additional_Boxes_Needed, Pallets_Required_Rounded_Up, Required_Delivery_Date, Rounding_Applied, Earlier_Delivery_Required, Earliest_Scheduled_Inbound_Date
- Data rows starting at row 7, one per entity, same order as Current Inventory.
- Numeric fields must be numeric (int or float), not strings.
- Date fields (Projected_OOS_Date, Required_Delivery_Date, Earliest_Scheduled_Inbound_Date) must be ISO strings or None/blank.
- Boolean fields (Rounding_Applied, Earlier_Delivery_Required) must be Python booleans (True/False), not strings.

#### Sheet 2: Additional_Freshness_Needed

- Row 1 header: Meal_Kit_ID, Required_Delivery_Date, Pallets_Required_Rounded_Up, Additional_Boxes_Needed, Rounding_Applied, Earlier_Delivery_Required
- Include only rows where Pallets_Required_Rounded_Up > 0.
- Same entity order as in Freshness_Results.
- Same data types as Sheet 1.

### Step 3: Validate

1. Re-open the output file and verify:
   - Exactly 2 sheets with correct names in correct order.
   - Metadata cells A1:B4 on Freshness_Results.
   - Header at row 6 with all 16 column names.
   - Row count matches entity count from source.
   - Spot-check 2-3 entities: verify Current_DOH, Additional_Boxes_Needed, Pallets_Required_Rounded_Up manually.
   - Sheet 2 contains only entities with Pallets > 0.
   - Boolean fields are actual booleans, not strings.
   - Date fields are strings in YYYY-MM-DD format (not datetime objects).
2. Print summary: number of entities, number needing additional freshness, sample values.

### Important Notes
- Do NOT modify the source file.
- If any source column names differ from what's expected, adapt by inspecting actual column names first.
- If Boxes_Expiring_By_Nov30 is not a direct column in Current Inventory, compute it using Shelf_Life data (e.g., if shelf life in days from a date field means the product expires before Nov 30, count those boxes).
- Use `import math; math.floor()` and `math.ceil()` for floor/ceil operations.
- Row 5 of Freshness_Results should be empty (row 6 is the header).

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