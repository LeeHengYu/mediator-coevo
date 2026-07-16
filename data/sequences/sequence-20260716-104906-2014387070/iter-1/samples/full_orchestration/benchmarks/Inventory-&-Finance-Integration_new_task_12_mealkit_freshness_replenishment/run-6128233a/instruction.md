# Task Instruction

## Task: Build Freshness Replenishment Plan Workbook

Create `/root/freshness_replenishment_plan_november_2025.xlsx` from source `/root/MealKits_Inventory_and_Inbound_Latest.xlsx`.

### Step 0 — Inspect Source Workbook

Before writing any code, inspect the source workbook thoroughly:

```python
import openpyxl
wb = openpyxl.load_workbook('/root/MealKits_Inventory_and_Inbound_Latest.xlsx', data_only=True)
for name in wb.sheetnames:
    ws = wb[name]
    print(f'\n=== Sheet: {name} ===')
    print(f'Dimensions: {ws.dimensions}')
    for row in ws.iter_rows(min_row=1, max_row=min(ws.max_row, 20), values_only=False):
        print([(c.coordinate, c.value) for c in row])
```

Record:
- The exact cell locations of AsOfDate (Current Inventory!B1) and PlanningHorizonEnd (Current Inventory!D1) — note their types (datetime vs string).
- The header row and data rows in Current Inventory — identify columns for Meal_Kit_ID, Current_Boxes, Daily_Order_Rate_Boxes, and Boxes_Expiring_By_Nov30.
- The Incoming Deliveries sheet — identify columns for entity ID, quantity, and delivery date.
- The Shelf_Life sheet — identify the conversion ratio (boxes per pallet) per entity, and any other relevant fields.

Print ALL rows of ALL sheets (if they are small) so you have complete data. Do NOT proceed until you have confirmed every column name and data layout.

### Step 1 — Write the Output Workbook

Using openpyxl (do NOT use formulas — compute all values in Python and write them as static values), create the output workbook with exactly two sheets in order: `Freshness_Results`, `Additional_Freshness_Needed`.

#### Sheet 1: Freshness_Results

**Metadata (rows 1-4):**
- A1="Field", B1="Value"
- A2="AsOfDate", B2=AsOfDate as string "YYYY-MM-DD"
- A3="PlanningHorizonEnd", B3=PlanningHorizonEnd as string "YYYY-MM-DD"
- A4="RemainingDaysInNovember", B4=integer (PlanningHorizonEnd - AsOfDate).days

**Row 5:** Leave blank.

**Row 6:** Header row with exactly these 16 column names:
Meal_Kit_ID, Current_Boxes, Boxes_Expiring_By_Nov30, Usable_Current_Boxes, Daily_Order_Rate_Boxes, Current_DOH, Projected_OOS_Date, Inbound_Boxes_By_Nov30, Delivered_DOH_To_Nov30, Remaining_November_Demand_Boxes, Additional_Boxes_Needed, Pallets_Required_Rounded_Up, Required_Delivery_Date, Rounding_Applied, Earlier_Delivery_Required, Earliest_Scheduled_Inbound_Date

**Data rows (starting row 7):** One row per entity from Current Inventory, preserving source order.

#### Calculation Rules (apply carefully in order):

```
AsOfDate = date from Current Inventory!B1
PlanningHorizonEnd = date from Current Inventory!D1
RemainingDaysInNovember = (PlanningHorizonEnd - AsOfDate).days

For each entity:
  Current_Boxes = from source
  Boxes_Expiring_By_Nov30 = from source (look up in source; this might be a column in Current Inventory or computed from Shelf_Life — inspect carefully)
  Daily_Order_Rate_Boxes = from source
  
  Usable_Current_Boxes = max(0, Current_Boxes - Boxes_Expiring_By_Nov30)
  
  Current_DOH = Usable_Current_Boxes / Daily_Order_Rate_Boxes   (if rate > 0, else None)
  
  Projected_OOS_Date = AsOfDate + timedelta(days=floor(Current_DOH))   (if rate > 0, else None)
  # Store as ISO string "YYYY-MM-DD"
  
  Inbound_Boxes_By_Nov30 = sum of inbound quantities for this entity where delivery_date <= PlanningHorizonEnd
  
  Delivered_DOH_To_Nov30 = (Usable_Current_Boxes + Inbound_Boxes_By_Nov30) / Daily_Order_Rate_Boxes   (if rate > 0, else None)
  
  Remaining_November_Demand_Boxes = Daily_Order_Rate_Boxes * RemainingDaysInNovember
  
  Additional_Boxes_Needed = max(0, Remaining_November_Demand_Boxes - Usable_Current_Boxes - Inbound_Boxes_By_Nov30)
  
  # Get boxes_per_pallet from Shelf_Life sheet for this entity
  Pallets_Required_Rounded_Up = ceil(Additional_Boxes_Needed / boxes_per_pallet) if Additional_Boxes_Needed > 0 else 0
  
  Earliest_Scheduled_Inbound_Date = earliest delivery date for this entity from Incoming Deliveries (any date, not just <= Nov30), else None
  # Store as ISO string "YYYY-MM-DD" or leave blank
  
  Rounding_Applied:
    if Additional_Boxes_Needed > 0 and (Pallets_Required_Rounded_Up * boxes_per_pallet) != Additional_Boxes_Needed:
      TRUE
    else:
      FALSE
  # Store as Python boolean True/False
  
  Required_Delivery_Date:
    if Pallets_Required_Rounded_Up == 0:
      None (blank)
    elif Earliest_Scheduled_Inbound_Date is not None and Earliest_Scheduled_Inbound_Date <= Projected_OOS_Date:
      AsOfDate + timedelta(days=floor(Delivered_DOH_To_Nov30))
    else:
      Projected_OOS_Date
  # Store as ISO string "YYYY-MM-DD"
  
  Earlier_Delivery_Required:
    if Pallets_Required_Rounded_Up > 0 and (Earliest_Scheduled_Inbound_Date is None or Required_Delivery_Date < Earliest_Scheduled_Inbound_Date):
      TRUE
    else:
      FALSE
  # Store as Python boolean True/False
```

**CRITICAL type rules:**
- Numeric fields (Current_Boxes, Boxes_Expiring_By_Nov30, Usable_Current_Boxes, Daily_Order_Rate_Boxes, Current_DOH, Inbound_Boxes_By_Nov30, Delivered_DOH_To_Nov30, Remaining_November_Demand_Boxes, Additional_Boxes_Needed, Pallets_Required_Rounded_Up) must be written as Python int or float (NOT strings).
- Date fields (Projected_OOS_Date, Required_Delivery_Date, Earliest_Scheduled_Inbound_Date) must be written as ISO strings "YYYY-MM-DD".
- Boolean fields (Rounding_Applied, Earlier_Delivery_Required) must be written as Python `True` or `False` (openpyxl will store them as Excel booleans).
- Blank/None fields: write None (openpyxl leaves cell empty).

#### Sheet 2: Additional_Freshness_Needed

**Row 1:** Header: Meal_Kit_ID, Required_Delivery_Date, Pallets_Required_Rounded_Up, Additional_Boxes_Needed, Rounding_Applied, Earlier_Delivery_Required

**Data rows (starting row 2):** Only entities where Pallets_Required_Rounded_Up > 0, in same order as Freshness_Results. Same types as Sheet 1.

### Step 2 — Validate Output

After writing the file, re-open it and:
1. Confirm exactly 2 sheets in correct order.
2. Print metadata cells A1:B4.
3. Print header row 6.
4. Print all data rows with their types (use `type(cell.value)` for each cell).
5. Confirm date columns contain strings, numeric columns contain numbers, boolean columns contain booleans.
6. Print Sheet 2 headers and all rows.
7. Confirm Sheet 2 only has rows where Pallets_Required_Rounded_Up > 0.

### Important Notes
- Do NOT modify the source file.
- Inspect the Shelf_Life sheet carefully for the boxes-per-pallet conversion ratio — it may be called something like "Boxes_Per_Pallet" or "Conversion_Ratio" or similar.
- Inspect Current Inventory carefully for Boxes_Expiring_By_Nov30 — determine whether this is a direct column or needs computation from Shelf_Life dates.
- If Boxes_Expiring_By_Nov30 needs computation: for each entity, check if (AsOfDate + shelf_life_days) or (production_date + shelf_life_days) falls before or on PlanningHorizonEnd. If the expiry date <= PlanningHorizonEnd, then Boxes_Expiring_By_Nov30 = Current_Boxes; otherwise 0. Inspect the actual data to determine the correct logic.
- Use `import math; math.floor()` and `math.ceil()` for floor/ceil operations.
- When comparing dates, ensure consistent types (both datetime.date or both datetime.datetime).

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