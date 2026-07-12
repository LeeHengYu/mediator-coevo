# Task Instruction

## Task: Build the additional_shipments_needed_updated_july_2025.xlsx workbook

You must produce a single Excel workbook at `/root/additional_shipments_needed_updated_july_2025.xlsx` using data from `/root/Inventory_and_Shipments_Latest.xlsx`.

### Step-by-step instructions

#### Step 0 — Inspect the source workbook
1. Open `/root/Inventory_and_Shipments_Latest.xlsx` with openpyxl.
2. Print the sheet names.
3. For `Current Inventory`: print rows 1-5 to understand the layout (AsOfDate location, Month End location, SKU columns, daily rate columns). Print all data rows.
4. For `Incoming Shipments`: print the header row and all data rows. Note the column names for SKU, delivery date, and number of cases.
5. For `Ratio`: print all rows to find cases-per-pallet conversion factors per SKU.
6. **Print everything before writing any code.** Understanding the exact layout is critical.

#### Step 1 — Parse source data into Python structures
Using the printed layouts:
- `AsOfDate` = the date value in `Current Inventory` cell B1 (convert to `datetime.date`).
- `PlanningHorizonEnd` = the date value in `Current Inventory` cell D1 (convert to `datetime.date`).
- `RemainingDaysInJuly` = `(PlanningHorizonEnd - AsOfDate).days` — for July 4 to July 31 this should be 27.
- Build a list of SKU records from `Current Inventory` preserving source row order. Each record: SKU name, Current_Cases, Daily_Rate_Cases_Per_Day.
- Build a dict mapping SKU → list of (delivery_date, cases_left) from `Incoming Shipments`.
- Build a dict mapping SKU → Cases_Per_Pallet from `Ratio`.

#### Step 2 — Compute per-SKU fields
For each SKU, compute all 14 columns for Sheet 1 exactly as specified:

1. `Product_SKU` — SKU name string
2. `Current_Cases` — numeric from source
3. `Daily_Rate_Cases_Per_Day` — numeric from source
4. `Current_DOH` = Current_Cases / Daily_Rate if rate > 0, else None
5. `Projected_OOS_Date` = AsOfDate + timedelta(days=floor(Current_DOH)) if rate > 0, else None → store as ISO string "YYYY-MM-DD" or None
6. `Inbound_Cases_By_July31` = sum of cases for that SKU where delivery_date <= PlanningHorizonEnd
7. `Delivered_DOH_To_July31` = (Current_Cases + Inbound_Cases_By_July31) / Daily_Rate if rate > 0, else None
8. `Remaining_July_Demand_Cases` = Daily_Rate * RemainingDaysInJuly
9. `Additional_Cases_Needed` = max(0, Remaining_July_Demand_Cases - Current_Cases - Inbound_Cases_By_July31)
10. `Pallets_Required_Rounded_Up` = math.ceil(Additional_Cases_Needed / Cases_Per_Pallet) if Additional_Cases_Needed > 0, else 0
11. `Required_Delivery_Date`:
    - If Pallets_Required_Rounded_Up == 0: None (blank)
    - Else if Earliest_Scheduled_Inbound_Date is not None AND Earliest_Scheduled_Inbound_Date <= Projected_OOS_Date (as dates): AsOfDate + timedelta(days=floor(Delivered_DOH_To_July31)) → ISO string
    - Else: Projected_OOS_Date (already ISO string)
12. `Rounding_Applied` = True if Additional_Cases_Needed > 0 AND (Additional_Cases_Needed / Cases_Per_Pallet) != ceil(Additional_Cases_Needed / Cases_Per_Pallet); else False. **Store as Python bool so openpyxl writes TRUE/FALSE.**
13. `Earlier_Delivery_Required` = True if Pallets_Required_Rounded_Up > 0 AND (Earliest_Scheduled_Inbound_Date is None OR Required_Delivery_Date < Earliest_Scheduled_Inbound_Date as date comparison); else False. **Store as Python bool.**
14. `Earliest_Scheduled_Inbound_Date` = min of delivery dates for that SKU from Incoming Shipments, else None → ISO string or None

**Critical date handling**: When comparing dates for Required_Delivery_Date logic, convert ISO strings back to date objects. When writing to Excel, date columns E, K, N must contain ISO strings ("YYYY-MM-DD"), not datetime objects.

#### Step 3 — Write the output workbook
Use openpyxl to create a new workbook.

**Sheet 1: `SKU_Results`**
- Remove the default sheet after creating named sheets, or rename it.
- Metadata cells:
  - A1="Field", B1="Value"
  - A2="AsOfDate", B2=AsOfDate as "YYYY-MM-DD" string
  - A3="PlanningHorizonEnd", B3=PlanningHorizonEnd as "YYYY-MM-DD" string
  - A4="RemainingDaysInJuly", B4=integer (e.g., 27)
- Row 6 (A6:N6): exact header names in order:
  `Product_SKU`, `Current_Cases`, `Daily_Rate_Cases_Per_Day`, `Current_DOH`, `Projected_OOS_Date`, `Inbound_Cases_By_July31`, `Delivered_DOH_To_July31`, `Remaining_July_Demand_Cases`, `Additional_Cases_Needed`, `Pallets_Required_Rounded_Up`, `Required_Delivery_Date`, `Rounding_Applied`, `Earlier_Delivery_Required`, `Earliest_Scheduled_Inbound_Date`
- Data rows start at row 7, one per SKU in source order.
- Numeric fields must be numbers (int or float), not strings.
- Boolean fields (Rounding_Applied, Earlier_Delivery_Required) must be Python bools.
- Date fields (cols E, K, N) must be strings in "YYYY-MM-DD" format, or None for blank.

**Sheet 2: `Additional_Shipments_Needed`**
- Header at row 1: `Product_SKU`, `Required_Delivery_Date`, `Pallets_Required_Rounded_Up`, `Additional_Cases_Needed`, `Rounding_Applied`, `Earlier_Delivery_Required`
- Include only SKUs where Pallets_Required_Rounded_Up > 0, same order as Sheet 1.
- Same data types: numerics as numbers, bools as bools, dates as ISO strings.

Ensure the workbook has exactly two sheets in order: `SKU_Results`, `Additional_Shipments_Needed`. Delete any default "Sheet" sheet.

Save to `/root/additional_shipments_needed_updated_july_2025.xlsx`.

#### Step 4 — Validate
1. Re-open the output file with openpyxl.
2. Verify sheet names are exactly `['SKU_Results', 'Additional_Shipments_Needed']`.
3. Print the metadata cells (A1:B4) from SKU_Results.
4. Print the header row (row 6) from SKU_Results.
5. Print all data rows from SKU_Results.
6. Print all rows from Additional_Shipments_Needed.
7. Verify:
   - B4 is an integer
   - Date columns contain strings matching YYYY-MM-DD pattern or are None
   - Boolean columns contain True/False
   - Numeric columns contain numbers
   - Additional_Shipments_Needed only has rows with Pallets > 0
   - Row count in Sheet 2 <= row count in Sheet 1 data

Do NOT modify the source file. The final deliverable is the output .xlsx file.

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