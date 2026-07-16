# Task Instruction

Create the output workbook `/root/additional_shipments_needed_updated_july_2025.xlsx` by reading the source workbook `/root/Inventory_and_Shipments_Latest.xlsx` and performing all calculations as specified. Follow these steps precisely:

## Step 1: Inspect the source workbook

Open `/root/Inventory_and_Shipments_Latest.xlsx` with openpyxl and inspect:
- `Current Inventory` sheet: print rows 1-5 to understand the layout, especially B1 (AsOfDate) and D1 (PlanningHorizonEnd). Print all data rows to see SKU names, inventory amounts, and daily rates. Note the exact column positions.
- `Incoming Shipments` sheet: print all rows to see column headers and data (SKU, delivery dates, number of cases).
- `Ratio` sheet: print all rows to see SKU-to-cases-per-pallet mapping.

Print everything before writing any code that creates the output.

## Step 2: Parse the source data

Using openpyxl (data_only=False is fine since we need raw values):
- Extract `AsOfDate` from `Current Inventory!B1` and `PlanningHorizonEnd` from `Current Inventory!D1`. These should be dates. If they are datetime objects, convert to date. Format as `YYYY-MM-DD` strings where needed.
- `RemainingDaysInJuly` = (PlanningHorizonEnd - AsOfDate).days — this should be 27 for July 4 to July 31.
- For each SKU row in `Current Inventory`, extract: SKU name, Current_Cases (numeric), Daily_Rate_Cases_Per_Day (numeric). Preserve the source row order.
- For `Incoming Shipments`, build a dict: SKU -> list of (delivery_date, cases_left). Identify the correct columns by header names.
- For `Ratio`, build a dict: SKU -> Cases_Per_Pallet.

## Step 3: Compute per-SKU values

For each SKU (in source order):

1. `Current_DOH` = Current_Cases / Daily_Rate if rate > 0, else None
2. `Projected_OOS_Date` = AsOfDate + timedelta(days=floor(Current_DOH)) if rate > 0, else None. Store as date, format as ISO string.
3. `Inbound_Cases_By_July31` = sum of cases for that SKU where delivery_date <= PlanningHorizonEnd. If no shipments, 0.
4. `Delivered_DOH_To_July31` = (Current_Cases + Inbound_Cases_By_July31) / Daily_Rate if rate > 0, else None
5. `Remaining_July_Demand_Cases` = Daily_Rate * RemainingDaysInJuly
6. `Additional_Cases_Needed` = max(0, Remaining_July_Demand_Cases - Current_Cases - Inbound_Cases_By_July31)
7. `Cases_Per_Pallet` from Ratio sheet for this SKU
8. `Pallets_Required_Rounded_Up` = math.ceil(Additional_Cases_Needed / Cases_Per_Pallet) if Additional_Cases_Needed > 0, else 0
9. `Earliest_Scheduled_Inbound_Date` = min of delivery dates for this SKU from Incoming Shipments, else None. Format as ISO string.
10. `Required_Delivery_Date`:
    - If Pallets_Required_Rounded_Up == 0: None (blank)
    - Else if Earliest_Scheduled_Inbound_Date is not None and Earliest_Scheduled_Inbound_Date <= Projected_OOS_Date: AsOfDate + timedelta(days=floor(Delivered_DOH_To_July31))
    - Else: Projected_OOS_Date
    - Format as ISO string.
11. `Rounding_Applied`: TRUE if Additional_Cases_Needed > 0 AND ceil(Additional_Cases_Needed/Cases_Per_Pallet) != Additional_Cases_Needed/Cases_Per_Pallet (i.e., there was actual rounding); else FALSE. Store as Python bool True/False.
12. `Earlier_Delivery_Required`: TRUE if Pallets_Required_Rounded_Up > 0 AND (Earliest_Scheduled_Inbound_Date is None OR Required_Delivery_Date < Earliest_Scheduled_Inbound_Date); else FALSE. Store as Python bool.

## Step 4: Write the output workbook

Use openpyxl to create a new workbook. Remove the default sheet after creating named sheets.

### Sheet 1: `SKU_Results`
- A1='Field', B1='Value'
- A2='AsOfDate', B2=AsOfDate as ISO string 'YYYY-MM-DD'
- A3='PlanningHorizonEnd', B3=PlanningHorizonEnd as ISO string
- A4='RemainingDaysInJuly', B4=integer value
- Row 6 (A6:N6): exact headers as specified: Product_SKU, Current_Cases, Daily_Rate_Cases_Per_Day, Current_DOH, Projected_OOS_Date, Inbound_Cases_By_July31, Delivered_DOH_To_July31, Remaining_July_Demand_Cases, Additional_Cases_Needed, Pallets_Required_Rounded_Up, Required_Delivery_Date, Rounding_Applied, Earlier_Delivery_Required, Earliest_Scheduled_Inbound_Date
- Data rows starting at row 7, one per SKU in source order.
- Date columns (E=Projected_OOS_Date, K=Required_Delivery_Date, N=Earliest_Scheduled_Inbound_Date): write as ISO format strings 'YYYY-MM-DD' or leave blank (None).
- Numeric fields must be numeric (int or float), not strings.
- Boolean fields (L=Rounding_Applied, M=Earlier_Delivery_Required): write as Python bool True/False so openpyxl stores them as Excel booleans.

### Sheet 2: `Additional_Shipments_Needed`
- Row 1 headers: Product_SKU, Required_Delivery_Date, Pallets_Required_Rounded_Up, Additional_Cases_Needed, Rounding_Applied, Earlier_Delivery_Required
- Include only SKUs where Pallets_Required_Rounded_Up > 0, in same order as SKU_Results.
- Same data types: dates as ISO strings, numbers as numbers, booleans as booleans.

Save to `/root/additional_shipments_needed_updated_july_2025.xlsx`.

## Step 5: Validate

After saving, re-open the output file with openpyxl and:
1. Verify sheet names are exactly ['SKU_Results', 'Additional_Shipments_Needed']
2. Print metadata cells A1:B4 from SKU_Results
3. Print header row 6 from SKU_Results
4. Print all data rows from SKU_Results
5. Print all rows from Additional_Shipments_Needed
6. Verify that B4 (RemainingDaysInJuly) is an integer
7. Verify date fields are strings in YYYY-MM-DD format
8. Verify numeric fields are numbers, not strings
9. Verify boolean fields are booleans
10. Verify Additional_Shipments_Needed only contains rows with Pallets > 0

Do NOT modify the source file. Print intermediate results at each step for debugging.

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