# Task Instruction

Create a Python script `/root/solve.py` and execute it to produce the output workbook `/root/additional_shipments_needed_updated_july_2025.xlsx`.

The script should:

1. **Read the source workbook** `/root/Inventory_and_Shipments_Latest.xlsx` using `openpyxl` (with `data_only=True` to get computed values if needed, but also try without to inspect formulas). Read all three sheets:
   - `Current Inventory` — get `AsOfDate` from cell B1, `PlanningHorizonEnd` (Month End) from cell D1, SKU data rows (product names, current cases, daily rate).
   - `Incoming Shipments` — get scheduled deliveries with columns for SKU, delivery date, and number of cases left.
   - `Ratio` — get cases-per-pallet conversion for each SKU.

2. **Before coding calculations, inspect and print** the first ~10 rows and headers of each sheet to understand exact column positions, header names, and data types. Print cells B1 and D1 of `Current Inventory` to confirm date values. Print a few rows of `Incoming Shipments` and `Ratio` to confirm column layout.

3. **Parse dates carefully:**
   - `AsOfDate` and `PlanningHorizonEnd` should be `datetime.date` objects. If they are strings, parse them. If they are datetime objects, extract `.date()`.
   - `RemainingDaysInJuly = (PlanningHorizonEnd - AsOfDate).days` — for July 4 to July 31 this should be 27.

4. **For each SKU** (preserving source row order from `Current Inventory`):
   - `Current_Cases` = inventory cases (numeric)
   - `Daily_Rate_Cases_Per_Day` = daily rate (numeric)
   - `Current_DOH` = `Current_Cases / Daily_Rate` if rate > 0, else `None`
   - `Projected_OOS_Date` = `AsOfDate + timedelta(days=floor(Current_DOH))` if rate > 0, else `None`
   - `Inbound_Cases_By_July31` = sum of `Number of Cases Left` from `Incoming Shipments` for that SKU where `Delivery Date <= PlanningHorizonEnd`
   - `Delivered_DOH_To_July31` = `(Current_Cases + Inbound_Cases_By_July31) / Daily_Rate` if rate > 0, else `None`
   - `Remaining_July_Demand_Cases` = `Daily_Rate * RemainingDaysInJuly`
   - `Additional_Cases_Needed` = `max(0, Remaining_July_Demand_Cases - Current_Cases - Inbound_Cases_By_July31)`
   - Look up `Cases_Per_Pallet` from the `Ratio` sheet for this SKU
   - `Pallets_Required_Rounded_Up` = `math.ceil(Additional_Cases_Needed / Cases_Per_Pallet)` if additional > 0, else `0`
   - `Rounding_Applied` = `True` if additional > 0 AND `Additional_Cases_Needed / Cases_Per_Pallet != ceil(...)` (i.e., not an exact integer), else `False`
   - `Earliest_Scheduled_Inbound_Date` = earliest delivery date for this SKU in `Incoming Shipments` (any date, not limited to <=July31), else `None`
   - `Required_Delivery_Date`:
     - `None` if `Pallets_Required_Rounded_Up == 0`
     - else if `Earliest_Scheduled_Inbound_Date` is not None AND `Earliest_Scheduled_Inbound_Date <= Projected_OOS_Date`: use `AsOfDate + timedelta(days=floor(Delivered_DOH_To_July31))`
     - else: use `Projected_OOS_Date`
   - `Earlier_Delivery_Required` = `True` if pallets > 0 AND (earliest date is None OR `Required_Delivery_Date < Earliest_Scheduled_Inbound_Date`), else `False`

5. **Write the output workbook** using `openpyxl`:

   **Sheet 1: `SKU_Results`**
   - A1='Field', B1='Value'
   - A2='AsOfDate', B2=AsOfDate as 'YYYY-MM-DD' string
   - A3='PlanningHorizonEnd', B3=PlanningHorizonEnd as 'YYYY-MM-DD' string
   - A4='RemainingDaysInJuly', B4=integer
   - Row 6: exact headers in order: Product_SKU, Current_Cases, Daily_Rate_Cases_Per_Day, Current_DOH, Projected_OOS_Date, Inbound_Cases_By_July31, Delivered_DOH_To_July31, Remaining_July_Demand_Cases, Additional_Cases_Needed, Pallets_Required_Rounded_Up, Required_Delivery_Date, Rounding_Applied, Earlier_Delivery_Required, Earliest_Scheduled_Inbound_Date
   - Data rows start at row 7. One row per SKU in source order.
   - Date columns (E=Projected_OOS_Date, K=Required_Delivery_Date, N=Earliest_Scheduled_Inbound_Date) must be ISO strings 'YYYY-MM-DD' or None/blank.
   - Boolean columns (L=Rounding_Applied, M=Earlier_Delivery_Required) must be Python `True`/`False` booleans so openpyxl writes them as Excel booleans.
   - Numeric fields must be numbers, not strings.

   **Sheet 2: `Additional_Shipments_Needed`**
   - Row 1 headers: Product_SKU, Required_Delivery_Date, Pallets_Required_Rounded_Up, Additional_Cases_Needed, Rounding_Applied, Earlier_Delivery_Required
   - Include only SKUs where `Pallets_Required_Rounded_Up > 0`, same order as SKU_Results.
   - Same data type rules: dates as ISO strings, booleans as booleans, numbers as numbers.

6. **Save** to `/root/additional_shipments_needed_updated_july_2025.xlsx`.

7. **Validate** after writing:
   - Re-open the output file and print sheet names, row counts, metadata cells, first few data rows of each sheet, and spot-check a couple of SKU calculations to confirm correctness.
   - Confirm `RemainingDaysInJuly` matches expected value.
   - Confirm date strings are in YYYY-MM-DD format.
   - Confirm boolean cells are actual booleans.

**Important notes:**
- SKU matching between sheets may need normalization (strip whitespace, case-insensitive comparison). Print any SKUs that don't match across sheets.
- If `Current Inventory` has header rows at different positions, adapt accordingly after inspection.
- Do NOT modify the source file.
- Use `import math` for `math.ceil` and `math.floor`, `from datetime import date, timedelta` for date operations.
- Handle edge cases: SKUs with zero daily rate, SKUs with no incoming shipments, SKUs not in Ratio sheet (print warning and skip or use a default).

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