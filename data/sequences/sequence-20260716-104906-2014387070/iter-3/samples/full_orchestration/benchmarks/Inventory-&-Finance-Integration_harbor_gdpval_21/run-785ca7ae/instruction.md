# Task Instruction

Execute the following steps in order:

## Step 0 – Inspect the source workbook

```python
import openpyxl, json
wb = openpyxl.load_workbook('/root/Inventory_and_Shipments_Latest.xlsx', data_only=True)
for name in wb.sheetnames:
    ws = wb[name]
    print(f'\n=== {name} (rows={ws.max_row}, cols={ws.max_column}) ===')
    for r in range(1, min(ws.max_row+1, 25)):
        row_vals = []
        for c in range(1, min(ws.max_column+1, 15)):
            row_vals.append(ws.cell(r, c).value)
        print(r, row_vals)
wb.close()
```

Carefully note:
- The exact cell where AsOfDate lives (expected `Current Inventory!B1`) and PlanningHorizonEnd (`Current Inventory!D1`).
- The row where SKU data starts, the column layout (SKU name, current cases, daily rate, month-end date).
- `Incoming Shipments` columns: SKU, Delivery Date, Number of Cases Left, etc.
- `Ratio` sheet: SKU and Cases_Per_Pallet mapping.
- Data types of dates (datetime objects vs strings).

Print everything you need; do NOT guess.

## Step 1 – Build the output workbook

Write a single Python script (using openpyxl) that:

1. Reads all source data into Python structures (dicts/lists).
2. Converts any date objects to `datetime.date` for safe comparison.
3. Computes every field per the rules below.
4. Writes the two sheets in the exact order: `SKU_Results` first, `Additional_Shipments_Needed` second.
5. Saves to `/root/additional_shipments_needed_updated_july_2025.xlsx`.

### Detailed calculation rules (follow exactly):

- `AsOfDate` = value from Current Inventory B1 (as date).
- `PlanningHorizonEnd` = value from Current Inventory D1 (as date).
- `RemainingDaysInJuly` = (PlanningHorizonEnd - AsOfDate).days  (e.g., July 31 - July 4 = 27).
- For each SKU (preserve source row order from Current Inventory):
  - `Current_Cases` = from source
  - `Daily_Rate_Cases_Per_Day` = from source
  - `Current_DOH` = Current_Cases / Daily_Rate if rate > 0, else None
  - `Projected_OOS_Date` = AsOfDate + timedelta(days=floor(Current_DOH)) if rate > 0, else None → store as ISO string YYYY-MM-DD
  - `Inbound_Cases_By_July31` = sum of "Number of Cases Left" from Incoming Shipments for this SKU where Delivery Date <= PlanningHorizonEnd
  - `Delivered_DOH_To_July31` = (Current_Cases + Inbound_Cases_By_July31) / Daily_Rate if rate > 0, else None
  - `Remaining_July_Demand_Cases` = Daily_Rate * RemainingDaysInJuly
  - `Additional_Cases_Needed` = max(0, Remaining_July_Demand_Cases - Current_Cases - Inbound_Cases_By_July31)
  - Look up `Cases_Per_Pallet` from Ratio sheet for this SKU.
  - `Pallets_Required_Rounded_Up` = math.ceil(Additional_Cases_Needed / Cases_Per_Pallet) if Additional_Cases_Needed > 0 else 0
  - `Rounding_Applied`: TRUE if Additional_Cases_Needed > 0 AND (Additional_Cases_Needed / Cases_Per_Pallet) != ceil(Additional_Cases_Needed / Cases_Per_Pallet); else FALSE. Store as Python bool.
  - `Earliest_Scheduled_Inbound_Date` = earliest Delivery Date for this SKU in Incoming Shipments, else None → ISO string or None
  - `Required_Delivery_Date`:
    - None (blank) if Pallets_Required_Rounded_Up == 0
    - else if Earliest_Scheduled_Inbound_Date is not None AND Earliest_Scheduled_Inbound_Date <= Projected_OOS_Date (compare as dates): AsOfDate + timedelta(days=floor(Delivered_DOH_To_July31)) → ISO string
    - else: Projected_OOS_Date (already ISO string)
  - `Earlier_Delivery_Required`: TRUE if Pallets_Required_Rounded_Up > 0 AND (Earliest_Scheduled_Inbound_Date is None OR Required_Delivery_Date < Earliest_Scheduled_Inbound_Date); else FALSE. Compare as date strings or dates consistently. Store as Python bool.

### SKU_Results sheet layout:
- A1='Field', B1='Value'
- A2='AsOfDate', B2=AsOfDate as YYYY-MM-DD string
- A3='PlanningHorizonEnd', B3=PlanningHorizonEnd as YYYY-MM-DD string
- A4='RemainingDaysInJuly', B4=integer
- Row 6 headers (A6:N6): Product_SKU, Current_Cases, Daily_Rate_Cases_Per_Day, Current_DOH, Projected_OOS_Date, Inbound_Cases_By_July31, Delivered_DOH_To_July31, Remaining_July_Demand_Cases, Additional_Cases_Needed, Pallets_Required_Rounded_Up, Required_Delivery_Date, Rounding_Applied, Earlier_Delivery_Required, Earliest_Scheduled_Inbound_Date
- Data rows start at row 7, one per SKU.
- Columns E (Projected_OOS_Date), K (Required_Delivery_Date), N (Earliest_Scheduled_Inbound_Date) must be ISO date strings, not Excel date serial numbers.
- Numeric fields (B-D, F-J) must be numbers, not strings.
- Boolean fields (L, M) must be Python booleans (True/False) so openpyxl writes them as Excel booleans.

### Additional_Shipments_Needed sheet layout:
- Row 1 headers: Product_SKU, Required_Delivery_Date, Pallets_Required_Rounded_Up, Additional_Cases_Needed, Rounding_Applied, Earlier_Delivery_Required
- Include only SKUs where Pallets_Required_Rounded_Up > 0, same order as SKU_Results.
- Same data types as above.

## Step 2 – Validate the output

Open the saved file with openpyxl and print:
- Sheet names
- Metadata cells A1:B4 from SKU_Results
- Headers at row 6
- First 5 data rows with all 14 columns
- All rows from Additional_Shipments_Needed
- Confirm date columns are strings, numeric columns are numbers, boolean columns are bools.

Fix any issues found and re-save if needed.

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