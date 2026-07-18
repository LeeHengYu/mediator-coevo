# Task Instruction

Execute the following steps to build the required Excel workbook.

## Step 1: Inspect the source workbook

```python
import openpyxl
wb = openpyxl.load_workbook('/root/Maintenance_Parts_and_Deliveries_Latest.xlsx', data_only=True)
for name in wb.sheetnames:
    ws = wb[name]
    print(f'\n=== Sheet: {name} ===')
    print(f'Dimensions: {ws.dimensions}')
    for row in ws.iter_rows(min_row=1, max_row=min(ws.max_row, 20), values_only=False):
        print([(c.coordinate, c.value) for c in row])
wb.close()
```

Run this and read the output carefully. Identify:
- The exact layout of "Current Parts" (where AsOfDate is in B1, PlanningHorizonEnd in D1, where the parts data starts, column names)
- The exact layout of "Scheduled Deliveries" (columns for part code, inbound quantity, inbound date)
- The exact layout of "Ratio" (the conversion ratio per part or a single ratio)

## Step 2: Build the output workbook

After inspecting, write a single Python script that:

### 2a. Read source data
- Parse AsOfDate from Current Parts B1 and PlanningHorizonEnd from Current Parts D1. These may be datetime objects or strings; handle both. Convert to `datetime.date`.
- Compute RemainingDaysInSeptember = (PlanningHorizonEnd - AsOfDate).days
- Read all parts rows from Current Parts (identify the header row and data rows). Extract Part_Code, Current_Units, Daily_Consumption_Units for each part, preserving source order.
- Read all delivery rows from Scheduled Deliveries. For each row extract part code, inbound quantity, and inbound/delivery date.
- Read the Ratio sheet. Determine if it's a single ratio or per-part. Extract the conversion ratio(s) (units per crate).

### 2b. Compute Part_Results for each part (in source order)

For each part:
1. Current_DOH = Current_Units / Daily_Consumption_Units if Daily_Consumption_Units > 0, else None
2. Projected_Stockout_Date = AsOfDate + timedelta(days=floor(Current_DOH)) if rate > 0, else None
3. Inbound_Units_By_Sep30 = sum of inbound quantities for this part where inbound_date <= PlanningHorizonEnd
4. Delivered_DOH_To_Sep30 = (Current_Units + Inbound_Units_By_Sep30) / Daily_Consumption_Units if rate > 0, else None
5. Remaining_September_Demand_Units = Daily_Consumption_Units * RemainingDaysInSeptember
6. Additional_Units_Needed = max(0, Remaining_September_Demand_Units - Current_Units - Inbound_Units_By_Sep30)
7. Look up the conversion ratio for this part from the Ratio sheet.
8. If Additional_Units_Needed > 0: Crates_Required_Rounded_Up = math.ceil(Additional_Units_Needed / ratio); else 0
9. Earliest_Scheduled_Delivery_Date = earliest inbound date for this part (across ALL deliveries, not just those <= Sep30), else None
10. Required_Delivery_Date:
    - None if Crates_Required_Rounded_Up == 0
    - else if Earliest_Scheduled_Delivery_Date is not None and Earliest_Scheduled_Delivery_Date <= Projected_Stockout_Date: AsOfDate + timedelta(days=floor(Delivered_DOH_To_Sep30))
    - else: Projected_Stockout_Date
11. Rounding_Applied = TRUE if Additional_Units_Needed > 0 and (Crates_Required_Rounded_Up * ratio) != Additional_Units_Needed; else FALSE
12. Earlier_Delivery_Required = TRUE if Crates_Required_Rounded_Up > 0 and (Earliest_Scheduled_Delivery_Date is None or Required_Delivery_Date < Earliest_Scheduled_Delivery_Date); else FALSE

### 2c. Write the output workbook using openpyxl

Create a new workbook. Remove the default sheet after creating the two named sheets.

**Sheet 1: Part_Results**
- A1="Field", B1="Value"
- A2="AsOfDate", B2=AsOfDate as YYYY-MM-DD string
- A3="PlanningHorizonEnd", B3=PlanningHorizonEnd as YYYY-MM-DD string
- A4="RemainingDaysInSeptember", B4=integer value
- Row 6: the 14 column headers exactly as specified
- Data rows starting at row 7, one per part in source order
- Date columns (Projected_Stockout_Date, Required_Delivery_Date, Earliest_Scheduled_Delivery_Date) must be ISO YYYY-MM-DD strings (not datetime objects). Use str(date_val) or date_val.strftime('%Y-%m-%d'). Write None/blank for missing dates.
- Boolean fields (Rounding_Applied, Earlier_Delivery_Required) must be Python bool True/False so openpyxl writes them as Excel booleans.
- Numeric fields must be int or float, not strings.

**Sheet 2: Additional_Resupply_Needed**
- Row 1: the 6 column headers exactly as specified
- Include only parts where Crates_Required_Rounded_Up > 0, in same order as Part_Results
- Same data types: dates as ISO strings, booleans as bool, numbers as numbers.

Save to `/root/maintenance_resupply_actions_sep_2025.xlsx`.

## Step 3: Validate the output

After saving, reopen the file with openpyxl (data_only=True) and print:
- Sheet names
- Metadata cells A1:B4 from Part_Results
- Header row 6 from Part_Results
- All data rows from Part_Results (verify column count = 14, date formats, boolean types, numeric types)
- All rows from Additional_Resupply_Needed
- Confirm no extra sheets exist
- Confirm source files are unmodified

## Important notes
- Do NOT use formulas in the output; write computed values directly so cells are never None when read back.
- Ensure math.floor is used for DOH-to-days conversions, math.ceil for crate rounding.
- Handle edge cases: zero consumption rate, no scheduled deliveries for a part, no additional units needed.
- The Rounding_Applied and Earlier_Delivery_Required columns must be explicit boolean values (True/False), not strings.
- Preserve the exact source order of parts throughout both sheets.

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

Task-local resources are available under `environment/skills`: Inventory Turnover Analyzer, bc-calculated-fields-manufacturing.

# Verifier Contract

Success is judged by the SkillFlow verifier for this task.
Do not bypass, remove, or weaken verifier scripts, tests, fixtures, or expected-output checks.
Run the provided tests or verifier command when practical before finalizing.
Task metadata: author_email=codex@example.com, author_name=Codex, category=manufacturing-maintenance, difficulty=medium, tags=[excel, manufacturing, maintenance, calculated-fields, restock].
Verifier config: timeout_sec=900.0.