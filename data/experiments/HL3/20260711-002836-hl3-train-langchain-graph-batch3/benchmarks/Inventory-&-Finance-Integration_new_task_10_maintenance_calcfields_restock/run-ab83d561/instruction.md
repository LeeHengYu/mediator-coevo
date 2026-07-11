# Task Instruction

Execute the following steps in order to produce /root/maintenance_resupply_actions_sep_2025.xlsx.

## Step 0 – Investigate the source workbook

Open /root/Maintenance_Parts_and_Deliveries_Latest.xlsx with openpyxl and print:
1. For sheet "Current Parts": cells A1:F1 (header area), B1, D1 (dates), then all rows (print first 3 and last row, plus total row count). Identify column layout: which column is Part_Code, Current_Units, Daily_Consumption_Units, etc.
2. For sheet "Scheduled Deliveries": print header row and first 5 data rows. Identify columns for Part_Code, inbound quantity, inbound date.
3. For sheet "Ratio": print all content. Identify the conversion ratio (units per crate) – it may be a single value or per-part.

Print all of this before writing any output.

## Step 1 – Build the output workbook with a Python script

After inspecting the source, write and run a Python script that:

### 1a. Read source data
- Load all three sheets with openpyxl (data_only=False first; if dates are stored as strings, parse them; if as Excel dates, convert).
- Extract AsOfDate from Current Parts!B1 and PlanningHorizonEnd from Current Parts!D1. Convert both to Python date objects.
- Compute RemainingDaysInSeptember = (PlanningHorizonEnd - AsOfDate).days
- Read each part row from "Current Parts" preserving source order. Store Part_Code, Current_Units, Daily_Consumption_Units.
- Read all rows from "Scheduled Deliveries". For each row store Part_Code, inbound_date, inbound_quantity.
- Read conversion ratio from "Ratio" sheet. If it is a single ratio, use it for all parts. If per-part, build a lookup dict.

### 1b. Compute per-part fields
For each part (in source order):

```
rate = Daily_Consumption_Units
current = Current_Units

if rate > 0:
    Current_DOH = current / rate
    Projected_Stockout_Date = AsOfDate + timedelta(days=int(math.floor(Current_DOH)))
else:
    Current_DOH = None
    Projected_Stockout_Date = None

# Sum inbound units where inbound_date <= PlanningHorizonEnd
Inbound_Units_By_Sep30 = sum of matching deliveries

if rate > 0:
    Delivered_DOH_To_Sep30 = (current + Inbound_Units_By_Sep30) / rate
else:
    Delivered_DOH_To_Sep30 = None

Remaining_September_Demand_Units = rate * RemainingDaysInSeptember

Additional_Units_Needed = max(0, Remaining_September_Demand_Units - current - Inbound_Units_By_Sep30)

ratio = <conversion ratio for this part or global>
if Additional_Units_Needed > 0:
    raw_crates = Additional_Units_Needed / ratio
    Crates_Required_Rounded_Up = math.ceil(raw_crates)
    Rounding_Applied = (Crates_Required_Rounded_Up != raw_crates)  # TRUE if ceil changed value
else:
    Crates_Required_Rounded_Up = 0
    Rounding_Applied = False

# Earliest_Scheduled_Delivery_Date: earliest inbound date for this part (any delivery, not just <=Sep30), else None
all_dates_for_part = [d.inbound_date for d in deliveries if d.part == part_code]
Earliest_Scheduled_Delivery_Date = min(all_dates_for_part) if all_dates_for_part else None

# Required_Delivery_Date
if Crates_Required_Rounded_Up == 0:
    Required_Delivery_Date = None
else:
    if Earliest_Scheduled_Delivery_Date is not None and Earliest_Scheduled_Delivery_Date <= Projected_Stockout_Date:
        Required_Delivery_Date = AsOfDate + timedelta(days=int(math.floor(Delivered_DOH_To_Sep30)))
    else:
        Required_Delivery_Date = Projected_Stockout_Date

# Earlier_Delivery_Required
if Crates_Required_Rounded_Up > 0 and (Earliest_Scheduled_Delivery_Date is None or Required_Delivery_Date < Earliest_Scheduled_Delivery_Date):
    Earlier_Delivery_Required = True
else:
    Earlier_Delivery_Required = False
```

### 1c. Write Sheet 1: Part_Results
- A1="Field", B1="Value"
- A2="AsOfDate", B2=AsOfDate as ISO string "YYYY-MM-DD"
- A3="PlanningHorizonEnd", B3=PlanningHorizonEnd as ISO string
- A4="RemainingDaysInSeptember", B4=integer
- Row 6: header row with exactly these 14 columns:
  Part_Code, Current_Units, Daily_Consumption_Units, Current_DOH, Projected_Stockout_Date, Inbound_Units_By_Sep30, Delivered_DOH_To_Sep30, Remaining_September_Demand_Units, Additional_Units_Needed, Crates_Required_Rounded_Up, Required_Delivery_Date, Rounding_Applied, Earlier_Delivery_Required, Earliest_Scheduled_Delivery_Date
- Data rows starting at row 7, one per part in source order.
- Date columns (Projected_Stockout_Date, Required_Delivery_Date, Earliest_Scheduled_Delivery_Date): write as ISO strings "YYYY-MM-DD" or leave cell empty/None for blank.
- Boolean columns (Rounding_Applied, Earlier_Delivery_Required): write Python True/False (openpyxl will store as Excel boolean).
- Numeric columns: write as numbers (int or float), not strings.
- Blank values: leave cell unset (None).

### 1d. Write Sheet 2: Additional_Resupply_Needed
- Header at row 1: Part_Code, Required_Delivery_Date, Crates_Required_Rounded_Up, Additional_Units_Needed, Rounding_Applied, Earlier_Delivery_Required
- Include only parts where Crates_Required_Rounded_Up > 0, same order as Part_Results.
- Same data types as Sheet 1.

### 1e. Save
- Save workbook to /root/maintenance_resupply_actions_sep_2025.xlsx
- Do NOT modify the source file.

## Step 2 – Validate the output

After saving, re-open the output file and print:
1. Sheet names (must be exactly ["Part_Results", "Additional_Resupply_Needed"]).
2. Part_Results metadata cells A1:B4.
3. Part_Results header row (row 6) – verify 14 columns match expected names.
4. First 3 data rows and last data row from Part_Results with all 14 fields.
5. Row count of data rows in Part_Results (should equal number of parts in source).
6. Additional_Resupply_Needed header and all rows – verify only rows with Crates > 0 appear.
7. Verify date fields are strings in YYYY-MM-DD format.
8. Verify boolean fields are actual booleans.
9. Verify numeric fields are numbers.

If any check fails, fix and re-save before finishing.

## Critical notes
- Investigate the source FIRST. Do not assume column positions or date formats.
- Use math.floor for DOH-to-days conversions, math.ceil for crate rounding.
- The Rounding_Applied boolean: TRUE when Additional_Units_Needed > 0 AND ceil changed the value (i.e., Additional_Units_Needed / ratio is not already an integer). FALSE otherwise.
- Dates must be ISO strings in output cells, not Excel date serial numbers.
- Preserve source part order everywhere.
- Do not create any extra sheets.

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