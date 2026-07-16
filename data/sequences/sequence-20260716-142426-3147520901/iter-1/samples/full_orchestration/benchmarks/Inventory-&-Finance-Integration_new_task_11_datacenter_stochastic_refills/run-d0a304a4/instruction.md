# Task Instruction

Execute the following steps to produce /root/stochastic_refill_plan_october_2025.xlsx.

## Step 1 – Inspect the source workbook

```python
import openpyxl, pprint
wb = openpyxl.load_workbook('/root/Backup_Fuel_and_Refills_Latest.xlsx', data_only=True)
for s in wb.sheetnames:
    ws = wb[s]
    print(f'\n=== {s} (rows={ws.max_row}, cols={ws.max_column}) ===')
    for r in ws.iter_rows(min_row=1, max_row=min(ws.max_row, 25), values_only=False):
        print([(c.coordinate, c.value) for c in r])
```

Run this first and read every printed line carefully. Identify:
- The exact cell positions of AsOfDate and PlanningHorizonEnd in "Current Fuel" (expected B1 and D1).
- Column headers and data rows in "Current Fuel" (Site_ID, Current_Liters, Expected_Daily_Burn_Liters, Daily_Burn_StdDev, etc.).
- Column headers and data rows in "Scheduled Refills" (Site_ID, inbound date, inbound quantity, etc.).
- Column headers and data in "Policy Parameters" (Service_Level_Z, tanker/container capacity conversion ratio, etc.).

## Step 2 – Build the output workbook

Write a single Python script that:

### 2a – Read source data
- Parse "Current Fuel" to get AsOfDate (B1) and PlanningHorizonEnd (D1). Convert them to `datetime.date` objects if they are datetime.
- Read the data rows from "Current Fuel" into a list of dicts preserving source order. Map column headers to the fields: Site_ID, Current_Liters, Expected_Daily_Burn_Liters, Daily_Burn_StdDev. Adapt header mapping to whatever the actual column names are.
- Parse "Scheduled Refills" into a list of dicts with at least: Site_ID, inbound_date, inbound_quantity. Again adapt to actual column names.
- Parse "Policy Parameters" to extract Service_Level_Z (float) and the tanker/container capacity (float). Print both values to confirm.

### 2b – Compute RemainingDaysInOctober
```
RemainingDaysInOctober = (PlanningHorizonEnd - AsOfDate).days
```

### 2c – Per-site calculations (preserve source order)
For each site from Current Fuel:

1. `Current_Liters` – numeric from source.
2. `Expected_Daily_Burn_Liters` – numeric from source.
3. `Daily_Burn_StdDev` – numeric from source.
4. `Current_DOH` = Current_Liters / Expected_Daily_Burn_Liters if rate > 0 else None.
5. `Projected_Runout_Date` = AsOfDate + timedelta(days=floor(Current_DOH)) if rate > 0 else None. Store as ISO string YYYY-MM-DD.
6. `Inbound_Liters_By_Oct31` = sum of inbound quantity for this site where inbound_date <= PlanningHorizonEnd.
7. `Delivered_DOH_To_Oct31` = (Current_Liters + Inbound_Liters_By_Oct31) / Expected_Daily_Burn_Liters if rate > 0 else None.
8. `Remaining_October_Burn_Liters` = Expected_Daily_Burn_Liters * RemainingDaysInOctober.
9. `Safety_Buffer_Liters` = Service_Level_Z * Daily_Burn_StdDev * sqrt(RemainingDaysInOctober).
10. `Additional_Liters_Needed` = max(0, Remaining_October_Burn_Liters + Safety_Buffer_Liters - Current_Liters - Inbound_Liters_By_Oct31).
11. `Tankers_Required_Rounded_Up` = ceil(Additional_Liters_Needed / tanker_capacity) if Additional_Liters_Needed > 0 else 0.
12. `Earliest_Scheduled_Refill_Date` = earliest inbound date for this site (any date, not just ≤ Oct31), else None. Store as ISO string.
13. `Rounding_Applied`:
    - TRUE if Additional_Liters_Needed > 0 AND (Additional_Liters_Needed / tanker_capacity) != ceil(Additional_Liters_Needed / tanker_capacity) (i.e., ceiling changed the value).
    - FALSE otherwise.
14. `Required_Refill_Date`:
    - None (blank) if Tankers_Required_Rounded_Up == 0.
    - Else if Earliest_Scheduled_Refill_Date is not None and Earliest_Scheduled_Refill_Date <= Projected_Runout_Date: AsOfDate + timedelta(days=floor(Delivered_DOH_To_Oct31)). Store as ISO string.
    - Else: Projected_Runout_Date (already ISO string).
15. `Earlier_Refill_Required`:
    - TRUE if Tankers_Required_Rounded_Up > 0 AND (Earliest_Scheduled_Refill_Date is None OR Required_Refill_Date < Earliest_Scheduled_Refill_Date).
    - FALSE otherwise.

**Important**: For boolean fields (Rounding_Applied, Earlier_Refill_Required), write Python `True`/`False` (which openpyxl stores as Excel booleans). For date columns (Projected_Runout_Date, Required_Refill_Date, Earliest_Scheduled_Refill_Date), write ISO format strings. Keep all numeric fields as int or float, not strings.

### 2d – Write Sheet 1: Site_Results
Create a new workbook. First sheet named "Site_Results".

Metadata:
- A1="Field", B1="Value"
- A2="AsOfDate", B2=AsOfDate as YYYY-MM-DD string
- A3="PlanningHorizonEnd", B3=PlanningHorizonEnd as YYYY-MM-DD string
- A4="RemainingDaysInOctober", B4=integer

Row 6: header row with exactly these 16 column names in order:
Site_ID, Current_Liters, Expected_Daily_Burn_Liters, Daily_Burn_StdDev, Current_DOH, Projected_Runout_Date, Inbound_Liters_By_Oct31, Delivered_DOH_To_Oct31, Remaining_October_Burn_Liters, Safety_Buffer_Liters, Additional_Liters_Needed, Tankers_Required_Rounded_Up, Required_Refill_Date, Rounding_Applied, Earlier_Refill_Required, Earliest_Scheduled_Refill_Date

Data rows start at row 7, one per site in source order.

### 2e – Write Sheet 2: Additional_Refills_Needed
Second sheet named "Additional_Refills_Needed".

Header row at row 1 with exactly these 7 columns:
Site_ID, Required_Refill_Date, Tankers_Required_Rounded_Up, Additional_Liters_Needed, Safety_Buffer_Liters, Rounding_Applied, Earlier_Refill_Required

Include only sites where Tankers_Required_Rounded_Up > 0, same order as Site_Results. Each qualifying site appears exactly once.

### 2f – Save
Save to `/root/stochastic_refill_plan_october_2025.xlsx`. Do NOT modify the source file.

## Step 3 – Validate

After saving, re-open the output workbook and print:
1. All sheet names.
2. Rows 1-4 of Site_Results (metadata).
3. Row 6 headers of Site_Results.
4. All data rows of Site_Results (row 7+), printing each cell value and its Python type.
5. All rows of Additional_Refills_Needed, printing each cell value and its Python type.

Confirm:
- Exactly 2 sheets in correct order.
- Boolean cells are actual booleans (True/False), not strings.
- Date cells are strings in YYYY-MM-DD format.
- Numeric cells are int or float.
- Additional_Refills_Needed contains only rows with Tankers_Required_Rounded_Up > 0.
- Site order matches source.

If any issue is found, fix and re-save before finishing.

## Step 4 – Check for verifier
Look for any test file (e.g., /root/test_output.py or similar) and if found, run it with `pytest -xvs` to confirm the output passes. Print the result.

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

Task-local resources are available under `environment/skills`: inventory-manager, stochastic-inventory-models.

# Verifier Contract

Success is judged by the SkillFlow verifier for this task.
Do not bypass, remove, or weaken verifier scripts, tests, fixtures, or expected-output checks.
Run the provided tests or verifier command when practical before finalizing.
Task metadata: author_email=codex@example.com, author_name=Codex, category=infrastructure-planning, difficulty=medium, tags=[excel, datacenter, stochastic, capacity, fuel].
Verifier config: timeout_sec=900.0.