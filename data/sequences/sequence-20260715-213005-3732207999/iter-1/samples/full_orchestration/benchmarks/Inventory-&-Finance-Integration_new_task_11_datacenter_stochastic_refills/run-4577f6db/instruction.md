# Task Instruction

## Task: Build Stochastic Refill Plan Workbook

Create `/root/stochastic_refill_plan_october_2025.xlsx` from `/root/Backup_Fuel_and_Refills_Latest.xlsx`.

### Step 1: Inspect the source workbook

Read the source workbook and print the structure of all three sheets:
- `Current Fuel` — print all rows (especially B1, D1 to get AsOfDate and PlanningHorizonEnd, and identify the header row and data rows; note the column layout for Site_ID, Current_Liters, Expected_Daily_Burn_Liters, Daily_Burn_StdDev)
- `Scheduled Refills` — print all rows (identify columns for entity/site ID, inbound date, inbound quantity)
- `Policy Parameters` — print all rows (identify Service_Level_Z and the tanker/container capacity conversion ratio)

Print everything verbatim so you understand exact column names, positions, date formats, and data types before writing any code.

### Step 2: Write a Python script to produce the output workbook

Use `openpyxl` (and `math`, `datetime` as needed). Do NOT use pandas for writing (to keep full control of cell placement). You may use pandas or openpyxl for reading.

The script must:

#### 2a. Read source data
- Parse AsOfDate from `Current Fuel` cell B1. Parse PlanningHorizonEnd from `Current Fuel` cell D1. Both should become `datetime.date` objects. If they are already datetime objects, extract `.date()`.
- Compute `RemainingDaysInOctober = (PlanningHorizonEnd - AsOfDate).days`.
- Read the data rows from `Current Fuel` preserving source order. Identify each site's: Site_ID, Current_Liters, Expected_Daily_Burn_Liters, Daily_Burn_StdDev. Map column names carefully based on what you observed in Step 1.
- Read `Scheduled Refills` into a list of (Site_ID, inbound_date, inbound_quantity) records. Parse dates properly.
- Read `Policy Parameters` to get Service_Level_Z (a float, e.g., 1.65 or similar) and the tanker capacity (liters per tanker/container).

#### 2b. Compute per-site values

For each site (in source order):

1. `Current_Liters` — from source
2. `Expected_Daily_Burn_Liters` — from source
3. `Daily_Burn_StdDev` — from source
4. `Current_DOH` = Current_Liters / Expected_Daily_Burn_Liters if rate > 0, else None
5. `Projected_Runout_Date` = AsOfDate + timedelta(days=floor(Current_DOH)) if rate > 0, else None. Store as ISO string (YYYY-MM-DD).
6. `Inbound_Liters_By_Oct31` = sum of inbound quantity for this site where inbound_date <= PlanningHorizonEnd
7. `Delivered_DOH_To_Oct31` = (Current_Liters + Inbound_Liters_By_Oct31) / Expected_Daily_Burn_Liters if rate > 0, else None
8. `Remaining_October_Burn_Liters` = Expected_Daily_Burn_Liters * RemainingDaysInOctober
9. `Safety_Buffer_Liters` = Service_Level_Z * Daily_Burn_StdDev * sqrt(RemainingDaysInOctober)
10. `Additional_Liters_Needed` = max(0, Remaining_October_Burn_Liters + Safety_Buffer_Liters - Current_Liters - Inbound_Liters_By_Oct31)
11. `Tankers_Required_Rounded_Up` = ceil(Additional_Liters_Needed / tanker_capacity) if Additional_Liters_Needed > 0, else 0
12. `Earliest_Scheduled_Refill_Date` = earliest inbound date for this site (across all scheduled refills, not just those <= Oct31), else None. Store as ISO string.
13. `Required_Refill_Date`:
    - None (blank) if Tankers_Required_Rounded_Up == 0
    - else if Earliest_Scheduled_Refill_Date is not None and earliest_date_obj <= projected_runout_date_obj: use AsOfDate + timedelta(days=floor(Delivered_DOH_To_Oct31))
    - else: use Projected_Runout_Date
    - Store as ISO string.
14. `Rounding_Applied`:
    - If Additional_Liters_Needed > 0: TRUE if ceil(Additional_Liters_Needed / tanker_capacity) != (Additional_Liters_Needed / tanker_capacity) — i.e., there was fractional rounding. Else FALSE.
    - If Additional_Liters_Needed <= 0 (or == 0): FALSE
15. `Earlier_Refill_Required`:
    - TRUE if Tankers_Required_Rounded_Up > 0 AND (Earliest_Scheduled_Refill_Date is None OR Required_Refill_Date < Earliest_Scheduled_Refill_Date)
    - else FALSE

**Important**: `Rounding_Applied` and `Earlier_Refill_Required` must be Python booleans (`True`/`False`) so openpyxl writes them as Excel boolean values (TRUE/FALSE), not strings.

#### 2c. Write Sheet 1: Site_Results

- A1="Field", B1="Value"
- A2="AsOfDate", B2=AsOfDate as ISO string (YYYY-MM-DD)
- A3="PlanningHorizonEnd", B3=PlanningHorizonEnd as ISO string
- A4="RemainingDaysInOctober", B4=integer
- Row 6 is the header row with exactly these 16 columns in order:
  Site_ID, Current_Liters, Expected_Daily_Burn_Liters, Daily_Burn_StdDev, Current_DOH, Projected_Runout_Date, Inbound_Liters_By_Oct31, Delivered_DOH_To_Oct31, Remaining_October_Burn_Liters, Safety_Buffer_Liters, Additional_Liters_Needed, Tankers_Required_Rounded_Up, Required_Refill_Date, Rounding_Applied, Earlier_Refill_Required, Earliest_Scheduled_Refill_Date
- Data rows start at row 7, one per site in source order.
- Numeric fields must be numbers (int or float), not strings.
- Date fields (Projected_Runout_Date, Required_Refill_Date, Earliest_Scheduled_Refill_Date) must be ISO strings or None (blank).
- Boolean fields (Rounding_Applied, Earlier_Refill_Required) must be Python True/False booleans.

#### 2d. Write Sheet 2: Additional_Refills_Needed

- Header at row 1 with exactly 7 columns: Site_ID, Required_Refill_Date, Tankers_Required_Rounded_Up, Additional_Liters_Needed, Safety_Buffer_Liters, Rounding_Applied, Earlier_Refill_Required
- Include only sites where Tankers_Required_Rounded_Up > 0, in the same order as Site_Results.
- Same data type rules: numbers numeric, dates as ISO strings, booleans as True/False.

#### 2e. Save

Save to `/root/stochastic_refill_plan_october_2025.xlsx`. Do NOT modify the source file.

### Step 3: Run the script

Execute the script. If there are errors, debug by re-reading the source data structure and fixing.

### Step 4: Validate the output

After the script runs successfully:
1. Re-open the output workbook and print both sheets completely.
2. Verify:
   - Sheet names are exactly `Site_Results` and `Additional_Refills_Needed` in that order.
   - Site_Results has metadata in A1:B4, header in row 6, data starting row 7.
   - All 16 columns present in correct order in Site_Results.
   - All 7 columns present in correct order in Additional_Refills_Needed.
   - Additional_Refills_Needed contains only rows with Tankers_Required_Rounded_Up > 0.
   - Date cells contain ISO format strings.
   - Boolean cells contain actual booleans (print type to confirm).
   - Numeric cells are numeric.
   - Spot-check one or two calculations manually.
3. If anything is wrong, fix and re-run.

### Key cautions
- Read the source file carefully before coding. Column names may differ from what you expect.
- The conversion ratio / tanker capacity is in Policy Parameters — find the exact cell.
- Service_Level_Z is in Policy Parameters — find the exact cell.
- Dates in the source may be datetime objects or strings — handle both.
- When comparing dates for Earliest_Scheduled_Refill_Date <= Projected_Runout_Date, convert both to date objects.
- `sqrt` is `math.sqrt`, `ceil` is `math.ceil`, `floor` is `math.floor`.
- Use `openpyxl.Workbook()` for output; do not copy from the source workbook.

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