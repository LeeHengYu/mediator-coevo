# Task Instruction

## Task: Build Stochastic Refill Plan Workbook

Create `/root/stochastic_refill_plan_october_2025.xlsx` from `/root/Backup_Fuel_and_Refills_Latest.xlsx`.

### Step 1: Inspect the source workbook

Read the source workbook and print the structure of all three sheets:
- `Current Fuel` — print all rows (especially B1 and D1 for dates, and the data rows for site info)
- `Scheduled Refills` — print all rows (site IDs, inbound dates, inbound quantities)
- `Policy Parameters` — print all rows (Service_Level_Z, tanker capacity / conversion ratio, any other params)

Print column names and all data so you understand the exact field names, date formats, and layout before coding.

### Step 2: Write a Python script to produce the output workbook

Use `openpyxl` to read the source and write the output. Here is the detailed logic:

#### 2a. Read metadata
- `AsOfDate` = value in `Current Fuel` cell B1 (convert to `datetime.date`)
- `PlanningHorizonEnd` = value in `Current Fuel` cell D1 (convert to `datetime.date`)
- `RemainingDaysInOctober` = `(PlanningHorizonEnd - AsOfDate).days`

#### 2b. Read Policy Parameters
- Find `Service_Level_Z` (a z-score, likely ~1.65 or similar)
- Find the tanker capacity / conversion ratio (liters per tanker)
- Print all parameters so you know exactly what's available.

#### 2c. Read Current Fuel data
- Identify the header row and data rows. Each row represents one site/entity.
- Extract: `Site_ID`, `Current_Liters`, `Expected_Daily_Burn_Liters`, `Daily_Burn_StdDev`
- Preserve source order.

#### 2d. Read Scheduled Refills
- For each site, collect all scheduled refill records: site ID, inbound date, inbound quantity.
- For each site, compute:
  - `Inbound_Liters_By_Oct31` = sum of inbound liters where inbound_date <= PlanningHorizonEnd
  - `Earliest_Scheduled_Refill_Date` = min of all inbound dates for that site (regardless of whether <= PlanningHorizonEnd), or None if no scheduled refills

#### 2e. Compute per-site fields

For each site (in source order):

```
import math

rate = Expected_Daily_Burn_Liters
stddev = Daily_Burn_StdDev
current = Current_Liters
inbound = Inbound_Liters_By_Oct31
rem_days = RemainingDaysInOctober
z = Service_Level_Z
tanker_cap = <conversion ratio from Policy Parameters>

Current_DOH = current / rate if rate > 0 else None
Projected_Runout_Date = AsOfDate + timedelta(days=math.floor(Current_DOH)) if rate > 0 else None
Delivered_DOH_To_Oct31 = (current + inbound) / rate if rate > 0 else None
Remaining_October_Burn_Liters = rate * rem_days
Safety_Buffer_Liters = z * stddev * math.sqrt(rem_days)
Additional_Liters_Needed = max(0, Remaining_October_Burn_Liters + Safety_Buffer_Liters - current - inbound)

if Additional_Liters_Needed > 0:
    raw_tankers = Additional_Liters_Needed / tanker_cap
    Tankers_Required_Rounded_Up = math.ceil(raw_tankers)
    Rounding_Applied = (Tankers_Required_Rounded_Up != raw_tankers)  # TRUE if ceil changed value
else:
    Tankers_Required_Rounded_Up = 0
    Rounding_Applied = False

# Earliest_Scheduled_Refill_Date: earliest inbound date for this site, or None

# Required_Refill_Date:
if Tankers_Required_Rounded_Up == 0:
    Required_Refill_Date = None  # blank
elif Earliest_Scheduled_Refill_Date is not None and Earliest_Scheduled_Refill_Date <= Projected_Runout_Date:
    Required_Refill_Date = AsOfDate + timedelta(days=math.floor(Delivered_DOH_To_Oct31))
else:
    Required_Refill_Date = Projected_Runout_Date

# Earlier_Refill_Required:
if Tankers_Required_Rounded_Up > 0 and (Earliest_Scheduled_Refill_Date is None or Required_Refill_Date < Earliest_Scheduled_Refill_Date):
    Earlier_Refill_Required = True
else:
    Earlier_Refill_Required = False
```

**CRITICAL**: For `Rounding_Applied`, compare `math.ceil(raw_tankers)` vs `raw_tankers` — it's TRUE when `raw_tankers` is not already an integer (i.e., `raw_tankers != int(raw_tankers)` or equivalently `Additional_Liters_Needed % tanker_cap != 0`). When `Additional_Liters_Needed` is 0, set FALSE.

#### 2f. Write Sheet 1: `Site_Results`

- Metadata block:
  - A1="Field", B1="Value"
  - A2="AsOfDate", B2=AsOfDate as ISO string "YYYY-MM-DD"
  - A3="PlanningHorizonEnd", B3=PlanningHorizonEnd as ISO string
  - A4="RemainingDaysInOctober", B4=integer

- Header row at row 6 with exactly these 16 columns in order:
  Site_ID, Current_Liters, Expected_Daily_Burn_Liters, Daily_Burn_StdDev, Current_DOH, Projected_Runout_Date, Inbound_Liters_By_Oct31, Delivered_DOH_To_Oct31, Remaining_October_Burn_Liters, Safety_Buffer_Liters, Additional_Liters_Needed, Tankers_Required_Rounded_Up, Required_Refill_Date, Rounding_Applied, Earlier_Refill_Required, Earliest_Scheduled_Refill_Date

- Data rows starting at row 7, one per site in source order.

- **Date columns** (`Projected_Runout_Date`, `Required_Refill_Date`, `Earliest_Scheduled_Refill_Date`): write as ISO strings ("YYYY-MM-DD") using `.isoformat()` or `strftime('%Y-%m-%d')`. Write `None` (blank cell) when the value is blank.

- **Boolean columns** (`Rounding_Applied`, `Earlier_Refill_Required`): write Python `True`/`False` booleans so openpyxl stores them as Excel booleans.

- **Numeric columns**: write as numbers (int or float), not strings. `Current_DOH`, `Delivered_DOH_To_Oct31` should be floats. `Tankers_Required_Rounded_Up` should be int. `Additional_Liters_Needed`, `Safety_Buffer_Liters`, `Remaining_October_Burn_Liters` should be numeric.

- Blank/None values: leave cell empty (don't write anything).

#### 2g. Write Sheet 2: `Additional_Refills_Needed`

- Header row at row 1 with exactly 7 columns:
  Site_ID, Required_Refill_Date, Tankers_Required_Rounded_Up, Additional_Liters_Needed, Safety_Buffer_Liters, Rounding_Applied, Earlier_Refill_Required

- Include only sites where `Tankers_Required_Rounded_Up > 0`.
- Same order as Site_Results.
- Same data types: dates as ISO strings, booleans as booleans, numbers as numbers.

### Step 3: Run the script and verify

After creating the file:
1. Re-open it with openpyxl and print sheet names (must be exactly `['Site_Results', 'Additional_Refills_Needed']`).
2. Print the metadata block (rows 1-4 of Site_Results).
3. Print the header row (row 6 of Site_Results) and verify all 16 column names.
4. Print all data rows of Site_Results and verify:
   - Dates are ISO strings
   - Booleans are actual booleans (True/False)
   - Numbers are numeric
   - Blanks are None
5. Print all rows of Additional_Refills_Needed and verify it only contains sites with Tankers > 0.
6. Verify the file exists at `/root/stochastic_refill_plan_october_2025.xlsx`.

### Important Notes
- Do NOT modify the source file.
- Be careful with date parsing from the source — dates might be datetime objects or strings. Handle both.
- When comparing dates (for Earliest_Scheduled_Refill_Date <= Projected_Runout_Date), ensure both are date objects.
- `math.sqrt` and `math.ceil` and `math.floor` from the `math` module.
- The conversion ratio / tanker capacity is in Policy Parameters — read it carefully; it might be labeled differently (e.g., "Tanker_Capacity_Liters" or similar).

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