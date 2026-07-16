# Task Instruction

## Task: Build Stochastic Refill Plan Workbook

Create `/root/stochastic_refill_plan_october_2025.xlsx` from `/root/Backup_Fuel_and_Refills_Latest.xlsx`.

### Step 0: Inspect the source workbook

Read the source workbook thoroughly before any computation:

```python
import openpyxl
wb = openpyxl.load_workbook('/root/Backup_Fuel_and_Refills_Latest.xlsx', data_only=True)
for name in wb.sheetnames:
    ws = wb[name]
    print(f"\n=== Sheet: {name} ===")
    for row in ws.iter_rows(min_row=1, max_row=min(ws.max_row, 30), values_only=False):
        print([(c.coordinate, c.value) for c in row])
```

Record:
- The exact date in Current Fuel!B1 (AsOfDate) and Current Fuel!D1 (PlanningHorizonEnd). Note their types (datetime vs string).
- The column layout of Current Fuel (Site_ID column, Current_Liters, Expected_Daily_Burn_Liters, Daily_Burn_StdDev, etc.).
- The column layout of Scheduled Refills (entity/site identifier, inbound date, inbound quantity).
- The column layout of Policy Parameters (find Service_Level_Z and the tanker/container conversion ratio in liters).
- Confirm the entity identifier column name used across sheets so joins are correct.
- Count entities in Current Fuel to know expected row count.

### Step 1: Compute all values in Python

Using openpyxl (or pandas for computation, openpyxl for writing), compute everything in Python with explicit numeric values (not Excel formulas). This ensures correctness.

#### Key formulas:

```
RemainingDaysInOctober = (PlanningHorizonEnd - AsOfDate).days
```
(This is a calendar day difference, an integer.)

For each site (preserving source order from Current Fuel):
```
Current_DOH = Current_Liters / Expected_Daily_Burn_Liters  (if rate > 0, else None)
Projected_Runout_Date = AsOfDate + timedelta(days=floor(Current_DOH))  (if rate > 0, else None)
Inbound_Liters_By_Oct31 = sum of inbound liters for this site where inbound_date <= PlanningHorizonEnd
Delivered_DOH_To_Oct31 = (Current_Liters + Inbound_Liters_By_Oct31) / Expected_Daily_Burn_Liters  (if rate > 0, else None)
Remaining_October_Burn_Liters = Expected_Daily_Burn_Liters * RemainingDaysInOctober
Safety_Buffer_Liters = Service_Level_Z * Daily_Burn_StdDev * sqrt(RemainingDaysInOctober)
Additional_Liters_Needed = max(0, Remaining_October_Burn_Liters + Safety_Buffer_Liters - Current_Liters - Inbound_Liters_By_Oct31)
Tankers_Required_Rounded_Up = ceil(Additional_Liters_Needed / tanker_capacity) if Additional_Liters_Needed > 0 else 0
Earliest_Scheduled_Refill_Date = min of scheduled inbound dates for this site (or None if no scheduled refills)
```

For Required_Refill_Date:
- If Tankers_Required_Rounded_Up == 0: None (blank)
- Else if Earliest_Scheduled_Refill_Date is not None AND Earliest_Scheduled_Refill_Date <= Projected_Runout_Date:
  use AsOfDate + timedelta(days=floor(Delivered_DOH_To_Oct31))
- Else: use Projected_Runout_Date

For Rounding_Applied:
- TRUE if Additional_Liters_Needed > 0 AND ceil(Additional/tanker_cap) != (Additional/tanker_cap)
  (i.e., there was actual rounding — the fractional part is nonzero)
- FALSE otherwise

For Earlier_Refill_Required:
- TRUE if Tankers_Required_Rounded_Up > 0 AND (Earliest_Scheduled_Refill_Date is None OR Required_Refill_Date < Earliest_Scheduled_Refill_Date)
- FALSE otherwise

### Step 2: Write the output workbook

Create a new workbook with exactly two sheets in order: `Site_Results`, `Additional_Refills_Needed`. Remove any default sheets.

#### Sheet 1: Site_Results

Row 1: A1="Field", B1="Value"
Row 2: A2="AsOfDate", B2=AsOfDate as YYYY-MM-DD string
Row 3: A3="PlanningHorizonEnd", B3=PlanningHorizonEnd as YYYY-MM-DD string
Row 4: A4="RemainingDaysInOctober", B4=integer

Row 6: Header row with exactly these 16 columns:
Site_ID, Current_Liters, Expected_Daily_Burn_Liters, Daily_Burn_StdDev, Current_DOH, Projected_Runout_Date, Inbound_Liters_By_Oct31, Delivered_DOH_To_Oct31, Remaining_October_Burn_Liters, Safety_Buffer_Liters, Additional_Liters_Needed, Tankers_Required_Rounded_Up, Required_Refill_Date, Rounding_Applied, Earlier_Refill_Required, Earliest_Scheduled_Refill_Date

Data rows start at row 7, one per entity, preserving source order.

**Critical formatting rules:**
- All date columns (Projected_Runout_Date, Required_Refill_Date, Earliest_Scheduled_Refill_Date) must be written as ISO date strings ("YYYY-MM-DD"), not datetime objects. Use `date_obj.strftime('%Y-%m-%d')` or `date_obj.isoformat()`. Write None/blank for missing.
- Rounding_Applied and Earlier_Refill_Required must be Python booleans (True/False), not strings. openpyxl will write them as Excel TRUE/FALSE.
- Numeric fields must be numeric (int or float), not strings.
- Blank means the cell is left as None (not written or written as None).

#### Sheet 2: Additional_Refills_Needed

Row 1: Header with exactly 7 columns:
Site_ID, Required_Refill_Date, Tankers_Required_Rounded_Up, Additional_Liters_Needed, Safety_Buffer_Liters, Rounding_Applied, Earlier_Refill_Required

Include only rows where Tankers_Required_Rounded_Up > 0, same order as Site_Results. Same formatting rules (dates as ISO strings, booleans as booleans, numerics as numbers).

### Step 3: Validate

After writing, re-read the output workbook and verify:
1. Sheet names are exactly ["Site_Results", "Additional_Refills_Needed"]
2. Site_Results metadata cells A1:B4 are correct
3. Site_Results header is at row 6 with all 16 columns in exact order
4. Data rows start at row 7, count matches source entity count
5. All date cells contain strings in YYYY-MM-DD format (not datetime objects)
6. Boolean cells contain actual booleans
7. Numeric cells are numbers
8. Additional_Refills_Needed contains only rows with Tankers > 0
9. Entity order matches source

Print a summary of the validation results.

### Important Cautions
- Do NOT modify the source file.
- Carefully match entity identifiers between Current Fuel and Scheduled Refills sheets (they might use slightly different column names — inspect first).
- The tanker capacity / conversion ratio is in Policy Parameters — find the exact cell and value.
- Service_Level_Z is in Policy Parameters — find the exact cell and value.
- Use `math.ceil` for ceiling, `math.floor` for floor, `math.sqrt` for square root.
- If a site has zero burn rate, many fields become blank — handle this edge case.
- Double-check that `Rounding_Applied` logic: it's TRUE when additional > 0 AND the ceiling operation actually changed the value (i.e., Additional_Liters_Needed is not an exact multiple of tanker capacity). If Additional_Liters_Needed == 0, it's FALSE even if formally ceil(0/cap) == 0.

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