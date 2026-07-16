# Task Instruction

## Task: Build maintenance resupply planning workbook

### Goal
Create a single Excel workbook at `/root/maintenance_resupply_actions_sep_2025.xlsx` with exactly two sheets (`Part_Results`, `Additional_Resupply_Needed`) derived from the source workbook `/root/Maintenance_Parts_and_Deliveries_Latest.xlsx`.

### Step-by-step instructions

#### Step 0: Inspect the source workbook
1. Open `/root/Maintenance_Parts_and_Deliveries_Latest.xlsx` using `openpyxl`.
2. Print all sheet names.
3. For each sheet (`Current Parts`, `Scheduled Deliveries`, `Ratio`), print the first 15 rows so you can see headers, column positions, and data types.
4. Pay special attention to:
   - `Current Parts!B1` — this should be the AsOfDate.
   - `Current Parts!D1` — this should be the PlanningHorizonEnd.
   - Where entity/part codes, current units, daily consumption rates, and any other fields live.
   - `Scheduled Deliveries` — identify columns for part code, inbound quantity, and inbound/delivery date.
   - `Ratio` — identify the conversion ratio (units per crate) and how it maps to parts (single global ratio or per-part).
5. Print all data from `Ratio` sheet completely.
6. Print all data from `Scheduled Deliveries` completely.
7. Do NOT modify the source file.

#### Step 1: Extract key parameters
- `AsOfDate` = value from `Current Parts!B1` (parse as a date).
- `PlanningHorizonEnd` = value from `Current Parts!D1` (parse as a date).
- `RemainingDaysInSeptember` = `(PlanningHorizonEnd - AsOfDate).days` (integer, calendar day difference).

#### Step 2: Build Part_Results data
For each entity/row in `Current Parts` (preserving source row order), compute:

1. **Part_Code** — from source.
2. **Current_Units** — from source (numeric).
3. **Daily_Consumption_Units** — from source (numeric).
4. **Current_DOH** — `Current_Units / Daily_Consumption_Units` if rate > 0, else leave blank (`None`).
5. **Projected_Stockout_Date** — `AsOfDate + timedelta(days=floor(Current_DOH))` if rate > 0, else blank. Store as ISO string `YYYY-MM-DD`.
6. **Inbound_Units_By_Sep30** — Sum of inbound quantity from `Scheduled Deliveries` for this part where delivery date <= PlanningHorizonEnd. If no deliveries, use 0.
7. **Delivered_DOH_To_Sep30** — `(Current_Units + Inbound_Units_By_Sep30) / Daily_Consumption_Units` if rate > 0, else blank.
8. **Remaining_September_Demand_Units** — `Daily_Consumption_Units * RemainingDaysInSeptember`.
9. **Additional_Units_Needed** — `max(0, Remaining_September_Demand_Units - Current_Units - Inbound_Units_By_Sep30)`.
10. **Crates_Required_Rounded_Up** — If Additional_Units_Needed > 0: `math.ceil(Additional_Units_Needed / ratio)` where ratio is the conversion factor from the Ratio sheet. Else 0.
11. **Earliest_Scheduled_Delivery_Date** — The earliest delivery date from Scheduled Deliveries for this part (if any), else blank. ISO string.
12. **Required_Delivery_Date**:
    - Blank if `Crates_Required_Rounded_Up == 0`.
    - Else if `Earliest_Scheduled_Delivery_Date` is not blank AND `Earliest_Scheduled_Delivery_Date <= Projected_Stockout_Date`: use `AsOfDate + timedelta(days=floor(Delivered_DOH_To_Sep30))`. ISO string.
    - Else: use `Projected_Stockout_Date`. ISO string.
13. **Rounding_Applied** — `True` if Additional_Units_Needed > 0 AND `(Crates_Required_Rounded_Up * ratio) != Additional_Units_Needed`; else `False`. (Boolean)
14. **Earlier_Delivery_Required** — `True` if Crates_Required_Rounded_Up > 0 AND (Earliest_Scheduled_Delivery_Date is blank OR Required_Delivery_Date < Earliest_Scheduled_Delivery_Date); else `False`. (Boolean)

#### Step 3: Write Sheet 1 — Part_Results
Using `openpyxl`, create a new workbook:
- **Row 1**: A1="Field", B1="Value"
- **Row 2**: A2="AsOfDate", B2=AsOfDate as `YYYY-MM-DD` string
- **Row 3**: A3="PlanningHorizonEnd", B3=PlanningHorizonEnd as `YYYY-MM-DD` string
- **Row 4**: A4="RemainingDaysInSeptember", B4=integer value
- **Row 5**: leave empty
- **Row 6**: Header row with exactly these 14 column names in order:
  `Part_Code, Current_Units, Daily_Consumption_Units, Current_DOH, Projected_Stockout_Date, Inbound_Units_By_Sep30, Delivered_DOH_To_Sep30, Remaining_September_Demand_Units, Additional_Units_Needed, Crates_Required_Rounded_Up, Required_Delivery_Date, Rounding_Applied, Earlier_Delivery_Required, Earliest_Scheduled_Delivery_Date`
- **Row 7+**: One row per part, in source order.
- Numeric fields must be written as numbers (int or float), not strings.
- Boolean fields (`Rounding_Applied`, `Earlier_Delivery_Required`) must be written as Python `True`/`False` booleans.
- Date fields (`Projected_Stockout_Date`, `Required_Delivery_Date`, `Earliest_Scheduled_Delivery_Date`) must be written as ISO date strings (`YYYY-MM-DD`) or `None` for blank.
- Blank values should be `None` (which openpyxl writes as empty cells).

#### Step 4: Write Sheet 2 — Additional_Resupply_Needed
- Header at row 1: `Part_Code, Required_Delivery_Date, Crates_Required_Rounded_Up, Additional_Units_Needed, Rounding_Applied, Earlier_Delivery_Required`
- Include only rows where `Crates_Required_Rounded_Up > 0`.
- Preserve same order as Part_Results.
- Same data type rules (numeric as numbers, booleans as booleans, dates as ISO strings).

#### Step 5: Save and validate
1. Rename the default sheet or ensure only two sheets exist: `Part_Results` first, `Additional_Resupply_Needed` second. Remove any extra sheets (like the default "Sheet").
2. Save to `/root/maintenance_resupply_actions_sep_2025.xlsx`.
3. Re-open the file and verify:
   - Exactly 2 sheets with correct names in correct order.
   - Part_Results: A1="Field", B1="Value", A2="AsOfDate", B2 is a date string, A4="RemainingDaysInSeptember", B4 is an integer.
   - Row 6 headers match exactly.
   - Data rows start at row 7.
   - Spot-check a few cells: numeric values are numbers, booleans are booleans, date strings match YYYY-MM-DD pattern.
   - Additional_Resupply_Needed has correct headers and only rows with Crates > 0.
4. Print a summary of what was written.

### Critical reminders
- Do NOT modify the source file.
- The Ratio sheet may have a single ratio or per-part ratios — inspect it and handle accordingly.
- When comparing dates for `Earliest_Scheduled_Delivery_Date <= Projected_Stockout_Date`, make sure both are actual date objects (not strings) during comparison.
- `floor()` means `math.floor()` — use it for Current_DOH when computing stockout date and Delivered_DOH_To_Sep30 when computing Required_Delivery_Date.
- `ceil()` means `math.ceil()` for crate rounding.
- Double-check the column order in both sheets matches the specification exactly.

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