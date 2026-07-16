# Task Instruction

## Task: Build Hospital Staffing Resilience Workbook

Create the file `/root/additional_shift_blocks_needed_august_2025.xlsx` from the source `/root/Staffing_and_Shifts_Latest.xlsx`.

### Step 1: Inspect the source workbook

Open `/root/Staffing_and_Shifts_Latest.xlsx` with openpyxl and inspect all three sheets:

1. **Current Staffing** – Print all cells (especially row 1 for dates in B1 and D1, and the data rows below). Identify:
   - AsOfDate from cell B1
   - PlanningHorizonEnd from cell D1
   - The entity/unit names and their columns (look for staff hours and daily required hours columns)
   - Note the exact column layout and row range

2. **Incoming Shifts** – Print all rows. Identify:
   - Which column has the entity/unit name
   - Which column has the inbound date
   - Which column has the inbound quantity (hours)

3. **Ratio** – Print all cells. Identify:
   - The conversion ratio value (hours per shift block)

Print everything before writing any code. Do NOT assume column positions.

### Step 2: Inspect the test file

Read `/root/test_output.py` completely to understand what the verifier checks. Note every assertion, expected value, cell reference, column name, data type expectation, and sheet name. This is critical — the verifier contract defines correctness.

### Step 3: Write the generation script

Create a Python script that:

1. Reads the source workbook (without modifying it).
2. Computes all values per the calculation rules below.
3. Writes the output workbook with exactly two sheets: `Unit_Results` and `Additional_Shifts_Needed` (in that order, no other sheets).

#### Metadata (Unit_Results sheet):
- A1="Field", B1="Value"
- A2="AsOfDate", B2=AsOfDate as ISO string "YYYY-MM-DD"
- A3="PlanningHorizonEnd", B3=PlanningHorizonEnd as ISO string "YYYY-MM-DD"
- A4="RemainingDaysInAugust", B4=integer (PlanningHorizonEnd - AsOfDate).days

#### Header row at row 6 (14 columns):
Care_Unit, Current_Staff_Hours, Daily_Required_Hours, Current_Coverage_Days, Projected_Understaff_Date, Incoming_Hours_By_Aug31, Delivered_Coverage_To_Aug31, Remaining_August_Demand_Hours, Additional_Hours_Needed, Shift_Blocks_Required_Rounded_Up, Required_Shift_Start_Date, Rounding_Applied, Earlier_Shift_Required, Earliest_Scheduled_Shift_Date

#### Data rows starting at row 7, one per entity from Current Staffing, preserving source order.

#### Calculation rules (apply carefully):
- `RemainingDaysInAugust` = (PlanningHorizonEnd - AsOfDate).days
- `Current_Coverage_Days` = Current_Staff_Hours / Daily_Required_Hours when Daily_Required_Hours > 0, else None/blank
- `Projected_Understaff_Date` = AsOfDate + timedelta(days=floor(Current_Coverage_Days)) when rate > 0, else None. Store as ISO string.
- `Incoming_Hours_By_Aug31` = sum of inbound hours for that entity where inbound date <= PlanningHorizonEnd
- `Delivered_Coverage_To_Aug31` = (Current_Staff_Hours + Incoming_Hours_By_Aug31) / Daily_Required_Hours when rate > 0, else None
- `Remaining_August_Demand_Hours` = Daily_Required_Hours * RemainingDaysInAugust
- `Additional_Hours_Needed` = max(0, Remaining_August_Demand_Hours - Current_Staff_Hours - Incoming_Hours_By_Aug31)
- `Shift_Blocks_Required_Rounded_Up` = math.ceil(Additional_Hours_Needed / ratio) when Additional_Hours_Needed > 0, else 0
- `Earliest_Scheduled_Shift_Date` = earliest inbound date for that entity (across all incoming shifts for that entity), else None. Store as ISO string.
- `Required_Shift_Start_Date`:
  - None when Shift_Blocks_Required_Rounded_Up == 0
  - else if Earliest_Scheduled_Shift_Date is not None and Earliest_Scheduled_Shift_Date <= Projected_Understaff_Date: AsOfDate + timedelta(days=floor(Delivered_Coverage_To_Aug31)), as ISO string
  - else: Projected_Understaff_Date (already ISO string)
- `Rounding_Applied` = Python boolean True when Additional_Hours_Needed > 0 and ceil(Additional_Hours_Needed/ratio) != Additional_Hours_Needed/ratio; else False
- `Earlier_Shift_Required` = Python boolean True when Shift_Blocks_Required_Rounded_Up > 0 and (Earliest_Scheduled_Shift_Date is None OR Required_Shift_Start_Date < Earliest_Scheduled_Shift_Date); else False

**Important type details:**
- Date columns (Projected_Understaff_Date, Required_Shift_Start_Date, Earliest_Scheduled_Shift_Date) must be ISO format strings ("YYYY-MM-DD"), not datetime objects.
- Rounding_Applied and Earlier_Shift_Required must be Python booleans (True/False), not strings.
- Numeric fields must be numbers (int or float), not strings.
- Blank means None (do not write empty string).

#### Sheet 2: Additional_Shifts_Needed
Header at row 1: Care_Unit, Required_Shift_Start_Date, Shift_Blocks_Required_Rounded_Up, Additional_Hours_Needed, Rounding_Applied, Earlier_Shift_Required

Include only rows where Shift_Blocks_Required_Rounded_Up > 0, same order as Unit_Results.

### Step 4: Run the script and verify

1. Run the generation script.
2. Verify the output file exists at `/root/additional_shift_blocks_needed_august_2025.xlsx`.
3. Quick-check: open the output with openpyxl and print both sheets' contents to confirm structure and values.
4. Run `cd /root && python test_output.py` (or `pytest test_output.py -v`) to check verifier results.
5. If any test fails, read the error carefully, re-inspect source data and calculations, fix, and re-run until all tests pass.

### Important Notes
- Do NOT modify the source workbook.
- When reading dates from the source, handle both datetime objects and strings gracefully.
- When comparing dates for Incoming_Hours filtering, ensure consistent date types.
- The output workbook must have exactly 2 sheets in the specified order. Remove any default sheets.
- Pay close attention to the Rounding_Applied logic: it's about whether ceiling changed the value, i.e., Additional_Hours_Needed/ratio is not an integer.

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

Task-local resources are available under `environment/skills`: inventory-manager.

# Verifier Contract

Success is judged by the SkillFlow verifier for this task.
Do not bypass, remove, or weaken verifier scripts, tests, fixtures, or expected-output checks.
Run the provided tests or verifier command when practical before finalizing.
Task metadata: author_email=codex@example.com, author_name=Codex, category=workforce-planning, difficulty=medium, tags=[excel, staffing, capacity, replenishment, operations].
Verifier config: timeout_sec=900.0.