# Task Instruction

## Task: Build Hospital Staffing Resilience Workbook

Create `/root/additional_shift_blocks_needed_august_2025.xlsx` from source `/root/Staffing_and_Shifts_Latest.xlsx`.

### Step 1: Inspect the Source Workbook

Read the source workbook and print the contents of all three sheets (`Current Staffing`, `Incoming Shifts`, `Ratio`) so you understand the exact structure, column names, date formats, and data values. Print all rows — do not truncate. Pay special attention to:
- `Current Staffing!B1` (AsOfDate) and `Current Staffing!D1` (PlanningHorizonEnd) — identify exactly which cells contain dates.
- The column layout: which columns hold entity/unit names, staff hours, daily required hours, etc.
- `Incoming Shifts`: which columns hold entity names, inbound dates, and inbound quantities (hours).
- `Ratio`: which cell(s) hold the conversion ratio (hours per shift block).

Print cell addresses and values explicitly so nothing is assumed.

### Step 2: Build the Output Workbook with openpyxl

Use Python with `openpyxl` to create the output workbook. Do NOT use pandas to write (pandas can silently change types); use openpyxl directly for full control over cell types and formatting.

#### Sheet 1: `Unit_Results`

**Metadata (rows 1-4):**
- A1="Field", B1="Value"
- A2="AsOfDate", B2=AsOfDate as ISO string "YYYY-MM-DD"
- A3="PlanningHorizonEnd", B3=PlanningHorizonEnd as ISO string "YYYY-MM-DD"
- A4="RemainingDaysInAugust", B4=integer: (PlanningHorizonEnd - AsOfDate).days

**Header row at row 6 (exactly these 14 column headers in order):**
1. Care_Unit
2. Current_Staff_Hours
3. Daily_Required_Hours
4. Current_Coverage_Days
5. Projected_Understaff_Date
6. Incoming_Hours_By_Aug31
7. Delivered_Coverage_To_Aug31
8. Remaining_August_Demand_Hours
9. Additional_Hours_Needed
10. Shift_Blocks_Required_Rounded_Up
11. Required_Shift_Start_Date
12. Rounding_Applied
13. Earlier_Shift_Required
14. Earliest_Scheduled_Shift_Date

**Data rows start at row 7**, one per entity from `Current Staffing`, preserving source order.

**Calculation rules (apply per entity/unit):**

Let:
- `AsOfDate` = date from Current Staffing!B1
- `PlanningHorizonEnd` = date from Current Staffing!D1
- `RemainingDaysInAugust` = (PlanningHorizonEnd - AsOfDate).days
- `current_hours` = entity's current staff hours from Current Staffing
- `daily_req` = entity's daily required hours from Current Staffing
- `ratio` = conversion ratio from Ratio sheet (hours per shift block)

Then:
- `Current_Coverage_Days` = current_hours / daily_req if daily_req > 0, else None (blank)
- `Projected_Understaff_Date` = AsOfDate + timedelta(days=floor(Current_Coverage_Days)) if daily_req > 0, else None — store as ISO string
- `Incoming_Hours_By_Aug31` = sum of inbound hours for this entity where inbound_date <= PlanningHorizonEnd
- `Delivered_Coverage_To_Aug31` = (current_hours + Incoming_Hours_By_Aug31) / daily_req if daily_req > 0, else None
- `Remaining_August_Demand_Hours` = daily_req * RemainingDaysInAugust
- `Additional_Hours_Needed` = max(0, Remaining_August_Demand_Hours - current_hours - Incoming_Hours_By_Aug31)
- `raw_blocks` = Additional_Hours_Needed / ratio (before ceiling)
- `Shift_Blocks_Required_Rounded_Up` = math.ceil(raw_blocks) if Additional_Hours_Needed > 0, else 0
- `Rounding_Applied` = True if Additional_Hours_Needed > 0 AND math.ceil(raw_blocks) != raw_blocks (i.e., raw_blocks is not already an integer); else False
  - **IMPORTANT**: Write Python boolean `True`/`False` so openpyxl stores them as Excel boolean. Do NOT write strings.
- `Earliest_Scheduled_Shift_Date` = earliest inbound date for this entity (regardless of whether <= PlanningHorizonEnd), else None — store as ISO string
- `Required_Shift_Start_Date`:
  - None (blank) if Shift_Blocks_Required_Rounded_Up == 0
  - else if Earliest_Scheduled_Shift_Date is not None AND Earliest_Scheduled_Shift_Date <= Projected_Understaff_Date: use AsOfDate + timedelta(days=floor(Delivered_Coverage_To_Aug31)) — store as ISO string
  - else: use Projected_Understaff_Date (already ISO string)
- `Earlier_Shift_Required` = True if Shift_Blocks_Required_Rounded_Up > 0 AND (Earliest_Scheduled_Shift_Date is None OR Required_Shift_Start_Date < Earliest_Scheduled_Shift_Date); else False
  - **IMPORTANT**: Write Python boolean `True`/`False`.

**Critical type rules:**
- All numeric fields must be stored as numbers (int or float), NOT strings.
- All date fields (Projected_Understaff_Date, Required_Shift_Start_Date, Earliest_Scheduled_Shift_Date) must be ISO strings "YYYY-MM-DD" or None for blank.
- Rounding_Applied and Earlier_Shift_Required must be Python booleans (True/False), which openpyxl will write as Excel booleans.
- B2 and B3 (metadata dates) must be ISO strings.
- B4 must be an integer.

#### Sheet 2: `Additional_Shifts_Needed`

**Header row at row 1 (exactly these 6 columns):**
1. Care_Unit
2. Required_Shift_Start_Date
3. Shift_Blocks_Required_Rounded_Up
4. Additional_Hours_Needed
5. Rounding_Applied
6. Earlier_Shift_Required

**Data rows:** Include only entities where Shift_Blocks_Required_Rounded_Up > 0, same order as Unit_Results. Same type rules apply.

### Step 3: Verify the Output

After writing the file, re-read it with openpyxl and verify:
1. Sheet names are exactly `['Unit_Results', 'Additional_Shifts_Needed']`.
2. Unit_Results metadata: A1="Field", B1="Value", A2="AsOfDate", B2 is ISO date string, A3="PlanningHorizonEnd", B3 is ISO date string, A4="RemainingDaysInAugust", B4 is int.
3. Unit_Results row 6 has exactly the 14 headers listed above.
4. Data rows start at row 7; count matches number of entities in source.
5. Numeric cells are numeric (not strings).
6. Boolean cells (Rounding_Applied, Earlier_Shift_Required) are actual booleans.
7. Date string cells match YYYY-MM-DD pattern.
8. Additional_Shifts_Needed has correct headers and only rows with Shift_Blocks_Required_Rounded_Up > 0.
9. Print all cell values for both sheets for visual confirmation.

### Constraints
- Do NOT modify the source file.
- The output file must be at exactly `/root/additional_shift_blocks_needed_august_2025.xlsx`.
- Use `import math` for `math.floor` and `math.ceil`.
- When comparing dates for Earliest_Scheduled_Shift_Date <= Projected_Understaff_Date, compare as date objects, not strings.
- When comparing Required_Shift_Start_Date < Earliest_Scheduled_Shift_Date for Earlier_Shift_Required, compare as date objects before converting to ISO strings.

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