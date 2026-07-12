# Task Instruction

Execute the following steps to build the output workbook.

## 1. Inspect the source workbook

```python
import openpyxl, datetime, math

wb_src = openpyxl.load_workbook('/root/Staffing_and_Shifts_Latest.xlsx', data_only=True)
print('Sheet names:', wb_src.sheetnames)

for sn in wb_src.sheetnames:
    ws = wb_src[sn]
    print(f'\n=== {sn} === (rows={ws.max_row}, cols={ws.max_column})')
    for r in range(1, min(ws.max_row+1, 25)):
        row_vals = []
        for c in range(1, min(ws.max_column+1, 15)):
            row_vals.append(ws.cell(r, c).value)
        print(f'  Row {r}: {row_vals}')
```

Run this first and read ALL output carefully. Identify:
- The AsOfDate from Current Staffing!B1
- The PlanningHorizonEnd from Current Staffing!D1
- The list of care-unit entities and their columns (name, current staff hours, daily required hours)
- The structure of the Incoming Shifts sheet (entity name, date, quantity columns)
- The conversion ratio from the Ratio sheet

## 2. Build the output workbook

After inspecting, write a single Python script that:

a) Reads the source data without modifying it.

b) Computes all values per the calculation rules below.

c) Creates `/root/additional_shift_blocks_needed_august_2025.xlsx` with exactly two sheets in order: `Unit_Results`, `Additional_Shifts_Needed`.

### Calculation rules (adapt column references based on what you found in step 1):

- **AsOfDate** = date value from Current Staffing!B1
- **PlanningHorizonEnd** = date value from Current Staffing!D1
- **RemainingDaysInAugust** = (PlanningHorizonEnd - AsOfDate).days  (calendar day difference)
- For each care unit row in Current Staffing (preserve source order):
  - **Care_Unit** = entity name
  - **Current_Staff_Hours** = from source
  - **Daily_Required_Hours** = from source
  - **Current_Coverage_Days** = Current_Staff_Hours / Daily_Required_Hours when Daily_Required_Hours > 0, else None
  - **Projected_Understaff_Date** = AsOfDate + timedelta(days=floor(Current_Coverage_Days)) when computable, else None. Store as ISO string YYYY-MM-DD.
  - **Incoming_Hours_By_Aug31** = sum of inbound quantity for that entity where inbound date <= PlanningHorizonEnd. Parse dates carefully (could be datetime or string). If no matching rows, use 0.
  - **Delivered_Coverage_To_Aug31** = (Current_Staff_Hours + Incoming_Hours_By_Aug31) / Daily_Required_Hours when rate > 0, else None
  - **Remaining_August_Demand_Hours** = Daily_Required_Hours * RemainingDaysInAugust
  - **Additional_Hours_Needed** = max(0, Remaining_August_Demand_Hours - Current_Staff_Hours - Incoming_Hours_By_Aug31)
  - **conversion_ratio** = the value from the Ratio sheet (likely a single number; inspect carefully)
  - **Shift_Blocks_Required_Rounded_Up** = math.ceil(Additional_Hours_Needed / conversion_ratio) when Additional_Hours_Needed > 0, else 0
  - **Earliest_Scheduled_Shift_Date** = earliest scheduled inbound date for the entity (any inbound row, not just those <= PlanningHorizonEnd), else None. ISO string.
  - **Required_Shift_Start_Date**:
    - None when Shift_Blocks_Required_Rounded_Up == 0
    - else if Earliest_Scheduled_Shift_Date is not None and Earliest_Scheduled_Shift_Date <= Projected_Understaff_Date: use AsOfDate + timedelta(days=floor(Delivered_Coverage_To_Aug31)), as ISO string
    - else: use Projected_Understaff_Date (already ISO string)
  - **Rounding_Applied** = True when Additional_Hours_Needed > 0 AND (Additional_Hours_Needed / conversion_ratio) != math.ceil(Additional_Hours_Needed / conversion_ratio); else False
  - **Earlier_Shift_Required** = True when Shift_Blocks_Required_Rounded_Up > 0 AND (Earliest_Scheduled_Shift_Date is None OR Required_Shift_Start_Date < Earliest_Scheduled_Shift_Date); else False

### Sheet 1: Unit_Results layout
- A1="Field", B1="Value"
- A2="AsOfDate", B2=AsOfDate as ISO string
- A3="PlanningHorizonEnd", B3=PlanningHorizonEnd as ISO string
- A4="RemainingDaysInAugust", B4=integer
- Row 6 = header row with the 14 column names exactly as specified
- Data rows start at row 7, one per entity in source order
- Date columns (Projected_Understaff_Date, Required_Shift_Start_Date, Earliest_Scheduled_Shift_Date) as ISO strings
- Boolean columns (Rounding_Applied, Earlier_Shift_Required) as Python True/False booleans
- Numeric columns as numbers (int or float)

### Sheet 2: Additional_Shifts_Needed layout
- Row 1 = header with 6 columns exactly as specified
- Only rows where Shift_Blocks_Required_Rounded_Up > 0
- Same entity order as Unit_Results
- Same data types as Unit_Results

## 3. Validate the output

After creating the file, re-open it and print:
- Sheet names and their order
- All metadata cells (A1:B4) from Unit_Results
- The header row from Unit_Results
- All data rows from Unit_Results
- The header row from Additional_Shifts_Needed
- All data rows from Additional_Shifts_Needed

Verify:
- Exactly 2 sheets in correct order
- All 14 columns present in Unit_Results header
- All 6 columns present in Additional_Shifts_Needed header
- Date fields are ISO strings not datetime objects
- Boolean fields are True/False not strings
- Numeric fields are numbers not strings
- Additional_Shifts_Needed only contains rows with Shift_Blocks_Required_Rounded_Up > 0

If anything looks wrong, fix and regenerate.

## Important notes
- Do NOT modify the source file.
- Be very careful with date parsing: source dates might be datetime objects or strings. Convert appropriately.
- When comparing dates for the Required_Shift_Start_Date logic, compare as date objects, not strings.
- The Ratio sheet may have a single conversion value or one per entity—inspect and handle accordingly.
- Use openpyxl for both reading and writing.

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