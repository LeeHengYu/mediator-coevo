# Task Instruction

## Task: Build maintenance resupply planning workbook

Create a single Excel workbook at `/root/maintenance_resupply_actions_sep_2025.xlsx` from the source workbook `/root/Maintenance_Parts_and_Deliveries_Latest.xlsx`.

### Step 0: Inspect the source workbook

Before writing any code, read and print the structure and contents of all three sheets in the source workbook:
- `Current Parts` — print all rows including headers. Pay special attention to cells B1 and D1 (dates). Identify the column layout: which columns hold Part_Code, Current_Units, Daily_Consumption_Units, etc.
- `Scheduled Deliveries` — print all rows. Identify columns for Part_Code, inbound quantity, and inbound/delivery date.
- `Ratio` — print all rows. Identify the conversion ratio (units per crate) for each part.

Print the exact cell values, types, and shapes so you understand the data before coding.

### Step 1: Write a Python script using `openpyxl` to produce the output workbook

The script must:

#### 1a. Read source data
- Parse `Current Parts` to get:
  - `AsOfDate` from cell B1 (convert to date if needed)
  - `PlanningHorizonEnd` from cell D1 (convert to date if needed)
  - The entity rows (one per part) with Part_Code, Current_Units, Daily_Consumption_Units. Preserve source row order.
- Parse `Scheduled Deliveries` to get each delivery record: Part_Code, quantity, delivery date.
- Parse `Ratio` to get the crate conversion ratio per Part_Code (units per crate).

#### 1b. Compute derived values
- `RemainingDaysInSeptember` = (PlanningHorizonEnd - AsOfDate).days  (integer, calendar day difference)
- For each part row, compute all 14 columns per the rules below.

#### 1c. Calculation rules (follow exactly)

1. **Part_Code**: from source
2. **Current_Units**: from source (numeric)
3. **Daily_Consumption_Units**: from source (numeric)
4. **Current_DOH**: `Current_Units / Daily_Consumption_Units` if rate > 0, else `None` (leave cell blank)
5. **Projected_Stockout_Date**: `AsOfDate + timedelta(days=floor(Current_DOH))` if rate > 0, else blank. Store as ISO string `YYYY-MM-DD`.
6. **Inbound_Units_By_Sep30**: sum of inbound quantities for this part where inbound date <= PlanningHorizonEnd. If no deliveries match, use 0.
7. **Delivered_DOH_To_Sep30**: `(Current_Units + Inbound_Units_By_Sep30) / Daily_Consumption_Units` if rate > 0, else blank
8. **Remaining_September_Demand_Units**: `Daily_Consumption_Units * RemainingDaysInSeptember`
9. **Additional_Units_Needed**: `max(0, Remaining_September_Demand_Units - Current_Units - Inbound_Units_By_Sep30)`
10. **Crates_Required_Rounded_Up**: `math.ceil(Additional_Units_Needed / ratio)` if Additional_Units_Needed > 0, else 0. Use the per-part ratio from the Ratio sheet.
11. **Required_Delivery_Date** (see below)
12. **Rounding_Applied**: boolean `True` if Additional_Units_Needed > 0 AND `Crates_Required_Rounded_Up * ratio != Additional_Units_Needed`; else `False`.
13. **Earlier_Delivery_Required**: boolean `True` if Crates_Required_Rounded_Up > 0 AND (Earliest_Scheduled_Delivery_Date is blank OR Required_Delivery_Date < Earliest_Scheduled_Delivery_Date); else `False`.
14. **Earliest_Scheduled_Delivery_Date**: the earliest scheduled inbound date for this part (across all deliveries regardless of date filter), else blank. Store as ISO string `YYYY-MM-DD`.

**Required_Delivery_Date logic:**
- If Crates_Required_Rounded_Up == 0: blank
- Else if Earliest_Scheduled_Delivery_Date is not blank AND Earliest_Scheduled_Delivery_Date <= Projected_Stockout_Date: use `AsOfDate + timedelta(days=floor(Delivered_DOH_To_Sep30))`, as ISO string
- Else: use Projected_Stockout_Date (already ISO string)

#### 1d. Build Sheet 1: `Part_Results`

- Metadata block:
  - A1="Field", B1="Value"
  - A2="AsOfDate", B2=AsOfDate as `YYYY-MM-DD` string
  - A3="PlanningHorizonEnd", B3=PlanningHorizonEnd as `YYYY-MM-DD` string
  - A4="RemainingDaysInSeptember", B4=integer
- Row 6 is the header row with exactly these 14 column names in this order:
  `Part_Code, Current_Units, Daily_Consumption_Units, Current_DOH, Projected_Stockout_Date, Inbound_Units_By_Sep30, Delivered_DOH_To_Sep30, Remaining_September_Demand_Units, Additional_Units_Needed, Crates_Required_Rounded_Up, Required_Delivery_Date, Rounding_Applied, Earlier_Delivery_Required, Earliest_Scheduled_Delivery_Date`
- Data rows start at row 7, one per part, in source order.
- Numeric fields must be stored as numbers (int or float), not strings.
- Boolean fields (`Rounding_Applied`, `Earlier_Delivery_Required`) must be stored as Python `True`/`False` booleans so openpyxl writes them as Excel booleans.
- Date columns (`Projected_Stockout_Date`, `Required_Delivery_Date`, `Earliest_Scheduled_Delivery_Date`) must be stored as strings in `YYYY-MM-DD` format (or None for blank).

#### 1e. Build Sheet 2: `Additional_Resupply_Needed`

- Header row at row 1 with exactly these 6 columns:
  `Part_Code, Required_Delivery_Date, Crates_Required_Rounded_Up, Additional_Units_Needed, Rounding_Applied, Earlier_Delivery_Required`
- Include only rows where `Crates_Required_Rounded_Up > 0`.
- Same order as Part_Results.
- Same data types as Part_Results for each column.

#### 1f. Save and verify

- Save to `/root/maintenance_resupply_actions_sep_2025.xlsx`
- Do NOT modify any source files.
- After saving, re-open the output workbook and print:
  - Sheet names
  - Part_Results metadata cells (A1:B4)
  - Part_Results header row (row 6)
  - All data rows from both sheets
  - Verify boolean cells are actual booleans (print type)
  - Verify date cells are strings
  - Verify numeric cells are numbers

### Important notes
- The output workbook must have exactly two sheets in order: `Part_Results`, `Additional_Resupply_Needed`.
- Rows 5 in Part_Results should be empty (metadata ends at row 4, header at row 6).
- Use `import math` for `math.floor` and `math.ceil`.
- Use `from datetime import datetime, timedelta` for date arithmetic.
- When comparing dates from the source (which may be datetime objects), convert consistently.
- If the Ratio sheet has a single universal ratio rather than per-part ratios, use that single ratio for all parts. Print what you find.
- Double-check the column mapping in the source sheets by printing headers before assuming positions.

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