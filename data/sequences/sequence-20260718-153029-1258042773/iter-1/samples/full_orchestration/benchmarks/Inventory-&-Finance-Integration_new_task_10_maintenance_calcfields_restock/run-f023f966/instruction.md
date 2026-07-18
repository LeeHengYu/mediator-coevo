# Task Instruction

## Task: Build maintenance resupply planning workbook

Create `/root/maintenance_resupply_actions_sep_2025.xlsx` from source `/root/Maintenance_Parts_and_Deliveries_Latest.xlsx`.

### Step 0: Inspect the source workbook thoroughly

Open `/root/Maintenance_Parts_and_Deliveries_Latest.xlsx` and inspect ALL three sheets:
1. **Current Parts** – Print all cells in the first few rows to find the layout. Specifically identify:
   - B1 (AsOfDate) and D1 (PlanningHorizonEnd) – note their types (datetime or string)
   - Where the parts data header row starts and what the column names are
   - All part rows and their fields (Part_Code, Current_Units, Daily_Consumption_Units, etc.)
2. **Scheduled Deliveries** – Print all rows. Identify columns for part code, delivery date, and inbound quantity. Note exact column names.
3. **Ratio** – Print all rows. Identify the conversion ratio (units per crate) for each part code. Note exact column names.

Print everything you find so you have full visibility before coding.

### Step 1: Compute all values in Python

Using openpyxl (or pandas for reading, openpyxl for writing), compute:

**Metadata:**
- `AsOfDate` = date from Current Parts!B1
- `PlanningHorizonEnd` = date from Current Parts!D1  
- `RemainingDaysInSeptember` = (PlanningHorizonEnd - AsOfDate).days  (calendar day difference, integer)

**Per-part calculations (preserve source order from Current Parts):**

For each part:
1. `Part_Code` – from source
2. `Current_Units` – from source (keep numeric)
3. `Daily_Consumption_Units` – from source (keep numeric)
4. `Current_DOH` = Current_Units / Daily_Consumption_Units when rate > 0, else leave blank (None)
5. `Projected_Stockout_Date` = AsOfDate + timedelta(days=floor(Current_DOH)) when rate > 0, else blank. Store as ISO string YYYY-MM-DD.
6. `Inbound_Units_By_Sep30` = sum of inbound quantities from Scheduled Deliveries for this part where delivery_date <= PlanningHorizonEnd. If no deliveries match, use 0.
7. `Delivered_DOH_To_Sep30` = (Current_Units + Inbound_Units_By_Sep30) / Daily_Consumption_Units when rate > 0, else blank
8. `Remaining_September_Demand_Units` = Daily_Consumption_Units * RemainingDaysInSeptember
9. `Additional_Units_Needed` = max(0, Remaining_September_Demand_Units - Current_Units - Inbound_Units_By_Sep30)
10. Look up the conversion ratio for this part from the Ratio sheet. `Crates_Required_Rounded_Up` = math.ceil(Additional_Units_Needed / ratio) when Additional_Units_Needed > 0, else 0 (integer)
11. `Earliest_Scheduled_Delivery_Date` = earliest delivery date for this part from Scheduled Deliveries (any date, not just <= Sep30), else blank. Store as ISO string.
12. `Required_Delivery_Date`:
    - blank (None) when Crates_Required_Rounded_Up == 0
    - else if Earliest_Scheduled_Delivery_Date is not blank AND earliest_date <= Projected_Stockout_Date: use AsOfDate + timedelta(days=floor(Delivered_DOH_To_Sep30)), as ISO string
    - else: use Projected_Stockout_Date (already ISO string)
13. `Rounding_Applied` = Python boolean True when Additional_Units_Needed > 0 AND (Crates_Required_Rounded_Up * ratio) != Additional_Units_Needed; else False
14. `Earlier_Delivery_Required` = Python boolean True when Crates_Required_Rounded_Up > 0 AND (Earliest_Scheduled_Delivery_Date is blank OR Required_Delivery_Date < Earliest_Scheduled_Delivery_Date); else False

**IMPORTANT on booleans:** Write Python `True`/`False` (which openpyxl stores as Excel booleans), NOT strings.

**IMPORTANT on dates:** Projected_Stockout_Date, Required_Delivery_Date, and Earliest_Scheduled_Delivery_Date must be written as ISO format strings ("YYYY-MM-DD"), not datetime objects.

**IMPORTANT on numerics:** Current_Units, Daily_Consumption_Units, Current_DOH, Inbound_Units_By_Sep30, Delivered_DOH_To_Sep30, Remaining_September_Demand_Units, Additional_Units_Needed, Crates_Required_Rounded_Up, RemainingDaysInSeptember must all be numeric (int or float), not strings.

### Step 2: Write the output workbook

Create a new workbook with exactly two sheets in this order:
1. **Part_Results**
2. **Additional_Resupply_Needed**

Remove any default sheets (like "Sheet").

#### Part_Results sheet:
- A1="Field", B1="Value"
- A2="AsOfDate", B2=AsOfDate as ISO string
- A3="PlanningHorizonEnd", B3=PlanningHorizonEnd as ISO string
- A4="RemainingDaysInSeptember", B4=integer
- Row 6 is the header row with exactly these 14 columns in order:
  Part_Code, Current_Units, Daily_Consumption_Units, Current_DOH, Projected_Stockout_Date, Inbound_Units_By_Sep30, Delivered_DOH_To_Sep30, Remaining_September_Demand_Units, Additional_Units_Needed, Crates_Required_Rounded_Up, Required_Delivery_Date, Rounding_Applied, Earlier_Delivery_Required, Earliest_Scheduled_Delivery_Date
- Data rows start at row 7, one per part in source order.

#### Additional_Resupply_Needed sheet:
- Row 1 header with exactly these 6 columns:
  Part_Code, Required_Delivery_Date, Crates_Required_Rounded_Up, Additional_Units_Needed, Rounding_Applied, Earlier_Delivery_Required
- Include only parts where Crates_Required_Rounded_Up > 0, same order as Part_Results.

### Step 3: Save and verify

Save to `/root/maintenance_resupply_actions_sep_2025.xlsx`.

Then re-open the file and:
1. Print sheet names and confirm exactly ["Part_Results", "Additional_Resupply_Needed"]
2. Print Part_Results rows 1-4 (metadata) and row 6 (headers)
3. Print all data rows from Part_Results, checking types (numeric vs string vs bool vs None)
4. Print all rows from Additional_Resupply_Needed
5. Verify: B2 and B3 are strings matching YYYY-MM-DD pattern, B4 is int
6. Verify: date columns contain strings, numeric columns contain numbers, boolean columns contain bools
7. Verify: Additional_Resupply_Needed only has rows with Crates_Required_Rounded_Up > 0

Do NOT modify the source file. The final deliverable is the .xlsx at the specified path.

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