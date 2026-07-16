# Task Instruction

Execute the following steps to produce /root/maintenance_resupply_actions_sep_2025.xlsx.

## Step 1 – Inspect the source workbook

```python
import openpyxl
wb = openpyxl.load_workbook('/root/Maintenance_Parts_and_Deliveries_Latest.xlsx', data_only=True)
for s in wb.sheetnames:
    ws = wb[s]
    print(f'\n=== {s} === (rows={ws.max_row}, cols={ws.max_column})')
    for r in ws.iter_rows(min_row=1, max_row=min(ws.max_row, 20), values_only=False):
        print([(c.coordinate, c.value) for c in r])
```

Read every cell carefully. Identify:
- In "Current Parts": where AsOfDate (B1) and PlanningHorizonEnd (D1) live; the header row; the columns for Part_Code, Current_Units, Daily_Consumption_Units; the row order of entities.
- In "Scheduled Deliveries": header row; columns for Part_Code (or equivalent entity identifier), inbound quantity, inbound/delivery date.
- In "Ratio": the conversion ratio (units per crate or similar). Note the exact cell and value.

Print all of these clearly before proceeding.

## Step 2 – Build the output workbook

Write a single Python script that:

```python
import openpyxl, math, datetime
from openpyxl import Workbook

# 1. Load source
src = openpyxl.load_workbook('/root/Maintenance_Parts_and_Deliveries_Latest.xlsx', data_only=True)

# 2. Read Current Parts
cp = src['Current Parts']
# Extract AsOfDate and PlanningHorizonEnd from the exact cells identified in Step 1.
# Extract entity rows (Part_Code, Current_Units, Daily_Consumption_Units) preserving source order.

# 3. Read Scheduled Deliveries
sd = src['Scheduled Deliveries']
# Build a dict: part_code -> list of (delivery_date, quantity)

# 4. Read Ratio
ratio_ws = src['Ratio']
# Extract the conversion ratio (units per crate). If there is one ratio per part, build a dict.

# 5. Compute metadata
# AsOfDate, PlanningHorizonEnd must be datetime.date objects.
# RemainingDaysInSeptember = (PlanningHorizonEnd - AsOfDate).days

# 6. For each entity compute all 14 fields:
#    a. Current_DOH = Current_Units / Daily_Consumption_Units  (if rate > 0, else None)
#    b. Projected_Stockout_Date = AsOfDate + timedelta(days=math.floor(Current_DOH))  (if rate > 0, else None)
#    c. Inbound_Units_By_Sep30 = sum of qty where delivery_date <= PlanningHorizonEnd
#    d. Delivered_DOH_To_Sep30 = (Current_Units + Inbound_Units_By_Sep30) / Daily_Consumption_Units  (if rate > 0, else None)
#    e. Remaining_September_Demand_Units = Daily_Consumption_Units * RemainingDaysInSeptember
#    f. Additional_Units_Needed = max(0, Remaining_September_Demand_Units - Current_Units - Inbound_Units_By_Sep30)
#    g. Crates_Required_Rounded_Up = math.ceil(Additional_Units_Needed / ratio) if Additional_Units_Needed > 0 else 0
#    h. Earliest_Scheduled_Delivery_Date = min of delivery dates for this part (or None)
#    i. Required_Delivery_Date:
#         - None if Crates_Required_Rounded_Up == 0
#         - else if Earliest_Scheduled_Delivery_Date is not None and Earliest_Scheduled_Delivery_Date <= Projected_Stockout_Date:
#              AsOfDate + timedelta(days=math.floor(Delivered_DOH_To_Sep30))
#         - else: Projected_Stockout_Date
#    j. Rounding_Applied:
#         - True if Additional_Units_Needed > 0 and (Additional_Units_Needed / ratio) != Crates_Required_Rounded_Up
#           (i.e., ceil changed the value, meaning Additional_Units_Needed % ratio != 0)
#         - else False
#    k. Earlier_Delivery_Required:
#         - True if Crates_Required_Rounded_Up > 0 and (Earliest_Scheduled_Delivery_Date is None or Required_Delivery_Date < Earliest_Scheduled_Delivery_Date)
#         - else False

# 7. Write Sheet 1: Part_Results
# Metadata in A1:B4, header in row 6, data from row 7.
# Date columns (Projected_Stockout_Date, Required_Delivery_Date, Earliest_Scheduled_Delivery_Date)
#   must be ISO strings (str in YYYY-MM-DD format), not datetime objects.
# Boolean columns (Rounding_Applied, Earlier_Delivery_Required) must be Python bool (True/False).
# Numeric columns must remain numeric (int or float).
# Blank means None (do not write empty string).

# 8. Write Sheet 2: Additional_Resupply_Needed
# Header in row 1. Only rows where Crates_Required_Rounded_Up > 0.
# Same entity order as Part_Results. Columns:
#   Part_Code, Required_Delivery_Date, Crates_Required_Rounded_Up,
#   Additional_Units_Needed, Rounding_Applied, Earlier_Delivery_Required

out = Workbook()
# ... build sheets ...
out.save('/root/maintenance_resupply_actions_sep_2025.xlsx')
print('Saved successfully.')
```

IMPORTANT DETAILS:
- When reading dates from the source workbook, handle both datetime objects and strings. Convert to datetime.date for arithmetic.
- The ratio sheet may have one global ratio or per-part ratios. Inspect carefully in Step 1 and adapt.
- Ensure Part_Code matching between Current Parts and Scheduled Deliveries is exact (strip whitespace if needed).
- Use `math.floor` for DOH-to-days conversion, `math.ceil` for crate rounding.
- For Rounding_Applied: compare `Additional_Units_Needed / ratio` to `math.ceil(Additional_Units_Needed / ratio)`. If they differ, True.
- For Earlier_Delivery_Required: compare dates properly. If Earliest_Scheduled_Delivery_Date is None, that counts as True (when crates > 0).

## Step 3 – Validate the output

After saving, reload the output file and print:
1. Sheet names (must be exactly ['Part_Results', 'Additional_Resupply_Needed'])
2. All metadata cells (A1:B4)
3. Header row (row 6) of Part_Results
4. All data rows of Part_Results with types noted for date/bool/numeric columns
5. Header row and all data rows of Additional_Resupply_Needed
6. Confirm: date columns contain strings (not datetime), booleans are bool, numerics are int/float
7. Confirm: Additional_Resupply_Needed only has rows where Crates_Required_Rounded_Up > 0
8. Confirm: entity order matches between sheets

If any issue is found, fix and re-save before finishing.

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