# Task Instruction

Execute the following steps to produce /root/maintenance_resupply_actions_sep_2025.xlsx.

## Step 1 – Inspect the source workbook

```python
import openpyxl, json
wb = openpyxl.load_workbook('/root/Maintenance_Parts_and_Deliveries_Latest.xlsx', data_only=True)
for s in wb.sheetnames:
    ws = wb[s]
    print(f'\n=== {s} === (rows={ws.max_row}, cols={ws.max_column})')
    for r in ws.iter_rows(min_row=1, max_row=min(ws.max_row, 25), values_only=False):
        print([(c.coordinate, c.value) for c in r])
```

Read and understand:
- **Current Parts**: where AsOfDate (B1), PlanningHorizonEnd (D1) live, the header row, column layout (Part_Code, Current_Units, Daily_Consumption_Units, etc.).
- **Scheduled Deliveries**: columns for part code, inbound date, inbound quantity.
- **Ratio**: the conversion ratio (units per crate) for each part or a single global ratio.

Print everything so you know exact column names, date formats, and data types.

## Step 2 – Build the output workbook with a Python script

Create `/root/solve.py` that:

1. Reads the source workbook with openpyxl (data_only=True).
2. Extracts AsOfDate and PlanningHorizonEnd from Current Parts (convert to `datetime.date` if needed).
3. Computes RemainingDaysInSeptember = (PlanningHorizonEnd - AsOfDate).days.
4. Reads all part rows from Current Parts into a list preserving source order.
5. Reads all scheduled deliveries into a dict keyed by part code → list of (date, qty).
6. Reads the Ratio sheet to get the crate conversion ratio per part (or a single ratio if that's the structure).
7. For each part, computes every field per the rules below.
8. Writes Sheet 1 (Part_Results) with metadata in A1:B4, header at row 6, data from row 7.
9. Writes Sheet 2 (Additional_Resupply_Needed) with header at row 1, filtered rows where Crates_Required_Rounded_Up > 0, same source order.
10. Saves to `/root/maintenance_resupply_actions_sep_2025.xlsx`.

### Calculation rules (implement exactly):

```
import math, datetime

for each part:
  current_units = <from source>
  daily = <Daily_Consumption_Units from source>

  # Current_DOH
  if daily > 0:
      current_doh = current_units / daily
  else:
      current_doh = None  # blank

  # Projected_Stockout_Date
  if daily > 0:
      projected_stockout = as_of_date + datetime.timedelta(days=math.floor(current_doh))
  else:
      projected_stockout = None

  # Inbound_Units_By_Sep30
  inbound = sum(qty for (d, qty) in deliveries[part_code] if d <= planning_end)

  # Delivered_DOH_To_Sep30
  if daily > 0:
      delivered_doh = (current_units + inbound) / daily
  else:
      delivered_doh = None

  # Remaining_September_Demand_Units
  remaining_demand = daily * remaining_days

  # Additional_Units_Needed
  additional = max(0, remaining_demand - current_units - inbound)

  # Crates_Required_Rounded_Up
  ratio = <from Ratio sheet for this part>
  if additional > 0:
      raw_crates = additional / ratio
      crates = math.ceil(raw_crates)
  else:
      crates = 0

  # Earliest_Scheduled_Delivery_Date
  if deliveries[part_code]:
      earliest_del = min(d for (d, q) in deliveries[part_code])
  else:
      earliest_del = None

  # Required_Delivery_Date
  if crates == 0:
      req_date = None
  elif earliest_del is not None and projected_stockout is not None and earliest_del <= projected_stockout:
      req_date = as_of_date + datetime.timedelta(days=math.floor(delivered_doh))
  else:
      req_date = projected_stockout  # could be None if daily==0, but crates>0 implies daily>0

  # Rounding_Applied
  if additional > 0 and crates != raw_crates:  # i.e. ceil changed the value
      rounding = True
  else:
      rounding = False

  # Earlier_Delivery_Required
  if crates > 0 and (earliest_del is None or (req_date is not None and req_date < earliest_del)):
      earlier = True
  else:
      earlier = False
```

### Writing rules:
- Date columns (Projected_Stockout_Date, Required_Delivery_Date, Earliest_Scheduled_Delivery_Date) must be written as ISO strings `YYYY-MM-DD` (use `str(date_obj)` or `date_obj.strftime('%Y-%m-%d')`).
- B2 and B3 metadata cells must also be ISO date strings.
- Numeric fields (Current_Units, Daily_Consumption_Units, Current_DOH, Inbound_Units_By_Sep30, Delivered_DOH_To_Sep30, Remaining_September_Demand_Units, Additional_Units_Needed, Crates_Required_Rounded_Up) must be Python int or float, NOT strings.
- Rounding_Applied and Earlier_Delivery_Required must be Python bool (True/False).
- Blank means None (don't write anything to that cell).
- Sheet order: Part_Results first, Additional_Resupply_Needed second.

## Step 3 – Run and validate

```bash
cd /root && python solve.py
```

Then validate the output:

```python
import openpyxl
wb = openpyxl.load_workbook('/root/maintenance_resupply_actions_sep_2025.xlsx', data_only=True)
print('Sheets:', wb.sheetnames)
for s in wb.sheetnames:
    ws = wb[s]
    print(f'\n=== {s} ===')
    for r in ws.iter_rows(min_row=1, max_row=min(ws.max_row, 30), values_only=False):
        print([(c.coordinate, c.value, type(c.value).__name__) for c in r])
```

Check:
1. Sheet names are exactly `['Part_Results', 'Additional_Resupply_Needed']`.
2. Part_Results metadata: A1='Field', B1='Value', A2='AsOfDate', B2=ISO date string, A3='PlanningHorizonEnd', B3=ISO date string, A4='RemainingDaysInSeptember', B4=integer.
3. Part_Results header at row 6 has exactly the 14 columns specified.
4. Data rows start at row 7, one per part, source order preserved.
5. Date columns contain strings in YYYY-MM-DD format or None.
6. Numeric columns contain int/float, not strings.
7. Boolean columns contain True/False, not strings or 0/1.
8. Additional_Resupply_Needed has exactly 6 columns, only rows with Crates_Required_Rounded_Up > 0, same order as Part_Results.
9. No extra sheets exist.

If anything is wrong, fix solve.py and re-run until all checks pass.

## Important notes
- Do NOT modify the source file.
- Pay close attention to the exact column names/mappings in the source sheets – they may differ from the output column names. Map them carefully after inspection.
- The Ratio sheet may have one ratio per part or a single global ratio. Inspect it and handle accordingly.
- For date handling: source dates may be datetime objects from openpyxl. Convert them to `datetime.date` before arithmetic.
- Use `math.floor` for DOH-based day calculations and `math.ceil` for crate rounding.
- The Rounding_Applied check: `additional > 0 and math.ceil(additional/ratio) != additional/ratio` (i.e., the division was not exact).

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