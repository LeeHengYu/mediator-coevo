# Task Instruction

Execute the following steps in order:

## Step 1: Inspect the input files

```python
import openpyxl

# Read demand data
wb = openpyxl.load_workbook('/root/hvac_demand_sheet.xlsx', data_only=True)
ws = wb['Install']
print('=== Install sheet ===')
for row in ws.iter_rows(min_row=1, max_row=ws.max_row, values_only=False):
    vals = [(c.value, c.column) for c in row]
    print(vals)
wb.close()

# Read existing plan
wb2 = openpyxl.load_workbook('/root/hvac_existing_plan.xlsx', data_only=True)
ws2 = wb2['Plan']
print('\n=== Existing Plan sheet ===')
for row in ws2.iter_rows(min_row=1, max_row=min(ws2.max_row, 25), values_only=False):
    vals = [(c.value, c.column) for c in row]
    print(vals)
wb2.close()
```

Examine the output carefully. Identify:
- Which row contains `HVAC Ductwork Demand (Std Hrs)` (or similar label)
- Which columns correspond to Phases 8 through 56
- The demand values for each phase
- The header row of the existing plan

## Step 2: Build the schedule using a Python script

Write and run a single Python script that does everything:

```python
import openpyxl

# --- 1. Read demand from hvac_demand_sheet.xlsx, sheet 'Install' ---
wb_demand = openpyxl.load_workbook('/root/hvac_demand_sheet.xlsx', data_only=True)
ws_demand = wb_demand['Install']

# Find the row labeled 'HVAC Ductwork Demand (Std Hrs)' (check column A or B for the label)
# and extract demand values for phases 8..56.
# First, find header row that contains phase numbers, and the demand row.
phase_row = None
demand_row = None
for row in ws_demand.iter_rows(min_row=1, max_row=ws_demand.max_row):
    for cell in row:
        v = cell.value
        if isinstance(v, str) and 'Ductwork Demand' in v:
            demand_row = cell.row
        # Phase numbers might be in a header row
        if v == 8 or v == '8':
            if phase_row is None:
                phase_row = cell.row

print(f'Phase row: {phase_row}, Demand row: {demand_row}')

# Build mapping: phase_number -> demand_value
# Read phase numbers from phase_row
phase_to_col = {}
for cell in ws_demand[phase_row]:
    v = cell.value
    if v is not None:
        try:
            p = int(v) if not isinstance(v, (int, float)) else int(v)
            if 8 <= p <= 56:
                phase_to_col[p] = cell.column
        except (ValueError, TypeError):
            pass

demand = {}
for phase_num, col in phase_to_col.items():
    val = ws_demand.cell(row=demand_row, column=col).value
    demand[phase_num] = float(val) if val is not None else 0.0

print('Demand values:')
for p in sorted(demand.keys()):
    print(f'  Phase {p}: {demand[p]}')

wb_demand.close()

# --- 2. Compute schedule for phases 8..56 ---
phases = list(range(8, 57))  # 8 to 56 inclusive = 49 phases

# Initial condition: at Phase 8, Calc Start + Scheduled Demand = 1138.66
# So calc_start_phase8 = 1138.66 - demand[8]
initial_total = 1138.66
calc_start_8 = initial_total - demand[8]

results = []  # list of dicts

prev_end_backlog = calc_start_8  # This is the 'prior phase End of Phase Backlog/Buffer' for phase 8's perspective
# Actually, for Phase 8: Calc Start = prior phase End of Phase Backlog/Buffer
# And we need: Calc Start + Scheduled Demand for phase 8 = initial condition value
# So: Calc Start for phase 8 = 1138.66 - demand[8]
# But Calc Start = prior phase End of Phase Backlog/Buffer
# So we set prev_end_backlog = 1138.66 - demand[8]

for phase in phases:
    sched_demand = demand.get(phase, 0.0)
    
    if phase == 8:
        calc_start = initial_total - sched_demand
    else:
        calc_start = prev_end_backlog
    
    start_past_due = max(0.0, calc_start)
    
    # Choose Days Worked
    if start_past_due > 0.01:
        # Try smallest in {5, 6} such that calc_start + demand - 35*days <= 0
        chosen = None
        for d in [5, 6]:
            if calc_start + sched_demand - 35 * d <= 0:
                chosen = d
                break
        if chosen is None:
            chosen = 6
        days_worked = chosen
    else:
        # start_past_due <= 0.01
        if sched_demand <= 140:
            days_worked = 4
        else:
            days_worked = 5
    
    weekly_capacity = 35.0 * days_worked
    end_backlog = calc_start + sched_demand - weekly_capacity
    overtime = 10.0 * max(0, days_worked - 4)
    
    results.append({
        'Phase': phase,
        'Days Worked': days_worked,
        'Scheduled Demand (Std Hrs)': round(sched_demand, 2),
        'Weekly Capacity (Std Hrs)': round(weekly_capacity, 2),
        'Start of Phase Past Due (Std Hrs)': round(start_past_due, 2),
        'End of Phase Backlog/Buffer (Std Hrs)': round(end_backlog, 2),
        'Overtime Hours': round(overtime, 2),
    })
    
    prev_end_backlog = end_backlog

# Print for verification
for r in results:
    print(r)

# Verify phase 8 initial condition
r8 = results[0]
print(f"\nPhase 8 check: Start Past Due ({r8['Start of Phase Past Due (Std Hrs)']}) + Demand ({r8['Scheduled Demand (Std Hrs)']}) = {r8['Start of Phase Past Due (Std Hrs)'] + r8['Scheduled Demand (Std Hrs)']}")
print(f"Expected: {initial_total}")

# --- 3. Write hvac_existing_plan.xlsx (overwrite) and hvac_schedule_plan.xlsx ---
headers = [
    'Phase',
    'Days Worked',
    'Scheduled Demand (Std Hrs)',
    'Weekly Capacity (Std Hrs)',
    'Start of Phase Past Due (Std Hrs)',
    'End of Phase Backlog/Buffer (Std Hrs)',
    'Overtime Hours',
]

for filepath in ['/root/hvac_existing_plan.xlsx', '/root/hvac_schedule_plan.xlsx']:
    wb_out = openpyxl.Workbook()
    ws_out = wb_out.active
    ws_out.title = 'Plan'
    
    # Write headers
    for col_idx, h in enumerate(headers, start=1):
        ws_out.cell(row=1, column=col_idx, value=h)
    
    # Write data rows - ensure numeric types
    for row_idx, r in enumerate(results, start=2):
        for col_idx, h in enumerate(headers, start=1):
            val = r[h]
            # Phase and Days Worked should be int
            if h in ('Phase', 'Days Worked'):
                ws_out.cell(row=row_idx, column=col_idx, value=int(val))
            else:
                ws_out.cell(row=row_idx, column=col_idx, value=float(val))
    
    wb_out.save(filepath)
    print(f'Saved {filepath}')

# --- 4. Create summary file ---
# Find first week with 5 days and first week with 4 days
first_5 = None
first_4 = None
for r in results:
    if r['Days Worked'] == 5 and first_5 is None:
        first_5 = r['Phase']
    if r['Days Worked'] == 4 and first_4 is None:
        first_4 = r['Phase']

first_5_str = str(first_5) if first_5 is not None else 'N/A'
first_4_str = str(first_4) if first_4 is not None else 'N/A'

# Build summary (<=60 words, <=3 sentences, mention both step-down phase numbers)
summary = f"The crew starts at 6-day weeks to clear a large backlog, steps down to 5-day weeks at Phase {first_5_str}, and reaches 4-day weeks at Phase {first_4_str}. Overtime decreases as the backlog is retired. The plan covers Phases 8 through 56 with demand-driven staffing adjustments."

# Verify word count
word_count = len(summary.split())
print(f'Summary word count: {word_count}')
if word_count > 60:
    print('WARNING: summary exceeds 60 words, need to shorten')

with open('/root/hvac_schedule_summary.txt', 'w') as f:
    f.write(f'First_Week_5_Days: {first_5_str}\n')
    f.write(f'First_Week_4_Days: {first_4_str}\n')
    f.write(f'Summary: {summary}\n')

print('\nSummary file written.')
print(f'First_Week_5_Days: {first_5_str}')
print(f'First_Week_4_Days: {first_4_str}')
```

## Step 3: Validate outputs

After running the script:

1. Re-read `/root/hvac_schedule_plan.xlsx` and print all rows to confirm 49 data rows, correct headers, numeric values.
2. Re-read `/root/hvac_existing_plan.xlsx` and confirm it matches.
3. Read `/root/hvac_schedule_summary.txt` and print its contents.
4. Verify:
   - Phase 8: `Start of Phase Past Due + Scheduled Demand` equals 1138.66
   - All Days Worked values are in {4, 5, 6}
   - End of Phase Backlog/Buffer chain is consistent
   - Summary is ≤ 60 words and ≤ 3 sentences
   - Summary mentions both step-down phases

## Important Notes

- The initial condition says `Start of Phase Past Due + Scheduled Demand = 1138.66` at Phase 8. Since `Start of Phase Past Due = max(0, calc_start)`, and `calc_start` is what we derive from prior state, we need `max(0, calc_start) + demand[8] = 1138.66`. If calc_start >= 0, then `calc_start = 1138.66 - demand[8]`. Set `prev_end_backlog` to this value so the Phase 8 computation works correctly.
- Use `openpyxl` to write numeric values (not strings) into Excel cells.
- The demand row label may vary slightly - search for 'Ductwork Demand' or 'HVAC Ductwork' substring.
- If the phase header row detection doesn't find phase 8 easily, inspect the sheet output from Step 1 and adjust the parsing logic accordingly.
- The summary must have EXACTLY 3 lines in the output file, each ending with a newline.

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

Inspect the task files, environment, tests, and expected outputs directly.

# Verifier Contract

Success is judged by the SkillFlow verifier for this task.
Do not bypass, remove, or weaken verifier scripts, tests, fixtures, or expected-output checks.
Run the provided tests or verifier command when practical before finalizing.
Task metadata: author_email=codex@openai.com, author_name=Codex, category=manufacturing-planning, difficulty=medium, tags=[xlsx, operations, capacity-planning, hvac, backlog].
Verifier config: timeout_sec=900.0.