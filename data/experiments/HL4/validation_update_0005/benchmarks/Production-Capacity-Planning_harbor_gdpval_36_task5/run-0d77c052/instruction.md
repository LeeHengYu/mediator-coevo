# Task Instruction

Execute the following Python script to produce both deliverables. Read the demand data from `/root/glass_demand_sheet.xlsx` (sheet `Glass`), compute the weekly plan for weeks 2–50, and write `/root/glass_furnace_plan.xlsx` and `/root/glass_furnace_summary.txt`.

```python
import openpyxl
import os

# ── 1. Read demand data ──────────────────────────────────────────────────────
wb_in = openpyxl.load_workbook('/root/glass_demand_sheet.xlsx', data_only=True)
ws = wb_in['Glass']

# Find the row labeled 'Glass Furnace Demand (Std Hrs)' and the header row with week numbers
header_row = None
demand_row = None
for row in ws.iter_rows(min_row=1, max_row=ws.max_row):
    for cell in row:
        val = cell.value
        if val is not None and isinstance(val, str) and 'Glass Furnace Demand' in val:
            demand_row = cell.row
        # Detect header row: look for numeric week numbers (e.g., 1, 2, ...)
    # Also check if this row has week numbers
    first_vals = [c.value for c in row[:10]]
    if any(v == 1 or v == 2 for v in first_vals if isinstance(v, (int, float))):
        if header_row is None:
            header_row = row[0].row

print(f'Header row: {header_row}, Demand row: {demand_row}')

# Build week->column mapping from the header row
week_col = {}  # week_number -> column_index
for cell in ws[header_row]:
    v = cell.value
    if isinstance(v, (int, float)) and 1 <= v <= 53:
        week_col[int(v)] = cell.column

print(f'Found {len(week_col)} week columns. Sample: {dict(list(week_col.items())[:5])}')

# Read demand values for weeks 2..50
demand = {}
for wk in range(2, 51):
    if wk in week_col:
        col = week_col[wk]
        val = ws.cell(row=demand_row, column=col).value
        demand[wk] = float(val) if val is not None else 0.0
    else:
        demand[wk] = 0.0

print(f'Demand week 2: {demand[2]}')
print(f'Demand sample (2-6): {[demand[w] for w in range(2,7)]}')

# ── 2. Compute the plan ─────────────────────────────────────────────────────
# Initial condition: Start of Week Past Due + Scheduled Demand = 910.80 at Week 2
# This means: Calc Start (prior week end backlog) + demand[2] = 910.80
# So the prior-week End of Week Backlog/Buffer = 910.80 - demand[2]

initial_calc_start = 910.80 - demand[2]
print(f'Initial Calc Start for Week 2: {initial_calc_start}')

rows_out = []
prior_end_backlog = initial_calc_start  # This feeds into Week 2

first_week_5 = None
first_week_4 = None

for wk in range(2, 51):
    sched_demand = demand[wk]
    
    # Step 1: Start of Week Past Due (for reporting)
    start_past_due = max(0.0, prior_end_backlog)
    
    # Step 2: Calc Start (signed)
    calc_start = prior_end_backlog
    
    # Step 3: Choose Days Worked
    if start_past_due > 0.01:
        # Past due: choose smallest in {5, 6} that clears backlog, else 6
        chosen = 6  # default if neither works
        for d in [5, 6]:
            if calc_start + sched_demand - (22 * d) <= 0:
                chosen = d
                break
    else:
        # No past due
        if sched_demand <= 110:
            chosen = 4
        else:
            chosen = 5
    
    days_worked = chosen
    weekly_capacity = 22 * days_worked
    end_backlog = calc_start + sched_demand - weekly_capacity
    overtime = 10 * max(0, days_worked - 4)
    
    # Track first occurrences for step-down
    if days_worked == 5 and first_week_5 is None:
        first_week_5 = wk
    if days_worked == 4 and first_week_4 is None:
        first_week_4 = wk
    
    rows_out.append({
        'Week': wk,
        'Days Worked': days_worked,
        'Scheduled Demand (Std Hrs)': round(sched_demand, 2),
        'Weekly Capacity (Std Hrs)': round(weekly_capacity, 2),
        'Start of Week Past Due (Std Hrs)': round(start_past_due, 2),
        'End of Week Backlog/Buffer (Std Hrs)': round(end_backlog, 2),
        'Overtime Hours': round(overtime, 2),
    })
    
    prior_end_backlog = end_backlog

print(f'Computed {len(rows_out)} rows')
print(f'First 3 rows: {rows_out[:3]}')
print(f'Last row: {rows_out[-1]}')
print(f'First Week 5 Days: {first_week_5}')
print(f'First Week 4 Days: {first_week_4}')

# ── 3. Write the Excel workbook ─────────────────────────────────────────────
wb_out = openpyxl.Workbook()
ws_out = wb_out.active
ws_out.title = 'Plan'

headers = [
    'Week',
    'Days Worked',
    'Scheduled Demand (Std Hrs)',
    'Weekly Capacity (Std Hrs)',
    'Start of Week Past Due (Std Hrs)',
    'End of Week Backlog/Buffer (Std Hrs)',
    'Overtime Hours',
]
ws_out.append(headers)

for r in rows_out:
    ws_out.append([r[h] for h in headers])

wb_out.save('/root/glass_furnace_plan.xlsx')
print('Saved /root/glass_furnace_plan.xlsx')

# ── 4. Write the summary file ───────────────────────────────────────────────
fw5 = str(first_week_5) if first_week_5 is not None else 'N/A'
fw4 = str(first_week_4) if first_week_4 is not None else 'N/A'

# Build a concise manager-facing summary (<=60 words, <=3 sentences, mentioning both step-down weeks)
if first_week_5 is not None and first_week_4 is not None:
    summary_text = (f'The furnace crew operates 6-day weeks to clear initial backlog, '
                    f'stepping down to 5 days in Week {first_week_5} and to 4 days in Week {first_week_4}. '
                    f'This plan eliminates past-due hours while minimizing overtime.')
elif first_week_5 is not None:
    summary_text = (f'The crew steps down from 6 to 5 days in Week {first_week_5}; '
                    f'a step-down to 4 days is N/A. '
                    f'Sustained 5-day weeks are needed to meet demand throughout the horizon.')
elif first_week_4 is not None:
    summary_text = (f'The crew steps down to 5 days at N/A and to 4 days in Week {first_week_4}. '
                    f'Demand is low enough to allow reduced schedules early.')
else:
    summary_text = ('Step-down to 5 days is N/A and to 4 days is N/A. '
                    'The crew maintains 6-day weeks throughout to address persistent backlog.')

# Verify word count
word_count = len(summary_text.split())
print(f'Summary word count: {word_count}')
assert word_count <= 60, f'Summary too long: {word_count} words'

with open('/root/glass_furnace_summary.txt', 'w') as f:
    f.write(f'First_Week_5_Days: {fw5}\n')
    f.write(f'First_Week_4_Days: {fw4}\n')
    f.write(f'Summary: {summary_text}\n')

print('Saved /root/glass_furnace_summary.txt')

# ── 5. Verification ─────────────────────────────────────────────────────────
# Re-read and verify
wb_v = openpyxl.load_workbook('/root/glass_furnace_plan.xlsx')
ws_v = wb_v['Plan']
print(f'Plan sheet row count (including header): {ws_v.max_row}')
assert ws_v.max_row == 50, f'Expected 50 rows (1 header + 49 data), got {ws_v.max_row}'
print(f'Headers: {[ws_v.cell(row=1, column=c).value for c in range(1, 8)]}')
print(f'Row 2 (Week 2): {[ws_v.cell(row=2, column=c).value for c in range(1, 8)]}')
print(f'Row 50 (Week 50): {[ws_v.cell(row=50, column=c).value for c in range(1, 8)]}')

with open('/root/glass_furnace_summary.txt', 'r') as f:
    content = f.read()
print('Summary file content:')
print(content)
print('Lines:', len(content.strip().split('\n')))
assert len(content.strip().split('\n')) == 3, 'Summary must have exactly 3 lines'

print('\n=== ALL CHECKS PASSED ===')
```

After running the script, verify:
1. `/root/glass_furnace_plan.xlsx` exists with sheet `Plan`, 7 columns with exact headers, 49 data rows (weeks 2–50).
2. `/root/glass_furnace_summary.txt` has exactly 3 lines with the correct format.
3. The Week 2 row should have `Start of Week Past Due` equal to `max(0, 910.80 - demand[2])` and the initial condition is satisfied.
4. All `Days Worked` values are in {4, 5, 6}.
5. The summary mentions both step-down week numbers (or N/A).

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
Task metadata: author_email=codex@openai.com, author_name=Codex, category=manufacturing-planning, difficulty=medium, tags=[xlsx, operations, capacity-planning, glass, backlog].
Verifier config: timeout_sec=900.0.