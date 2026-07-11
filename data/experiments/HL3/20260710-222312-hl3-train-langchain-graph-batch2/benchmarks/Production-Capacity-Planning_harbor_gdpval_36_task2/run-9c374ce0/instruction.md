# Task Instruction

Execute the following steps in order:

## 1. Read the demand data

Open `/root/mill_demand_sheet.xlsx`, sheet `Mill`. Locate the row labeled `CNC Mill Demand (Hrs)` (check column A for that label). Extract the 52 numeric demand values for Periods 1–52 from that row. Print them to confirm.

## 2. Compute the schedule using Python

Use Python with `openpyxl` to compute and write results. Follow this logic exactly:

```python
import openpyxl

# Load demand
wb_in = openpyxl.load_workbook('/root/mill_demand_sheet.xlsx', data_only=True)
ws_in = wb_in['Mill']

# Find the row with 'CNC Mill Demand (Hrs)' in column A
demand_row = None
for row in ws_in.iter_rows(min_col=1, max_col=1):
    cell = row[0]
    if cell.value and 'CNC Mill Demand' in str(cell.value):
        demand_row = cell.row
        break

assert demand_row is not None, 'Could not find CNC Mill Demand row'

# Extract 52 demand values (columns B onward, or wherever periods start)
# First, find which columns hold periods 1-52 by checking the header row
# Usually row above demand_row or a known header row has period numbers
# Inspect and adapt as needed
demands = []
for col in range(2, 54):  # columns B through BA (52 columns)
    val = ws_in.cell(row=demand_row, column=col).value
    if val is None:
        val = 0.0
    demands.append(float(val))

assert len(demands) == 52
print('Demands:', demands)

# Initial condition: at Period 1, Start of Period Past Due + Scheduled Demand = 538.08
# This means calc_start for period 1 = 538.08 - demands[0]
# Because: calc_start + demand = 538.08 => Start Past Due = max(0, calc_start)
# And Start Past Due + demand = 538.08 when calc_start >= 0
# So calc_start = 538.08 - demands[0]

calc_start_initial = 538.08 - demands[0]

results = []
prev_end = calc_start_initial  # This is the 'prior period End of Period Backlog/Buffer' equivalent

for i in range(52):
    demand = demands[i]
    period = i + 1
    
    if i == 0:
        calc_start = calc_start_initial
    else:
        calc_start = prev_end
    
    start_past_due = max(0.0, calc_start)
    
    # Choose Days Worked
    if start_past_due > 0.01:
        # Try 5 first, then 6
        chosen = None
        for d in [5, 6]:
            if calc_start + demand - (25 * d) <= 0:
                chosen = d
                break
        if chosen is None:
            chosen = 6
        days_worked = chosen
    else:
        # No significant past due
        if demand <= 125:
            days_worked = 4
        else:
            days_worked = 5
    
    weekly_capacity = 25 * days_worked
    end_backlog = calc_start + demand - weekly_capacity
    overtime = 10 * max(0, days_worked - 4)
    
    results.append({
        'period': period,
        'days_worked': days_worked,
        'demand': demand,
        'capacity': weekly_capacity,
        'start_past_due': round(start_past_due, 4),
        'end_backlog': round(end_backlog, 4),
        'overtime': overtime
    })
    
    prev_end = end_backlog

# Print first few and last few for verification
for r in results[:5]:
    print(r)
print('...')
for r in results[-3:]:
    print(r)
```

Run this and verify the output looks correct. In particular, check Period 1: `start_past_due = max(0, 538.08 - demand[0])`, and `start_past_due + demand[0]` should equal 538.08.

## 3. Write the Excel workbook

```python
wb_out = openpyxl.Workbook()
ws_out = wb_out.active
ws_out.title = 'Plan'

headers = [
    'Period',
    'Days Worked',
    'Scheduled Demand (Std Hrs)',
    'Weekly Capacity (Std Hrs)',
    'Start of Period Past Due (Std Hrs)',
    'End of Period Backlog/Buffer (Std Hrs)',
    'Overtime Hours'
]
for col_idx, h in enumerate(headers, 1):
    ws_out.cell(row=1, column=col_idx, value=h)

for row_idx, r in enumerate(results, 2):
    ws_out.cell(row=row_idx, column=1, value=r['period'])
    ws_out.cell(row=row_idx, column=2, value=r['days_worked'])
    ws_out.cell(row=row_idx, column=3, value=r['demand'])
    ws_out.cell(row=row_idx, column=4, value=r['capacity'])
    ws_out.cell(row=row_idx, column=5, value=r['start_past_due'])
    ws_out.cell(row=row_idx, column=6, value=r['end_backlog'])
    ws_out.cell(row=row_idx, column=7, value=r['overtime'])

wb_out.save('/root/mill_catch_up_plan.xlsx')
print('Excel saved.')
```

Ensure all cell values are numeric (int or float), NOT strings.

## 4. Write the summary file

Determine:
- `First_Week_5_Days`: the first period where `days_worked == 5` AND the previous period had `days_worked == 6`. This is the step-down from 6 to 5. If no period starts at 6 days, then find the first period where days_worked is 5 (the initial step-down from the catch-up phase). More precisely: find the first period where days_worked drops from a higher value to 5 (i.e., the prior period had days_worked > 5). If the very first period has days_worked == 5 and there's no prior 6-day period, use N/A.
- `First_Week_4_Days`: the first period where `days_worked == 4` (step-down to normal).

Actually, re-reading the task: it says "step-down period numbers". The step-downs are:
- The first period where days_worked becomes 5 after having been 6 (step-down from 6→5)
- The first period where days_worked becomes 4 (step-down from 5→4)

But the summary keys are `First_Week_5_Days` and `First_Week_4_Days`. So:
- `First_Week_5_Days` = first period number where days_worked == 5 (and it represents a step-down, i.e., prior period was 6). If the schedule never has 6-day weeks, this might just be the first 5-day period. Use the simplest interpretation: the first period where days_worked is exactly 5.
- `First_Week_4_Days` = first period where days_worked is exactly 4.

Compute these:

```python
first_5 = None
first_4 = None
for r in results:
    if first_5 is None and r['days_worked'] == 5:
        # Check if this is a step-down (prior was 6)
        idx = r['period'] - 1
        if idx == 0 or results[idx-1]['days_worked'] == 6:
            first_5 = r['period']
    if first_4 is None and r['days_worked'] == 4:
        first_4 = r['period']

# If first_5 was never set but there are 5-day periods, just use the first one
if first_5 is None:
    for r in results:
        if r['days_worked'] == 5:
            first_5 = r['period']
            break

first_5_str = str(first_5) if first_5 else 'N/A'
first_4_str = str(first_4) if first_4 else 'N/A'

# Build summary (≤60 words, ≤3 sentences, mention both step-down periods)
summary = f'The mill catch-up plan clears the initial backlog by scheduling 6-day weeks, stepping down to 5-day weeks at Period {first_5_str} and to 4-day weeks at Period {first_4_str}. Overtime decreases as backlog is eliminated. The plan covers all 52 periods with demand met each week.'

# Verify word count
word_count = len(summary.split())
print(f'Summary word count: {word_count}')
assert word_count <= 60, f'Summary too long: {word_count} words'

with open('/root/mill_catch_up_summary.txt', 'w') as f:
    f.write(f'First_Week_5_Days: {first_5_str}\n')
    f.write(f'First_Week_4_Days: {first_4_str}\n')
    f.write(f'Summary: {summary}\n')

print('Summary file saved.')
```

## 5. Verify outputs

1. Re-open `/root/mill_catch_up_plan.xlsx` and confirm:
   - Sheet name is exactly `Plan`
   - Row 1 has exactly the 7 required headers
   - There are exactly 52 data rows (rows 2-53)
   - All values are numeric (not strings)
   - Period 1 values: `start_past_due + demand == 538.08`
   - No gaps or duplicates in Period column

2. Re-read `/root/mill_catch_up_summary.txt` and confirm:
   - Exactly 3 lines
   - Line 1 starts with `First_Week_5_Days:`
   - Line 2 starts with `First_Week_4_Days:`
   - Line 3 starts with `Summary:`
   - Summary ≤ 60 words and ≤ 3 sentences

## Important notes

- The demand data columns may not start at column B. Inspect the sheet structure first and adapt column indices accordingly.
- The `round()` on start_past_due and end_backlog is for display; use 4 decimal places to avoid floating point drift but keep precision.
- Store all Excel cell values as Python `int` or `float`, never as strings.
- If the demand row label doesn't contain 'CNC Mill Demand' exactly, try partial matching or inspect all row labels to find the correct one.

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
Task metadata: author_email=codex@openai.com, author_name=Codex, category=manufacturing-planning, difficulty=medium, tags=[xlsx, operations, capacity-planning, cnc, backlog].
Verifier config: timeout_sec=900.0.