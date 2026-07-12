# Task Instruction

Execute the following steps in order:

## Step 1: Inspect the input data

Read `/root/chemical_demand.json` and print the first 10 entries and the last 5 entries so we understand the structure. Also print the total number of entries.

## Step 2: Build the schedule with a Python script

Create and run a Python script `/root/build_schedule.py` that does the following:

```python
import json
import openpyxl

# 1. Load data
with open('/root/chemical_demand.json', 'r') as f:
    data = json.load(f)

# 2. Extract first valid (non-null) demand for each week in range 10..58
demand_map = {}
for entry in data:
    w = entry['week']
    if w < 10 or w > 58:
        continue
    if w in demand_map:
        continue  # already have first valid occurrence
    d = entry['data']['demand_per_week']
    if d is not None:
        demand_map[w] = d

# Verify we have all phases 10..58
for w in range(10, 59):
    if w not in demand_map:
        print(f"WARNING: No valid demand for week {w}, defaulting to 0")
        demand_map[w] = 0.0

# 3. Compute schedule
# Initial condition: Calc Start for phase 10 = 1453.06
results = []
calc_start = 1453.06  # This is the initial condition for phase 10

for phase in range(10, 59):
    scheduled_demand = demand_map[phase]
    
    # Start of Phase Past Due (for reporting) = max(0, calc_start)
    start_past_due = max(0.0, calc_start)
    
    # Choose Days Worked
    if start_past_due > 0.01:
        # Try 5 first, then 6
        chosen = None
        for d in [5, 6]:
            if calc_start + scheduled_demand - (40 * d) <= 0:
                chosen = d
                break
        if chosen is None:
            chosen = 6
        days_worked = chosen
    else:
        # start_past_due <= 0.01
        if scheduled_demand <= 160:
            days_worked = 4
        else:
            days_worked = 5
    
    weekly_capacity = 40 * days_worked
    end_backlog = calc_start + scheduled_demand - weekly_capacity
    overtime = 10 * max(0, days_worked - 4)
    
    results.append({
        'Phase': phase,
        'Days Worked': days_worked,
        'Scheduled Demand (Std Hrs)': scheduled_demand,
        'Weekly Capacity (Std Hrs)': weekly_capacity,
        'Start of Phase Past Due (Std Hrs)': start_past_due,
        'End of Phase Backlog/Buffer (Std Hrs)': end_backlog,
        'Overtime Hours': overtime,
    })
    
    # Next phase calc_start = this phase's end_backlog (signed value)
    calc_start = end_backlog

# 4. Create Excel workbook
wb = openpyxl.Workbook()
ws = wb.active
ws.title = 'Plan'

headers = [
    'Phase',
    'Days Worked',
    'Scheduled Demand (Std Hrs)',
    'Weekly Capacity (Std Hrs)',
    'Start of Phase Past Due (Std Hrs)',
    'End of Phase Backlog/Buffer (Std Hrs)',
    'Overtime Hours',
]
ws.append(headers)

for row in results:
    ws.append([row[h] for h in headers])

wb.save('/root/chemical_schedule_plan.xlsx')
print('Workbook saved.')

# 5. Determine summary values
first_5 = None
first_4 = None
for row in results:
    if first_5 is None and row['Days Worked'] == 5:
        first_5 = row['Phase']
    if first_4 is None and row['Days Worked'] == 4:
        first_4 = row['Phase']

first_5_str = str(first_5) if first_5 is not None else 'N/A'
first_4_str = str(first_4) if first_4 is not None else 'N/A'

# Build summary sentence (<=60 words, <=3 sentences, mention both step-down phases)
if first_5 is not None and first_4 is not None:
    summary = (f"The crew operated at 6-day weeks to clear the initial backlog, "
               f"stepping down to 5 days at Phase {first_5} and to 4 days at Phase {first_4}. "
               f"Overtime decreased as the backlog was eliminated.")
elif first_5 is not None:
    summary = (f"The crew stepped down to 5 days at Phase {first_5} but never reached 4-day weeks (N/A). "
               f"Sustained overtime was required throughout the planning horizon.")
elif first_4 is not None:
    summary = (f"The crew stepped down to 4 days at Phase {first_4} but a 5-day step-down was N/A. "
               f"Backlog was cleared efficiently.")
else:
    summary = (f"The crew never stepped down from 6-day weeks; both 5-day (N/A) and 4-day (N/A) milestones were not reached. "
               f"Persistent backlog required maximum overtime throughout.")

with open('/root/chemical_schedule_summary.txt', 'w') as f:
    f.write(f'First_Week_5_Days: {first_5_str}\n')
    f.write(f'First_Week_4_Days: {first_4_str}\n')
    f.write(f'Summary: {summary}\n')

print('Summary saved.')
print(f'First_Week_5_Days: {first_5_str}')
print(f'First_Week_4_Days: {first_4_str}')
print(f'Summary: {summary}')

# Print first few and last few rows for verification
print('\n--- First 5 rows ---')
for r in results[:5]:
    print(r)
print('\n--- Last 5 rows ---')
for r in results[-5:]:
    print(r)
print(f'\nTotal rows: {len(results)}')
```

## Step 3: Verify the outputs

1. Confirm `/root/chemical_schedule_plan.xlsx` exists and has a sheet named `Plan` with 50 rows (1 header + 49 data rows).
2. Confirm `/root/chemical_schedule_summary.txt` exists and has exactly 3 lines.
3. Read and print the summary file contents.
4. Open the workbook and print the header row and the first 3 data rows and last 3 data rows to verify correctness.
5. Verify Phase 10's Start of Phase Past Due is 1453.06 (since initial calc_start is 1453.06 and max(0, 1453.06) = 1453.06).
6. Check that the summary word count is <= 60 words and <= 3 sentences.

## Important Notes
- The initial condition `1453.06` is the `Calc Start` for Phase 10 (i.e., `Start of Phase Past Due + Scheduled Demand = 1453.06` means the initial backlog/buffer carried into Phase 10 is 1453.06).
- Use the **signed** `End of Phase Backlog/Buffer` as the next phase's `Calc Start` (it can go negative, representing a buffer).
- `Start of Phase Past Due` is `max(0, Calc Start)` for reporting only.
- Days Worked must be integers in {4, 5, 6} only.
- When past due > 0.01, try 5 first, then 6 (smallest in {5,6} that clears the backlog); if neither works, use 6.
- When not past due, use 4 if demand <= 160, else 5.

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
Task metadata: author_email=codex@openai.com, author_name=Codex, category=manufacturing-planning, difficulty=medium, tags=[json, xlsx, operations, capacity-planning, chemical, backlog].
Verifier config: timeout_sec=900.0.