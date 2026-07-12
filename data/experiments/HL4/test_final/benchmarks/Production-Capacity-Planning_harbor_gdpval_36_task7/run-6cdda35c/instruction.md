# Task Instruction

Execute the following steps exactly:

## Step 1: Inspect the CSV
```bash
cat /root/ship_demand.csv
```
Understand the layout: first row has `Week,5,6,...,53` and second row has `Demand,<values>`. Each column after the first gives a (week, demand) pair.

## Step 2: Create the Python script
Create and run `/root/solve.py` with the following logic:

```python
import csv
import openpyxl

# --- Read CSV ---
with open('/root/ship_demand.csv', 'r') as f:
    reader = csv.reader(f)
    header = next(reader)   # ['Week', '5', '6', ..., '53']
    demand_row = next(reader)  # ['Demand', val1, val2, ...]

weeks = [int(h) for h in header[1:]]
demands = [float(d) for d in demand_row[1:]]
week_demand = dict(zip(weeks, demands))

# Verify we have weeks 5..53
assert sorted(week_demand.keys()) == list(range(5, 54)), f"Unexpected weeks: {sorted(week_demand.keys())}"

# --- Compute plan ---
# Initial condition: at Week 5, Calc Start + Scheduled Demand = 1014.51
# So Calc Start for week 5 = 1014.51 - demand[5]
# But actually the instruction says "Start of Week Past Due + Scheduled Demand = 1014.51"
# and Calc Start = prior week End of Week Backlog/Buffer, and for week 5 this IS the initial condition.
# Re-reading: "Initial condition at Week 5: Start of Week Past Due + Scheduled Demand = 1014.51"
# Start of Week Past Due = max(0, prior_eow). Calc Start = prior_eow.
# For week 5, prior_eow is unknown. The initial condition tells us:
#   max(0, prior_eow) + demand[5] = 1014.51
#   => Start of Week Past Due (week 5) = 1014.51 - demand[5]
# Since this should be >= 0 (it's past due), prior_eow = 1014.51 - demand[5] as well.
# Let's just set prior_eow = 1014.51 - demand[5] for week 5.

results = []
first_5_day = None
first_4_day = None

prior_eow = 1014.51 - week_demand[5]

for w in range(5, 54):
    demand = week_demand[w]
    
    # Step 1: Start of Week Past Due (for reporting)
    start_past_due = max(0.0, prior_eow)
    
    # Step 2: Calc Start (signed)
    calc_start = prior_eow
    
    # Step 3: Choose Days Worked
    if start_past_due > 0.01:
        # Try 5 first, then 6
        chosen = None
        for d in [5, 6]:
            if calc_start + demand - 28 * d <= 0:
                chosen = d
                break
        if chosen is None:
            chosen = 6
        days_worked = chosen
    else:
        if demand <= 112:
            days_worked = 4
        else:
            days_worked = 5
    
    # Track first occurrences
    if days_worked == 5 and first_5_day is None:
        first_5_day = w
    if days_worked == 4 and first_4_day is None:
        first_4_day = w
    
    # Step 4
    weekly_capacity = 28 * days_worked
    
    # Step 5
    eow = calc_start + demand - weekly_capacity
    
    # Step 6
    overtime = 10 * max(0, days_worked - 4)
    
    results.append({
        'Week': w,
        'Days Worked': days_worked,
        'Scheduled Demand (Std Hrs)': round(demand, 2),
        'Weekly Capacity (Std Hrs)': weekly_capacity,
        'Start of Week Past Due (Std Hrs)': round(start_past_due, 2),
        'End of Week Backlog/Buffer (Std Hrs)': round(eow, 2),
        'Overtime Hours': overtime
    })
    
    prior_eow = eow

# --- Write Excel ---
wb = openpyxl.Workbook()
ws = wb.active
ws.title = 'Plan'

headers = [
    'Week',
    'Days Worked',
    'Scheduled Demand (Std Hrs)',
    'Weekly Capacity (Std Hrs)',
    'Start of Week Past Due (Std Hrs)',
    'End of Week Backlog/Buffer (Std Hrs)',
    'Overtime Hours'
]
ws.append(headers)

for r in results:
    ws.append([r[h] for h in headers])

wb.save('/root/ship_block_plan.xlsx')
print(f'Wrote {len(results)} rows to /root/ship_block_plan.xlsx')

# --- Write Summary ---
first_5_str = str(first_5_day) if first_5_day is not None else 'N/A'
first_4_str = str(first_4_day) if first_4_day is not None else 'N/A'

# Build summary sentence (<=60 words, <=3 sentences, mention both step-down weeks)
summary_text = (
    f"The crew operated at 6-day weeks to clear initial backlog, "
    f"stepping down to 5-day weeks at Week {first_5_str} "
    f"and to 4-day weeks at Week {first_4_str}. "
    f"Overtime decreased as past-due hours were eliminated, "
    f"stabilizing at a sustainable 4-day schedule."
)

with open('/root/ship_block_summary.txt', 'w') as f:
    f.write(f'First_Week_5_Days: {first_5_str}\n')
    f.write(f'First_Week_4_Days: {first_4_str}\n')
    f.write(f'Summary: {summary_text}\n')

print('Summary:')
with open('/root/ship_block_summary.txt') as f:
    print(f.read())

# Validation
print(f'\nFirst 5-day week: {first_5_str}')
print(f'First 4-day week: {first_4_str}')
print(f'Total rows: {len(results)}')
print(f'Week range: {results[0]["Week"]} to {results[-1]["Week"]}')

# Print first few and last few rows for verification
print('\nFirst 5 rows:')
for r in results[:5]:
    print(r)
print('\nLast 3 rows:')
for r in results[-3:]:
    print(r)
```

## Step 3: Run the script
```bash
pip install openpyxl 2>/dev/null
python3 /root/solve.py
```

## Step 4: Validate outputs
1. Confirm `/root/ship_block_plan.xlsx` exists and has 50 rows (1 header + 49 data).
2. Confirm `/root/ship_block_summary.txt` exists and has exactly 3 lines.
3. Verify Week 5 initial condition: `Start of Week Past Due + Scheduled Demand` should equal `1014.51`.
4. Verify all Days Worked values are in {4, 5, 6}.
5. Verify the summary text is ≤60 words and ≤3 sentences.
6. Check that the summary mentions both step-down week numbers (or N/A).

```bash
python3 -c "
import openpyxl
wb = openpyxl.load_workbook('/root/ship_block_plan.xlsx')
ws = wb['Plan']
print(f'Sheet name: {ws.title}')
print(f'Rows (including header): {ws.max_row}')
print(f'Headers: {[c.value for c in ws[1]]}')
print(f'Row 2 (Week 5): {[c.value for c in ws[2]]}')
# Check initial condition
row2 = [c.value for c in ws[2]]
past_due = row2[4]  # Start of Week Past Due
demand = row2[2]    # Scheduled Demand
print(f'Week 5 check: Past Due ({past_due}) + Demand ({demand}) = {past_due + demand} (should be 1014.51)')
"
```

```bash
cat /root/ship_block_summary.txt
wc -l /root/ship_block_summary.txt
```

If the initial condition check fails or any validation fails, debug and fix before finishing. The summary word count must be verified — if over 60 words, shorten it. If either step-down week is N/A, make sure the summary text handles that correctly.

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
Task metadata: author_email=codex@openai.com, author_name=Codex, category=manufacturing-planning, difficulty=medium, tags=[csv, xlsx, operations, capacity-planning, shipbuilding, backlog].
Verifier config: timeout_sec=900.0.