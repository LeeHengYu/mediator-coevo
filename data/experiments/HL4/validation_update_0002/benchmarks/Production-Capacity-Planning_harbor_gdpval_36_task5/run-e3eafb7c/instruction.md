# Task Instruction

Execute the following steps in order to produce `/root/glass_furnace_plan.xlsx` and `/root/glass_furnace_summary.txt`.

---

### Step 1 – Read the demand data

```python
import openpyxl, math

wb_in = openpyxl.load_workbook('/root/glass_demand_sheet.xlsx', data_only=True)
ws = wb_in['Glass']
```

Inspect the sheet to find the row whose first non-empty cell contains the text `Glass Furnace Demand (Std Hrs)` (case-insensitive substring match on "glass furnace demand"). Print the first 5 cells of that row and the header row (row 1 or whichever row contains the week numbers) so you can map week numbers to column indices.

Build a Python dict `demand = {week_number: value}` for weeks 2–50 from that row. Print the dict to verify. All demand values should be non-negative floats.

### Step 2 – Run the deterministic simulation

Implement exactly this logic in Python (no spreadsheet formulas):

```
results = []  # list of dicts, one per week

# Week 2 initial condition
# "Start of Week Past Due + Scheduled Demand = 910.80" for Week 2
# This means Calc Start (the signed carryover entering Week 2) satisfies:
#   Calc_Start_W2 + demand[2] = 910.80
# So: Calc_Start_W2 = 910.80 - demand[2]

calc_start = 910.80 - demand[2]

for week in range(2, 51):
    d = demand[week]
    past_due = max(0.0, calc_start)          # reported only

    # Choose Days Worked
    if past_due > 0.01:
        # Try 5 first, then 6; if neither clears, use 6
        if calc_start + d - 22*5 <= 0:
            days = 5
        elif calc_start + d - 22*6 <= 0:
            days = 6
        else:
            days = 6
    else:
        days = 4 if d <= 110 else 5

    capacity = 22 * days
    end_backlog = calc_start + d - capacity
    overtime = 10 * max(0, days - 4)

    results.append({
        'Week': week,
        'Days Worked': days,
        'Scheduled Demand (Std Hrs)': round(d, 2),
        'Weekly Capacity (Std Hrs)': capacity,
        'Start of Week Past Due (Std Hrs)': round(past_due, 2),
        'End of Week Backlog/Buffer (Std Hrs)': round(end_backlog, 2),
        'Overtime Hours': overtime,
    })

    # Propagate signed carryover
    calc_start = end_backlog
```

Print the first 5 rows and last 5 rows of `results` to verify.

### Step 3 – Detect step-down weeks

```python
first_5 = None
first_4 = None
for r in results:
    if first_5 is None and r['Days Worked'] == 5:
        first_5 = r['Week']
    if first_4 is None and r['Days Worked'] == 4:
        first_4 = r['Week']
```

But be careful: "First_Week_5_Days" means the first week where days worked drops to 5 (i.e., the step-down from 6 to 5). "First_Week_4_Days" means the first week where days worked drops to 4 (step-down from 5 to 4). So track these as the first occurrence of 5-day and 4-day weeks respectively in the sequence. Print both values.

### Step 4 – Write the Excel workbook

```python
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

for r in results:
    ws_out.append([r[h] for h in headers])

wb_out.save('/root/glass_furnace_plan.xlsx')
```

Verify: re-open the file, confirm sheet name is `Plan`, row count is 50 (1 header + 49 data), and print headers and first 3 data rows.

### Step 5 – Write the summary text file

Format the two step-down week numbers (use `N/A` if the corresponding days-worked value never appears). Write the summary as ≤ 3 sentences and ≤ 60 words mentioning both step-down week numbers.

```python
w5 = str(first_5) if first_5 is not None else 'N/A'
w4 = str(first_4) if first_4 is not None else 'N/A'

summary = (f'The furnace crew starts at 6 days/week and steps down to 5 days '
           f'in Week {w5}, then to 4 days in Week {w4}. '
           f'Overtime is eliminated once the 4-day schedule begins. '
           f'This plan clears the initial backlog while minimizing excess capacity.')

# Adjust summary if word count > 60 or sentence count > 3
lines = [
    f'First_Week_5_Days: {w5}',
    f'First_Week_4_Days: {w4}',
    f'Summary: {summary}',
]

with open('/root/glass_furnace_summary.txt', 'w') as f:
    f.write('\n'.join(lines) + '\n')
```

Verify: read the file back and print it. Confirm exactly 3 lines (ignoring trailing newline), word count of the summary portion is ≤ 60, and both step-down weeks are mentioned.

### Step 6 – Final validation

1. Confirm `/root/glass_furnace_plan.xlsx` exists and has sheet `Plan` with 49 data rows.
2. Confirm `/root/glass_furnace_summary.txt` exists with exactly 3 lines in the required format.
3. Spot-check Week 2: `Start of Week Past Due` should equal `max(0, 910.80 - demand[2])`, and `End of Week Backlog/Buffer` should equal `(910.80 - demand[2]) + demand[2] - capacity = 910.80 - capacity`.
4. Print the total overtime hours across all weeks.

**Important edge cases:**
- The demand row label may span merged cells or have slight formatting differences. Search flexibly.
- Week columns may not start at column B. Inspect the header row to find where week numbers appear.
- If demand values are stored as strings, convert to float.
- Ensure all `Days Worked` values are Python ints (not floats) before writing to Excel.
- The summary text must be ≤ 60 words and ≤ 3 sentences. Count words after writing and trim if needed.

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