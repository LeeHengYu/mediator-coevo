# Task Instruction

Execute the following steps in order:

## Step 1 – Inspect the input workbook

```bash
cd /root
python3 -c "
import openpyxl
wb = openpyxl.load_workbook('dye_demand_sheet.xlsx', data_only=True)
for name in wb.sheetnames:
    ws = wb[name]
    print(f'=== Sheet: {name} ===')
    for row in ws.iter_rows(min_row=1, max_row=min(ws.max_row, 10), values_only=False):
        print([(c.coordinate, c.value) for c in row])
    print()
"
```

Note the exact row labels and column layout for both `Dye` and `Adjust` sheets. Identify which row contains `Dye Demand (Std Hrs)` and which contains `Demand Adjustment (Std Hrs)`, and how weeks 3–51 map to columns.

## Step 2 – Build the plan and write both deliverables

Write a single Python script `/root/build_plan.py` that does the following:

### 2a – Read demand data
- Open `dye_demand_sheet.xlsx`.
- From the `Dye` sheet, find the row whose first-column value matches `Dye Demand (Std Hrs)` (strip/compare case-insensitively if needed). Read weekly values for weeks 3–51.
- From the `Adjust` sheet, find the row whose first-column value matches `Demand Adjustment (Std Hrs)`. Read weekly values for weeks 3–51.
- Compute `Scheduled Demand = Dye Demand + Demand Adjustment` for each week. Treat missing/None values as 0.
- Determine how weeks map to columns: the header row likely has week numbers; match them to columns.

### 2b – Simulate week-by-week (deterministic policy)

Initialize: for Week 3, `Calc Start + Scheduled Demand = 598.24`, so `Calc Start = 598.24 - Scheduled_Demand_Week3`.

Wait – re-read the instruction: "Initial condition at Week 3: Start of Week Past Due + Scheduled Demand = 598.24". This means `Start of Week Past Due` for week 3 plus `Scheduled Demand` for week 3 equals 598.24. Since week 3 is the first week, `Start of Week Past Due = max(0, prior_End_of_Week_Backlog)`. There is no prior week, so we derive: `Start of Week Past Due (week 3) = 598.24 - Scheduled_Demand(week 3)`. And `Calc Start (week 3) = Start of Week Past Due (week 3)` (since it's the first week and there's no negative carryover yet).

Actually, let me be more precise. The initial condition says `Start of Week Past Due + Scheduled Demand = 598.24` at Week 3. `Start of Week Past Due = max(0, prior_Backlog)`. For week 3 there is no prior week, so `prior_Backlog` is some initial value. Let `prior_Backlog = 598.24 - Scheduled_Demand(week 3)`. Then:
- `Start of Week Past Due (week 3) = max(0, prior_Backlog)`
- `Calc Start (week 3) = prior_Backlog` (the signed value)

But the condition says `Start of Week Past Due + Scheduled Demand = 598.24`. If `prior_Backlog >= 0`, then `Start of Week Past Due = prior_Backlog`, so `prior_Backlog = 598.24 - Scheduled_Demand(week 3)`. If that value is negative, the condition couldn't hold with `max(0,...)`. So assume `prior_Backlog = 598.24 - Scheduled_Demand(week 3)` and it should be >= 0.

For each week w = 3..51:
1. `start_past_due = max(0, prior_end_backlog)` — for display
2. `calc_start = prior_end_backlog` — signed value for calculation
3. `scheduled_demand = effective_demand[w]`
4. Choose `days_worked`:
   - If `start_past_due > 0.01`:
     - Try 5: if `calc_start + scheduled_demand - 18*5 <= 0`, pick 5
     - Else try 6: if `calc_start + scheduled_demand - 18*6 <= 0`, pick 6
     - Else pick 6
   - Else (start_past_due <= 0.01):
     - If `scheduled_demand <= 72`: pick 4
     - Else: pick 5
5. `weekly_capacity = 18 * days_worked`
6. `end_backlog = calc_start + scheduled_demand - weekly_capacity`
7. `overtime = 10 * max(0, days_worked - 4)`
8. Store row; set `prior_end_backlog = end_backlog` for next week.

### 2c – Write `/root/dye_catch_up_plan.xlsx`

Using openpyxl, create a workbook with a single sheet named exactly `Plan`. Row 1 headers (exactly):
```
Week | Days Worked | Scheduled Demand (Std Hrs) | Weekly Capacity (Std Hrs) | Start of Week Past Due (Std Hrs) | End of Week Backlog/Buffer (Std Hrs) | Overtime Hours
```
Then 49 data rows (weeks 3–51), ascending, no gaps, no duplicates. Round numeric values to 2 decimal places.

### 2d – Write `/root/dye_catch_up_summary.txt`

Scan the results to find:
- `First_Week_5_Days`: the first week where `Days Worked == 5` (not 6). If none, `N/A`.
- `First_Week_4_Days`: the first week where `Days Worked == 4`. If none, `N/A`.

Write exactly 3 lines:
```
First_Week_5_Days: <value>
First_Week_4_Days: <value>
Summary: <≤60 words, ≤3 sentences mentioning both step-down week numbers or N/A>
```

No trailing newline after the third line is fine, but ensure no extra lines.

## Step 3 – Run the script

```bash
cd /root && python3 build_plan.py
```

## Step 4 – Validate outputs

### 4a – Validate the Excel file
```python
import openpyxl
wb = openpyxl.load_workbook('/root/dye_catch_up_plan.xlsx')
assert wb.sheetnames == ['Plan'], f'Sheet names: {wb.sheetnames}'
ws = wb['Plan']
headers = [c.value for c in ws[1]]
expected = ['Week','Days Worked','Scheduled Demand (Std Hrs)','Weekly Capacity (Std Hrs)','Start of Week Past Due (Std Hrs)','End of Week Backlog/Buffer (Std Hrs)','Overtime Hours']
assert headers == expected, f'Headers mismatch: {headers}'
weeks = [ws.cell(row=r, column=1).value for r in range(2, 51)]
assert weeks == list(range(3, 52)), f'Week column: {weeks}'
for r in range(2, 51):
    dw = ws.cell(row=r, column=2).value
    assert dw in (4,5,6), f'Row {r} Days Worked={dw}'
print('Excel validation passed')
```

### 4b – Validate the summary file
```python
with open('/root/dye_catch_up_summary.txt') as f:
    lines = f.read().strip().split('\n')
assert len(lines) == 3, f'Expected 3 lines, got {len(lines)}'
assert lines[0].startswith('First_Week_5_Days:')
assert lines[1].startswith('First_Week_4_Days:')
assert lines[2].startswith('Summary:')
summary_text = lines[2].split(':', 1)[1].strip()
word_count = len(summary_text.split())
assert word_count <= 60, f'Summary has {word_count} words'
print('Summary validation passed')
```

### 4c – Spot-check Week 3 arithmetic
Print the Week 3 row and verify:
- `Start of Week Past Due + Scheduled Demand == 598.24` (within rounding)
- `End of Week Backlog/Buffer == Calc Start + Scheduled Demand - Weekly Capacity`
- Days Worked follows the policy correctly given start_past_due value

If any validation fails, debug and fix before marking complete.

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
Task metadata: author_email=codex@openai.com, author_name=Codex, category=manufacturing-planning, difficulty=medium, tags=[xlsx, operations, capacity-planning, textile, backlog].
Verifier config: timeout_sec=900.0.