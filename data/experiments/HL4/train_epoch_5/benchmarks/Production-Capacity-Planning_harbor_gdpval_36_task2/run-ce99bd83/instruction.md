# Task Instruction

## Task: Build CNC Mill Catch-Up Plan

### Step 1: Read the demand data

1. Open `/root/mill_demand_sheet.xlsx`, sheet `Mill`.
2. Find the row labeled `CNC Mill Demand (Hrs)` (search column A or the first column for this label).
3. Extract the demand values for Periods 1 through 52. The periods are likely in columns (one column per period). Print the first 10 demand values to confirm they look reasonable (positive numbers, typically in the range of ~50-150 hours).

### Step 2: Understand the initial condition

The task says: "Initial condition at Period 1: `Start of Period Past Due + Scheduled Demand = 538.08`"

This means for Period 1:
- `Start of Period Past Due (Std Hrs)` = 538.08 - demand[1]
- `Calc Start` for Period 1 = 538.08 - demand[1] (this is the same value since it's positive)

So the initial backlog BEFORE adding Period 1's demand is `538.08 - demand[Period 1]`. This value serves as both the reported `Start of Period Past Due` AND the `Calc Start` for Period 1.

**CRITICAL**: Do NOT set Calc Start = 538.08. The 538.08 already INCLUDES Period 1's scheduled demand. So `Calc Start` (which is the carryover from before this period) = 538.08 - demand[1]. Then the formula `End of Period = Calc Start + Scheduled Demand - Weekly Capacity` will correctly use `(538.08 - demand[1]) + demand[1] - capacity = 538.08 - capacity`.

Verification: For Period 1, `Calc Start + Scheduled Demand` should equal 538.08.

### Step 3: Implement the deterministic policy

For each period t = 1..52:

```python
import math

# Period 1 initialization
calc_start_1 = 538.08 - demand[0]  # demand[0] is Period 1's demand
start_past_due_1 = max(0, calc_start_1)
# For Period 1, calc_start = calc_start_1

# General loop
results = []
prev_end_backlog = calc_start_1  # This is the "prior period End of Period Backlog" for Period 1
# Wait - actually for Period 1, there IS no prior period. Let me re-think.

# Better approach:
# Define: before Period 1 starts, the "virtual prior End of Period Backlog" = 538.08 - demand[0]
# This way:
#   Calc Start for Period 1 = prior End = 538.08 - demand[0]
#   Start Past Due for Period 1 = max(0, prior End) = max(0, 538.08 - demand[0])
#   End of Period 1 = Calc Start + demand[0] - capacity = 538.08 - capacity

prev_end_backlog = 538.08 - demand[0]  # virtual prior-period end backlog

for t in range(52):  # t=0 is Period 1
    sched_demand = demand[t]
    calc_start = prev_end_backlog
    start_past_due = max(0.0, prev_end_backlog)
    
    # Choose Days Worked
    if start_past_due > 0.01:
        # Try 5 first, then 6
        if calc_start + sched_demand - (25 * 5) <= 0:
            days_worked = 5
        elif calc_start + sched_demand - (25 * 6) <= 0:
            days_worked = 6
        else:
            days_worked = 6
    else:
        # No significant past due
        if sched_demand <= 125:
            days_worked = 4
        else:
            days_worked = 5
    
    weekly_capacity = 25 * days_worked
    end_backlog = calc_start + sched_demand - weekly_capacity
    overtime = 10 * max(0, days_worked - 4)
    
    results.append({
        'Period': t + 1,
        'Days Worked': days_worked,
        'Scheduled Demand (Std Hrs)': round(sched_demand, 2),
        'Weekly Capacity (Std Hrs)': weekly_capacity,
        'Start of Period Past Due (Std Hrs)': round(start_past_due, 2),
        'End of Period Backlog/Buffer (Std Hrs)': round(end_backlog, 2),
        'Overtime Hours': overtime
    })
    
    prev_end_backlog = end_backlog
```

### Step 4: Validate Period 1

After computing, verify:
- Period 1: `Calc Start + Scheduled Demand` = `(538.08 - demand[0]) + demand[0]` = 538.08 ✓
- Print Period 1 values to confirm.

### Step 5: Write `/root/mill_catch_up_plan.xlsx`

Use openpyxl to create a workbook with a single sheet named `Plan`. Headers in row 1 must be EXACTLY:
1. `Period`
2. `Days Worked`
3. `Scheduled Demand (Std Hrs)`
4. `Weekly Capacity (Std Hrs)`
5. `Start of Period Past Due (Std Hrs)`
6. `End of Period Backlog/Buffer (Std Hrs)`
7. `Overtime Hours`

Write 52 data rows (rows 2-53), one per period in ascending order.

### Step 6: Determine summary values

- `First_Week_5_Days`: The first period where Days Worked = 5 AND the reason was the "no past due" branch (i.e., `start_past_due <= 0.01` and `demand > 125`). Actually, re-reading the task: it says "step-down period numbers". The step-downs are:
  - First time days drop TO 5 (from 6) — this is the first period with Days Worked = 5.
  - First time days drop TO 4 — this is the first period with Days Worked = 4.

Actually, let me reconsider. The labels are `First_Week_5_Days` and `First_Week_4_Days`. These are simply:
- `First_Week_5_Days`: the first period number where Days Worked = 5
- `First_Week_4_Days`: the first period number where Days Worked = 4

If no such period exists, use `N/A`.

### Step 7: Write `/root/mill_catch_up_summary.txt`

Exactly 3 lines:
```
First_Week_5_Days: <number or N/A>
First_Week_4_Days: <number or N/A>
Summary: <manager-facing summary, ≤60 words, ≤3 sentences, mentioning both step-down periods>
```

No trailing newline after the third line (or a single trailing newline is fine, but no extra blank lines).

### Step 8: Final validation

1. Re-read the generated xlsx to confirm 52 rows, correct headers, and spot-check a few values.
2. Re-read the summary txt to confirm format.
3. Print the first 5 and last 5 rows of the plan for visual inspection.
4. Verify that for Period 1: `Start of Period Past Due + Scheduled Demand` ≈ 538.08 (within 0.01).
5. Verify all Days Worked values are in {4, 5, 6}.
6. Verify End of Period Backlog transitions are consistent (each period's calc_start equals prior period's end_backlog).

### Important warnings from prior failures

- The cross-task failure artifacts show that a common mistake is mishandling the initial condition. The value 538.08 is NOT the initial calc_start — it is `Start of Period Past Due + Scheduled Demand` for Period 1. So `Calc Start` for Period 1 = 538.08 - demand[Period 1].
- Do NOT set calc_start for Period 1 to 538.08. That would double-count the demand.
- Use floating point carefully; avoid unnecessary rounding in intermediate calculations. Only round for display/output.

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