# Task Instruction

Execute the following steps to produce `/root/glass_furnace_plan.xlsx` and `/root/glass_furnace_summary.txt`.

## Step 1: Read demand data

Open `/root/glass_demand_sheet.xlsx`, sheet `Glass`. Inspect the sheet to find the row labeled `Glass Furnace Demand (Std Hrs)`. Extract the scheduled demand values for Weeks 2 through 50 inclusive. The week numbers should be in the column headers (or a header row). Print the demand values to confirm you have exactly 49 values.

## Step 2: Implement the deterministic policy

Use Python with `openpyxl`. Implement the following logic exactly:

```python
import openpyxl
from openpyxl import Workbook

# After reading demands into a dict: demand[week] -> float

results = []

# Week 2 initial condition:
# Calc Start + Scheduled Demand = 910.80 at Week 2
# So Calc Start for Week 2 = 910.80 - demand[2]
# (This means prior week End of Week Backlog/Buffer = 910.80 - demand[2])

prev_end_backlog = 910.80 - demand[2]  # This is the implicit prior-week carryover

# Wait -- re-read the instruction: "Start of Week Past Due + Scheduled Demand = 910.80"
# Start of Week Past Due = max(0, prior week End of Week Backlog/Buffer)
# So for Week 2: Start of Week Past Due + Scheduled Demand = 910.80
# => Start of Week Past Due = 910.80 - demand[2]
# Since Start of Week Past Due = max(0, prior_end_backlog), and this value should be >= 0,
# we need prior_end_backlog such that max(0, prior_end_backlog) = 910.80 - demand[2]
# The simplest consistent interpretation: prior_end_backlog = 910.80 - demand[2]
# AND Calc Start (the signed carryover) = prior_end_backlog = 910.80 - demand[2]
# So effectively Calc Start + demand[2] = 910.80 for Week 2.

# Actually, let me re-read more carefully:
# "Initial condition at Week 2: Start of Week Past Due + Scheduled Demand = 910.80"
# Start of Week Past Due for Week 2 = max(0, prior_end_backlog)
# Calc Start for Week 2 = prior_end_backlog (signed)
# Since this is the initial condition and past due should be positive:
#   Start of Week Past Due = 910.80 - demand[2]
#   prior_end_backlog = 910.80 - demand[2]  (assuming it's positive)

prev_end_backlog = 910.80 - demand[2]

for week in range(2, 51):
    scheduled = demand[week]
    
    # Step 1: Reporting value
    start_past_due = max(0, prev_end_backlog)
    
    # Step 2: Calc Start (signed)
    calc_start = prev_end_backlog
    
    # For Week 2, verify: start_past_due + scheduled should equal 910.80
    # calc_start + scheduled = 910.80 - demand[2] + demand[2] = 910.80 ✓
    
    # Step 3: Choose Days Worked
    if start_past_due > 0.01:
        # Try 5 first, then 6
        if calc_start + scheduled - (22 * 5) <= 0:
            days_worked = 5
        elif calc_start + scheduled - (22 * 6) <= 0:
            days_worked = 6
        else:
            days_worked = 6
    else:
        if scheduled <= 110:
            days_worked = 4
        else:
            days_worked = 5
    
    # Step 4
    weekly_capacity = 22 * days_worked
    
    # Step 5
    end_backlog = calc_start + scheduled - weekly_capacity
    
    # Step 6
    overtime = 10 * max(0, days_worked - 4)
    
    results.append({
        'Week': week,
        'Days Worked': days_worked,
        'Scheduled Demand (Std Hrs)': scheduled,
        'Weekly Capacity (Std Hrs)': weekly_capacity,
        'Start of Week Past Due (Std Hrs)': round(start_past_due, 2),
        'End of Week Backlog/Buffer (Std Hrs)': round(end_backlog, 2),
        'Overtime Hours': overtime
    })
    
    prev_end_backlog = end_backlog
```

## Step 3: Write the Excel file

Create `/root/glass_furnace_plan.xlsx` with a single sheet named `Plan`. Row 1 must have exactly these 7 headers in this order:
1. `Week`
2. `Days Worked`
3. `Scheduled Demand (Std Hrs)`
4. `Weekly Capacity (Std Hrs)`
5. `Start of Week Past Due (Std Hrs)`
6. `End of Week Backlog/Buffer (Std Hrs)`
7. `Overtime Hours`

Rows 2–50 contain data for Weeks 2–50 (49 data rows). Verify the sheet has exactly 50 rows (1 header + 49 data).

## Step 4: Determine summary values

Scan the results to find:
- `first_5_day_week`: The first week where `Days Worked == 5`. If none, use `N/A`.
- `first_4_day_week`: The first week where `Days Worked == 4`. If none, use `N/A`.

Based on previous successful execution, expect Week 2 as first 5-day week and Week 36 as first 4-day week. But compute from actual results.

## Step 5: Write the summary file

Create `/root/glass_furnace_summary.txt` with exactly 3 lines (no trailing blank lines):

```
First_Week_5_Days: <week-number-or-N/A>
First_Week_4_Days: <week-number-or-N/A>
Summary: <summary text>
```

The summary text MUST:
- Be ≤ 60 words and ≤ 3 sentences.
- Mention BOTH step-down week numbers explicitly (e.g., "Week 2" and "Week 36"), or the literal string `N/A` if a transition didn't occur.
- Be a manager-facing description of the catch-up plan.

**CRITICAL**: If either value is `N/A`, the literal string `N/A` MUST appear in the Summary line. Always include both week identifiers (or `N/A`) in the summary text.

Example summary (adjust to actual values):
```
Summary: The furnace crew works 6-day weeks to clear initial backlog, stepping down to 5 days at Week 2 and to 4 days at Week 36 as demand eases. Overtime decreases progressively, balancing capacity with scheduled demand across Weeks 2–50.
```

## Step 6: Validate

1. Re-open `/root/glass_furnace_plan.xlsx` and verify: sheet name is `Plan`, 7 headers match exactly, 49 data rows, Week column goes 2..50.
2. Re-read `/root/glass_furnace_summary.txt` and verify: exactly 3 lines, first two lines match the format, summary mentions both week numbers, word count ≤ 60, sentence count ≤ 3.
3. Print all validation results.

Do all work in a single Python script. Print intermediate values (demand readings, first few rows of results, summary values) for debugging.

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