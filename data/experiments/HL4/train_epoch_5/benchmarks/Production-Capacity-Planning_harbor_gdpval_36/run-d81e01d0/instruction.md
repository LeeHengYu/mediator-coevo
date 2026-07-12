# Task Instruction

Execute the following steps precisely to produce `/root/catch_up_plan.xlsx` and `/root/catch_up_summary.txt`.

## Step 1: Read the source data

Open `/root/copy_of_capacity_sheet.xlsx`, sheet `Weld`. Locate the row whose first non-empty cell (or label cell) contains the text `MIG weld Demand Total` (case-insensitive match; try partial match if exact fails). Extract the demand values for Weeks 4 through 52 inclusive. The week numbers should correspond to column headers or column positions in that sheet — inspect the sheet structure first to determine the mapping. Print the first 10 demand values to confirm they are numeric and non-zero. If any demand value for weeks 4-52 is missing or zero, re-examine the row/column mapping.

## Step 2: Compute the plan

Using Python, compute the 49-row plan (Weeks 4..52) with these variables per week:

```python
import openpyxl

# demand[w] = scheduled demand for week w (from Step 1)
# Initial condition: calc_start for week 4 = 438.81

rows = []
calc_start = 438.81  # Week 4 initial Calc Start

for i, week in enumerate(range(4, 53)):
    demand = demand_values[i]  # demand for this week
    
    # Start of Week Past Due (for reporting)
    if i == 0:
        past_due = max(0, calc_start)  # Week 4: calc_start=438.81 > 0
    else:
        past_due = max(0, calc_start)  # calc_start = prior week's end_of_week
    
    # Note: calc_start is the SIGNED carryover, used for calculations
    # past_due is max(0, calc_start), used only for reporting and the decision rule
    
    # Choose Days Worked
    if past_due > 0.01:
        # Try 5 first, then 6
        if calc_start + demand - (30 * 5) <= 0:
            days = 5
        elif calc_start + demand - (30 * 6) <= 0:
            days = 6
        else:
            days = 6
    else:
        # past_due <= 0.01
        if demand <= 120:
            days = 4
        else:
            days = 5
    
    capacity = 30 * days
    end_of_week = calc_start + demand - capacity
    overtime = 10 * max(0, days - 4)
    
    rows.append({
        'Week': week,
        'Days Worked': days,
        'Scheduled Demand (Std Hrs)': round(demand, 2),
        'Weekly Capacity (Std Hrs)': capacity,
        'Start of Week Past Due (Std Hrs)': round(past_due, 2),
        'End of Week Backlog/Buffer (Std Hrs)': round(end_of_week, 2),
        'Overtime Hours': overtime,
    })
    
    # Next week's calc_start = this week's end_of_week (signed)
    calc_start = end_of_week
```

Print the first 5 rows and last 3 rows for verification. Confirm Week 4 has past_due = 438.81.

## Step 3: Write `/root/catch_up_plan.xlsx`

Create the workbook with a single sheet named `Plan`. Row 1 must contain exactly these headers in order:
1. `Week`
2. `Days Worked`
3. `Scheduled Demand (Std Hrs)`
4. `Weekly Capacity (Std Hrs)`
5. `Start of Week Past Due (Std Hrs)`
6. `End of Week Backlog/Buffer (Std Hrs)`
7. `Overtime Hours`

Write 49 data rows (Weeks 4-52) starting from row 2. Store numeric values as numbers, not strings. Save the file.

## Step 4: Determine summary values

Scan the rows in week order:
- `First_Week_5_Days`: the first week where `Days Worked == 5` AND the prior week had `Days Worked == 6` (i.e., the first step-down from 6 to 5). If no such transition exists, check if the very first occurrence of 5-day weeks after any 6-day period counts. If Days Worked never equals 5 after being 6, use `N/A`.

Wait — re-read the task: it says "step-down week numbers". The natural interpretation:
- `First_Week_5_Days`: The first week where Days Worked = 5 (simply the first 5-day week).
- `First_Week_4_Days`: The first week where Days Worked = 4 (simply the first 4-day week).

Use the simpler interpretation: first week number where days=5, first week number where days=4. Use `N/A` if never reached.

## Step 5: Write `/root/catch_up_summary.txt`

Exactly 3 lines, no trailing blank lines:
```
First_Week_5_Days: <week_number_or_N/A>
First_Week_4_Days: <week_number_or_N/A>
Summary: <manager-facing summary, ≤60 words, ≤3 sentences, mentioning both step-down week numbers or N/A>
```

The summary should mention the transition from 6-day to 5-day weeks at week X and from 5-day to 4-day weeks at week Y (using the values found).

## Step 6: Validate

1. Re-open `/root/catch_up_plan.xlsx` and confirm: sheet name is `Plan`, 7 headers match exactly, 49 data rows, Week column goes 4..52 with no gaps.
2. Re-read `/root/catch_up_summary.txt` and confirm exactly 3 lines, correct format, summary ≤ 60 words and ≤ 3 sentences.
3. Print the full contents of the summary file.
4. Print a few sample rows from the Excel file to confirm numeric types.

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

Task-local resources are available under `environment/skills`: Create Manufacturing Schedule Analysis Dashboard, autonomy-windowed, manufacturing-team-momentum, token-efficiency-guide, verification-before-completion.

# Verifier Contract

Success is judged by the SkillFlow verifier for this task.
Do not bypass, remove, or weaken verifier scripts, tests, fixtures, or expected-output checks.
Run the provided tests or verifier command when practical before finalizing.
Task metadata: author_email=codex@openai.com, author_name=Codex, category=manufacturing-planning, difficulty=medium, tags=[xlsx, operations, capacity-planning, backlog].
Verifier config: timeout_sec=900.0.