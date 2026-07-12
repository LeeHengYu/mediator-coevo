# Task Instruction

Execute the following steps to produce `/root/ship_block_plan.xlsx` and `/root/ship_block_summary.txt`.

## Step 1: Inspect the CSV
Read `/root/ship_demand.csv` and print its first few rows to confirm the layout. The file has week numbers as column headers in row 1 (Week, 5, 6, 7, ..., 53) and demand values in row 2 (Demand, val5, val6, ..., val53). Transpose this: for each column after the first, the header is the week number and the row-2 value is that week's scheduled demand.

## Step 2: Build the plan using Python
Write a Python script that:

1. Reads the CSV and builds a dict `{week_number: demand}` for weeks 5..53.
2. Iterates weeks 5 through 53 in order, computing each row per the deterministic policy below.
3. Writes the results to `/root/ship_block_plan.xlsx` (single sheet named `Plan`) and `/root/ship_block_summary.txt`.

### Initial condition (Week 5)
- The initial condition states: `Start of Week Past Due + Scheduled Demand = 1014.51`.
- Week 5's `Scheduled Demand` comes from the CSV.
- Therefore `Calc Start` for Week 5 = `1014.51 - Scheduled Demand(week5)`.
- `Start of Week Past Due` for Week 5 = `max(0, Calc Start)`.

### Per-week logic (for week w):
```
if w == 5:
    calc_start = 1014.51 - demand[w]
else:
    calc_start = prior_end_of_week_backlog

start_past_due = max(0, calc_start)

scheduled_demand = demand[w]

# Choose Days Worked
if start_past_due > 0.01:
    # Try 5 first, then 6
    if calc_start + scheduled_demand - 28*5 <= 0:
        days = 5
    elif calc_start + scheduled_demand - 28*6 <= 0:
        days = 6
    else:
        days = 6
else:
    if scheduled_demand <= 112:
        days = 4
    else:
        days = 5

capacity = 28 * days
end_backlog = calc_start + scheduled_demand - capacity
overtime = 10 * max(0, days - 4)
```

Store `end_backlog` as `prior_end_of_week_backlog` for the next week.

### Excel output
Use `openpyxl` to create `/root/ship_block_plan.xlsx` with sheet name `Plan`. Row 1 headers exactly:
- `Week`
- `Days Worked`
- `Scheduled Demand (Std Hrs)`
- `Weekly Capacity (Std Hrs)`
- `Start of Week Past Due (Std Hrs)`
- `End of Week Backlog/Buffer (Std Hrs)`
- `Overtime Hours`

Rows 2..50 contain weeks 5..53 (49 data rows), ascending, no gaps or duplicates. Write numeric values (not formulas).

### Summary output
Track:
- `first_week_5_days`: the first week where `Days Worked == 5` AND the previous week had `Days Worked == 6` (i.e., the first step-down from 6 to 5). If no week ever has 6 days, look for the first week with exactly 5 days. More precisely: scan weeks in order and record the first week where Days Worked is exactly 5.
- `first_week_4_days`: the first week where `Days Worked == 4`.
- If either never occurs, use `N/A`.

Write `/root/ship_block_summary.txt` with exactly 3 lines:
```
First_Week_5_Days: <week-number-or-N/A>
First_Week_4_Days: <week-number-or-N/A>
Summary: <manager-facing summary, ≤60 words, ≤3 sentences, mentioning both step-down week numbers or N/A>
```

The summary should briefly describe the catch-up trajectory: starting with 6-day weeks to clear backlog, stepping down to 5-day weeks at week X, and to 4-day weeks at week Y.

## Step 3: Verify
1. Confirm the Excel file has exactly 49 data rows and 7 columns with correct headers.
2. Print the first 5 and last 5 rows of the plan for visual inspection.
3. Print the contents of `/root/ship_block_summary.txt`.
4. Verify the summary is ≤60 words and ≤3 sentences.

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