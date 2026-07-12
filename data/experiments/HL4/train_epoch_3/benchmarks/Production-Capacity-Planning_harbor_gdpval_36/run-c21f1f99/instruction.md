# Task Instruction

Execute the following steps in order:

## Step 1 – Inspect the source workbook

Open `/root/copy_of_capacity_sheet.xlsx`, sheet `Weld`.
- Find the row whose label contains `MIG weld Demand Total` (or similar).
- Print all column headers (row 1) and the demand row so you can see which columns correspond to Weeks 4–52.
- Note the exact demand values for every week from 4 through 52 inclusive (49 values). Print them all.

## Step 2 – Build the schedule with Python

Write a Python script that:

1. Reads the 49 weekly demand values (Weeks 4–52) from the spreadsheet using `openpyxl`.
2. Implements the deterministic policy below **exactly**:

```
initial_condition = 438.81   # Calc Start for Week 4

For each week w (4..52) in order:
   demand = scheduled_demand[w]          # from the spreadsheet

   if w == 4:
       calc_start = initial_condition - demand   # because initial_condition = past_due + demand
       # Equivalently: Start of Week Past Due for week 4 = initial_condition - demand
   else:
       calc_start = prior_week_end_of_week_backlog

   start_of_week_past_due = max(0, calc_start)

   # Choose Days Worked
   if start_of_week_past_due > 0.01:
       # Try 5 first, then 6
       if calc_start + demand - (30 * 5) <= 0:
           days_worked = 5
       elif calc_start + demand - (30 * 6) <= 0:
           days_worked = 6
       else:
           days_worked = 6
   else:
       # Past due is essentially zero
       if demand <= 120:
           days_worked = 4
       else:
           days_worked = 5

   weekly_capacity = 30 * days_worked
   end_of_week_backlog = calc_start + demand - weekly_capacity
   overtime_hours = 10 * max(0, days_worked - 4)

   Store: Week, Days Worked, Scheduled Demand (Std Hrs), Weekly Capacity (Std Hrs),
          Start of Week Past Due (Std Hrs), End of Week Backlog/Buffer (Std Hrs), Overtime Hours

   prior_week_end_of_week_backlog = end_of_week_backlog
```

**IMPORTANT initial-condition interpretation:** The problem says "Initial condition at Week 4: Start of Week Past Due + Scheduled Demand = 438.81". This means:
- `Start of Week Past Due` for Week 4 = 438.81 − demand[Week 4]
- `Calc Start` for Week 4 = that same value (438.81 − demand[Week 4])

Wait — re-read: "Start of Week Past Due + Scheduled Demand = 438.81" could also mean the total work to do in Week 4 is 438.81. Since `Calc Start + Scheduled Demand` is the total work entering the week, we have `Calc Start` for Week 4 = 438.81 − demand[Week 4]. And `Start of Week Past Due` = max(0, Calc Start) for Week 4.

Actually, let me reconsider. The simplest reading: the initial condition states that `Start of Week Past Due + Scheduled Demand = 438.81` for Week 4. Since `Start of Week Past Due = max(0, Calc Start)` and `Calc Start` for week 4 has no prior week, the most natural reading is:
- `Calc Start` (week 4) = 438.81 - demand[week 4]
- `Start of Week Past Due` (week 4) = max(0, Calc Start)

Use this interpretation. Print the first 5 rows and last 5 rows of the schedule for verification.

## Step 3 – Write `/root/catch_up_plan.xlsx`

Using `openpyxl`, create a workbook with a single sheet named `Plan`. Row 1 must contain exactly these headers in this order:
1. `Week`
2. `Days Worked`
3. `Scheduled Demand (Std Hrs)`
4. `Weekly Capacity (Std Hrs)`
5. `Start of Week Past Due (Std Hrs)`
6. `End of Week Backlog/Buffer (Std Hrs)`
7. `Overtime Hours`

Rows 2–50 contain the 49 data rows (Weeks 4–52), in ascending week order. Save to `/root/catch_up_plan.xlsx`.

## Step 4 – Write `/root/catch_up_summary.txt`

Determine:
- `First_Week_5_Days`: the first week where Days Worked drops to exactly 5 after having been 6 at some earlier point. If Days Worked is never 6, or never transitions to 5 after being 6, use `N/A`. More precisely: the first week where Days Worked == 5 (the step-down from 6 to 5).
- `First_Week_4_Days`: the first week where Days Worked == 4.

Actually, re-read the requirement: "step-down week numbers". These are the first week the schedule steps down to 5 days and the first week it steps down to 4 days. So:
- `First_Week_5_Days` = the first week number where Days Worked == 5 (if it was 6 in some prior week, this is the step-down to 5; if the very first week is already 5, that still counts as the first week with 5 days).
- `First_Week_4_Days` = the first week number where Days Worked == 4.

Write exactly 3 lines:
```
First_Week_5_Days: <number or N/A>
First_Week_4_Days: <number or N/A>
Summary: <manager-facing summary, ≤60 words, ≤3 sentences, mentioning both step-down week numbers or N/A>
```

No trailing newline after the third line is fine, but ensure exactly 3 lines.

## Step 5 – Validate

1. Re-open `/root/catch_up_plan.xlsx` and verify:
   - Sheet name is exactly `Plan`
   - 7 headers match exactly
   - 49 data rows, weeks 4–52 with no gaps or duplicates
   - Week 4 row: confirm `Start of Week Past Due + Scheduled Demand` equals 438.81 (within rounding)
   - All Days Worked values are in {4, 5, 6}
   - `End of Week Backlog/Buffer = Calc Start + Demand - Capacity` spot-check a few rows
2. Re-read `/root/catch_up_summary.txt` and verify it has exactly 3 lines in the required format, summary ≤60 words and ≤3 sentences.
3. Print confirmation of all checks.

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