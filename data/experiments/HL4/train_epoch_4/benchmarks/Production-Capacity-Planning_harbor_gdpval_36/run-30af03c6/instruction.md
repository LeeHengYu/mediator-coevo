# Task Instruction

## Task: Production Capacity Planning – MIG Welding Catch-Up Plan

You must read `/root/copy_of_capacity_sheet.xlsx` (sheet `Weld`), compute a deterministic week-by-week plan for Weeks 4–52, and produce two output files.

### Step 1: Read the input data

Open `/root/copy_of_capacity_sheet.xlsx`, sheet `Weld`. Find the row labeled `MIG weld Demand Total`. Extract the scheduled demand values for Weeks 4 through 52 inclusive. The columns in the sheet correspond to week numbers — inspect the header row carefully to map week numbers to columns. Print out the demand values you extract so you can verify them.

### Step 2: Compute the plan

Initialize: For Week 4, `Calc Start` = 438.81 - (Week 4's Scheduled Demand). Wait — re-read: the initial condition says "Start of Week Past Due + Scheduled Demand = 438.81" at Week 4. This means `Calc Start + Scheduled Demand = 438.81` for Week 4. So `Calc Start` for Week 4 = 438.81 - Week4_Demand. NO — re-read more carefully:

The initial condition is: at Week 4, `Start of Week Past Due + Scheduled Demand = 438.81`. The `Calc Start` for Week 4 comes from the initial condition. Since Week 4 is the first week, there is no prior week. The way to interpret this: the total workload entering Week 4 is 438.81 std hrs. So `Calc Start` for Week 4 = 438.81 - Scheduled_Demand_Week4. Then `Start of Week Past Due` for Week 4 = max(0, Calc Start). Actually wait — let me re-read once more.

Actually the simplest consistent interpretation: `Calc Start` for Week 4 = 438.81 - Scheduled_Demand_Week4, and `Start of Week Past Due` = max(0, Calc Start). Then the total entering = Calc Start + Scheduled Demand = 438.81. YES, that is consistent.

So for Week 4:
- `Calc Start = 438.81 - Scheduled_Demand_Week4`
- `Start of Week Past Due = max(0, Calc Start)`

Then for each week (starting at Week 4):
1. `Start of Week Past Due = max(0, Calc Start)` — for reporting.
2. Choose `Days Worked`:
   - If `Start of Week Past Due > 0.01`: pick the smallest value from {5, 6} such that `Calc Start + Scheduled Demand - (30 * Days Worked) <= 0`. If neither works, pick 6.
   - If `Start of Week Past Due <= 0.01`: pick 4 if `Scheduled Demand <= 120`, else 5.
3. `Weekly Capacity = 30 * Days Worked`
4. `End of Week Backlog/Buffer = Calc Start + Scheduled Demand - Weekly Capacity`
5. `Overtime Hours = 10 * max(0, Days Worked - 4)`
6. For next week: `Calc Start = End of Week Backlog/Buffer` (the signed value, not clamped).

Compute all 49 weeks (Week 4 through Week 52).

### Step 3: Identify transition weeks

- `First_Week_5_Days`: The first week where Days Worked drops to exactly 5 after having been 6. More precisely: scan weeks in order; find the first week where Days Worked == 5. If the policy never picks 5 (or only picks 5 from the start due to past-due logic), be careful — it's the first week where 5 days are chosen under the past-due branch OR the normal branch. Actually, re-read: it's the first week where Days Worked is 5 (the step-down from 6 to 5). Scan in order and find the first week with Days Worked == 5.
- `First_Week_4_Days`: The first week where Days Worked == 4.
- If either never occurs, use `N/A`.

### Step 4: Write `/root/catch_up_plan.xlsx`

Using openpyxl, create a workbook with a single sheet named `Plan`. Row 1 must have exactly these headers:
1. `Week`
2. `Days Worked`
3. `Scheduled Demand (Std Hrs)`
4. `Weekly Capacity (Std Hrs)`
5. `Start of Week Past Due (Std Hrs)`
6. `End of Week Backlog/Buffer (Std Hrs)`
7. `Overtime Hours`

Then 49 data rows (Weeks 4–52), one per row, in ascending order. All numeric values should be stored as numbers (not strings). Round floats to 2 decimal places for cleanliness but do not truncate precision that matters.

### Step 5: Write `/root/catch_up_summary.txt`

Exactly 3 lines, no trailing blank lines:
```
First_Week_5_Days: <week-number-or-N/A>
First_Week_4_Days: <week-number-or-N/A>
Summary: <summary>
```

The summary must be ≤60 words, ≤3 sentences, and must mention both step-down week numbers (or N/A).

### Validation

After creating both files:
1. Re-open `/root/catch_up_plan.xlsx` and verify: sheet name is `Plan`, headers match exactly, there are exactly 49 data rows, Week column goes 4..52 with no gaps.
2. Re-read `/root/catch_up_summary.txt` and verify: exactly 3 lines, correct format, summary ≤60 words and ≤3 sentences.
3. Spot-check Week 4 computation manually and print it.
4. Print the first few and last few rows of the plan for visual inspection.

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