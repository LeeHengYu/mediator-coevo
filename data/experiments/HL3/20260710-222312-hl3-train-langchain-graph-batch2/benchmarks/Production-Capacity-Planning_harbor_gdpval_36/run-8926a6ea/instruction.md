# Task Instruction

## Task: Build MIG Welding Catch-Up Plan

You must read `/root/copy_of_capacity_sheet.xlsx` (sheet `Weld`) and produce two files:
1. `/root/catch_up_plan.xlsx` — a single sheet named `Plan`
2. `/root/catch_up_summary.txt` — exactly 3 lines

### Step 1: Inspect the source data

Open `/root/copy_of_capacity_sheet.xlsx`, sheet `Weld`. Find the row labeled `MIG weld Demand Total`. Identify which columns correspond to Weeks 4 through 52. Print out the demand values for all 49 weeks so you can verify them. Note: the week numbers may be in a header row; inspect carefully to map column positions to week numbers.

### Step 2: Implement the deterministic planning algorithm in Python

Use `openpyxl` to read the source and write the output.

Variables for Week 4 initialization:
- `calc_start` for Week 4 = 438.81 - (Scheduled Demand for Week 4). Wait — re-read: "Initial condition at Week 4: Start of Week Past Due + Scheduled Demand = 438.81". This means at Week 4, the total of past due plus demand equals 438.81. So `calc_start` (which equals prior week's End of Week Backlog/Buffer) must satisfy: `Start of Week Past Due + Scheduled Demand = 438.81`. Since Week 4 is the first week, `Start of Week Past Due = max(0, calc_start)` and `calc_start + demand = 438.81` only if we interpret the initial condition as: the effective load entering Week 4 is 438.81. So treat Week 4 as: `calc_start = 438.81 - scheduled_demand_week4`, and `Start of Week Past Due = max(0, calc_start)`. 

ACTUALLY, re-read more carefully. The policy says:
- `Calc Start = prior week End of Week Backlog/Buffer` (Week 4 starts from the initial condition)
- `Start of Week Past Due = max(0, prior week End of Week Backlog/Buffer)`

The initial condition says `Start of Week Past Due + Scheduled Demand = 438.81` at Week 4. So `Start of Week Past Due` at Week 4 = 438.81 - demand_week4. And since `Start of Week Past Due = max(0, calc_start)`, we need `calc_start` such that `max(0, calc_start) = 438.81 - demand_week4`. If 438.81 - demand_week4 >= 0, then `calc_start = 438.81 - demand_week4`. This is the most natural reading.

So: For Week 4, set `calc_start = 438.81 - scheduled_demand_week4` and `start_of_week_past_due = max(0, calc_start)`.

For each week w (4..52) in order:
1. If w == 4: `calc_start = 438.81 - demand[w]`; `past_due = max(0, calc_start)`
   Else: `calc_start = prev_end_of_week_backlog`; `past_due = max(0, calc_start)`
2. `demand = scheduled_demand[w]`
3. Choose `days_worked`:
   - If `past_due > 0.01`:
     - Try 5: if `calc_start + demand - 150 <= 0`, use 5
     - Else try 6: if `calc_start + demand - 180 <= 0`, use 6
     - Else use 6
   - Else (past_due <= 0.01):
     - If `demand <= 120`: days = 4
     - Else: days = 5
4. `capacity = 30 * days_worked`
5. `end_backlog = calc_start + demand - capacity`
6. `overtime = 10 * max(0, days_worked - 4)`
7. Store row: [w, days_worked, demand, capacity, past_due, end_backlog, overtime]
8. `prev_end_of_week_backlog = end_backlog`

### Step 3: Write `/root/catch_up_plan.xlsx`

Create workbook with single sheet named `Plan`. Row 1 headers exactly:
`Week`, `Days Worked`, `Scheduled Demand (Std Hrs)`, `Weekly Capacity (Std Hrs)`, `Start of Week Past Due (Std Hrs)`, `End of Week Backlog/Buffer (Std Hrs)`, `Overtime Hours`

Write 49 data rows (Weeks 4-52) starting at row 2. All numeric values must be written as Python floats/ints, NOT strings. Verify by re-reading the file after writing.

### Step 4: Determine summary values

- `First_Week_5_Days`: The first week number where `Days Worked == 5` AND the previous week had `Days Worked == 6` (i.e., the step-down from 6 to 5). If this never happens, check: it could also mean the first week where days_worked becomes 5 after having been 6. If days never reach 6, use the first week with 5 days. If no week has 5 days, use N/A.

Wait — re-read the instruction: "First_Week_5_Days" and "First_Week_4_Days" with "step-down week numbers". This means:
- `First_Week_5_Days`: first week where days worked drops TO 5 (from 6). i.e., the first week with days_worked=5 that follows a period of days_worked=6.
- `First_Week_4_Days`: first week where days worked drops TO 4 (from 5 or 6).

Actually, simpler reading: just the first week number where Days Worked equals 5, and the first week number where Days Worked equals 4. The summary mentions "step-down" because going from 6→5→4 represents stepping down. Let me go with the simplest interpretation:
- `First_Week_5_Days` = first week number where Days Worked == 5
- `First_Week_4_Days` = first week number where Days Worked == 4
- Use N/A if that never occurs.

### Step 5: Write `/root/catch_up_summary.txt`

Exactly 3 lines, no trailing blank lines:
```
First_Week_5_Days: <number or N/A>
First_Week_4_Days: <number or N/A>
Summary: <1-3 sentences, ≤60 words, mentioning both step-down week numbers or N/A>
```

### Step 6: Validate

1. Re-open `/root/catch_up_plan.xlsx` and verify:
   - Sheet name is exactly `Plan`
   - 7 columns with exact header names
   - 49 data rows, weeks 4-52 in order, no gaps/duplicates
   - All values are numeric (not strings)
   - Week 4 row: verify `Start of Week Past Due + Scheduled Demand` ≈ 438.81
   - Spot-check a few weeks' calculations
2. Read `/root/catch_up_summary.txt` and verify format: exactly 3 lines, summary ≤ 60 words and ≤ 3 sentences.
3. Print key diagnostic values: Week 4 calc_start, demand, past_due, days, end_backlog. Print total overtime. Print first/last few rows.

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