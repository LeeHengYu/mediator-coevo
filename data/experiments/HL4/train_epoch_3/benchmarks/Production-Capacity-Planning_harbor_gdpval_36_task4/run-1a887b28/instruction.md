# Task Instruction

Execute the following steps in order:

## Step 1: Inspect input files

1. Read `/root/hvac_demand_sheet.xlsx` (sheet `Install`) to find the row labeled `HVAC Ductwork Demand (Std Hrs)`. Extract demand values for Phases 8 through 56 inclusive. Print them all so we can verify.
2. Read `/root/hvac_existing_plan.xlsx` (sheet `Plan`) to see its current structure (headers, rows). Print the headers and first few data rows.

## Step 2: Build the schedule using Python

Write and run a Python script that:

### 2a: Load demand data
- Opens `/root/hvac_demand_sheet.xlsx`, sheet `Install`.
- Finds the row whose first non-empty cell contains `HVAC Ductwork Demand (Std Hrs)` (search all rows, match case-insensitively or with partial match on 'Ductwork Demand').
- Finds the header row that maps column positions to phase numbers (likely a row above or the first row containing numeric phase identifiers 8..56). Extract the demand value for each phase 8 through 56.
- Print all 49 (phase, demand) pairs.

### 2b: Compute the plan
- Initialize: For Phase 8, `Calc Start` is derived from the initial condition: `Start of Phase Past Due + Scheduled Demand = 1138.66`, so `Calc Start + Scheduled Demand = 1138.66`, meaning `Calc Start = 1138.66 - demand[8]`. The `Start of Phase Past Due` for Phase 8 = `max(0, Calc Start)` (which equals `Calc Start` since it should be positive given the initial condition).

Wait — re-read the instruction carefully:
- "Initial condition at Phase 8: Start of Phase Past Due + Scheduled Demand = 1138.66"
- "Calc Start = prior phase End of Phase Backlog/Buffer (Phase 8 starts from the initial condition)"

So for Phase 8, `Calc Start` is such that `Start of Phase Past Due + Scheduled Demand = 1138.66`. Since Phase 8 is the first phase, `Start of Phase Past Due = max(0, Calc Start)`. If `Calc Start >= 0`, then `Calc Start + demand[8] = 1138.66`, so `Calc Start = 1138.66 - demand[8]`.

For each phase p from 8 to 56:
1. If p == 8: `calc_start = 1138.66 - demand[8]`; `past_due = max(0, calc_start)`
2. Else: `calc_start = prev_end_backlog`; `past_due = max(0, calc_start)`
3. Choose `days_worked`:
   - If `past_due > 0.01`:
     - Try 5: if `calc_start + demand[p] - 35*5 <= 0`, use 5
     - Else try 6: if `calc_start + demand[p] - 35*6 <= 0`, use 6
     - Else use 6
   - Else (past_due <= 0.01):
     - If `demand[p] <= 140`: use 4
     - Else: use 5
4. `weekly_capacity = 35 * days_worked`
5. `end_backlog = calc_start + demand[p] - weekly_capacity`
6. `overtime = 10 * max(0, days_worked - 4)`
7. Store row: Phase, Days Worked, Scheduled Demand, Weekly Capacity, Start of Phase Past Due, End of Phase Backlog/Buffer, Overtime Hours
8. Set `prev_end_backlog = end_backlog` for next phase.

Print all 49 rows to verify.

### 2c: Find summary values
- `First_Week_5_Days`: the first phase where `days_worked == 5` AND the previous phase had `days_worked == 6`. If no such phase, check if the very first phase (8) has days_worked == 5. Actually re-read: "First_Week_5_Days" means the first phase where days worked steps down to 5 from 6. More precisely, it's simply the first phase where days_worked == 5. Print and record it.
- Wait, re-read: the summary must mention "step-down phase numbers". So `First_Week_5_Days` = first phase with exactly 5 days worked (stepping down from 6). `First_Week_4_Days` = first phase with exactly 4 days worked (stepping down from 5 or 6). Simply: first occurrence of 5-day week and first occurrence of 4-day week.
- If never occurs, use `N/A`.

### 2d: Write output files

1. Write `/root/hvac_existing_plan.xlsx` with sheet named `Plan`, headers exactly:
   `Phase`, `Days Worked`, `Scheduled Demand (Std Hrs)`, `Weekly Capacity (Std Hrs)`, `Start of Phase Past Due (Std Hrs)`, `End of Phase Backlog/Buffer (Std Hrs)`, `Overtime Hours`
   — 49 data rows for phases 8-56, ascending order. Use openpyxl. Round numeric values to 2 decimal places for display.

2. Copy that file to `/root/hvac_schedule_plan.xlsx` (use shutil.copy).

3. Write `/root/hvac_schedule_summary.txt` with exactly 3 lines:
   ```
   First_Week_5_Days: <phase or N/A>
   First_Week_4_Days: <phase or N/A>
   Summary: <summary text, ≤60 words, ≤3 sentences, mentioning both step-down phases>
   ```
   The summary should be a concise manager-facing statement like: "Crew works 6-day weeks through Phase X to clear the backlog, stepping down to 5-day weeks at Phase Y and to 4-day weeks at Phase Z. Overtime decreases as past-due hours are eliminated."

## Step 3: Validate

1. Re-read `/root/hvac_schedule_plan.xlsx` and print all rows to confirm 49 data rows, correct headers, correct phase range 8-56, no gaps/duplicates.
2. Verify Phase 8: `Start of Phase Past Due + Scheduled Demand` should equal `1138.66`.
3. Re-read `/root/hvac_existing_plan.xlsx` and confirm it matches `/root/hvac_schedule_plan.xlsx`.
4. Read and print `/root/hvac_schedule_summary.txt`, confirm exactly 3 lines, word count of summary line ≤ 60, sentence count ≤ 3.
5. Print the first phase with 5 days and first phase with 4 days to cross-check the summary file values.

Do NOT skip any validation step. If any check fails, fix and re-validate.

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
Task metadata: author_email=codex@openai.com, author_name=Codex, category=manufacturing-planning, difficulty=medium, tags=[xlsx, operations, capacity-planning, hvac, backlog].
Verifier config: timeout_sec=900.0.