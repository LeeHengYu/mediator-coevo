# Task Instruction

Create a single Python script `/root/solve.py` and execute it. The script must:

1. **Read `/root/ship_demand.csv`**: The CSV has two rows. The first row is `Week,5,6,7,...,53` and the second row is `Demand,<value>,<value>,...`. Parse this by reading the header row to get week numbers (columns 1 onward) and the second row to get corresponding demand values. Build a dict mapping week number (int) to demand (float).

2. **Run the simulation for Weeks 5 through 53 (49 weeks)**:
   - **Initial condition**: `Calc_Start` for Week 5 = 1014.51 (this is the initial `Start of Week Past Due + Scheduled Demand`, which serves as the starting backlog before Week 5's own demand is added... actually, re-read carefully: the initial condition says "Start of Week Past Due + Scheduled Demand = 1014.51" at Week 5. This means for Week 5, `Calc_Start + Scheduled_Demand = 1014.51`, so `Calc_Start = 1014.51 - Scheduled_Demand_Week5`. Wait — let me re-read the policy more carefully.)

   Actually, the policy says:
   - `Calc Start = prior week End of Week Backlog/Buffer` (Week 5 starts from the initial condition).
   - `Start of Week Past Due = max(0, prior week End of Week Backlog/Buffer)` for reporting.
   - `End of Week Backlog/Buffer = Calc Start + Scheduled Demand - Weekly Capacity`

   The initial condition states: at Week 5, `Start of Week Past Due + Scheduled Demand = 1014.51`. Since `Start of Week Past Due = max(0, Calc_Start)` and for Week 5 `Calc_Start` is the initial value, and since this sum equals 1014.51, and `Scheduled Demand` for Week 5 comes from the CSV, we get: `max(0, Calc_Start) + Demand_Week5 = 1014.51`. Since this is the start and past due is positive, `Calc_Start = 1014.51 - Demand_Week5`. Use this as the initial `Calc_Start` for Week 5.

   For each week (5..53) in order:
   - `Start_of_Week_Past_Due = max(0, Calc_Start)`
   - Determine `Days_Worked`:
     - If `Start_of_Week_Past_Due > 0.01`: pick smallest from {5, 6} such that `Calc_Start + Demand - 28*Days <= 0`. If neither works, pick 6.
     - Else (`Start_of_Week_Past_Due <= 0.01`): pick 4 if `Demand <= 112`, else 5.
   - `Weekly_Capacity = 28 * Days_Worked`
   - `End_of_Week_Backlog = Calc_Start + Demand - Weekly_Capacity`
   - `Overtime = 10 * max(0, Days_Worked - 4)`
   - For the next week: `Calc_Start = End_of_Week_Backlog` (the signed value, not clamped).

3. **Write `/root/ship_block_plan.xlsx`** using `openpyxl`:
   - Single worksheet named `Plan`.
   - Row 1 headers exactly: `Week`, `Days Worked`, `Scheduled Demand (Std Hrs)`, `Weekly Capacity (Std Hrs)`, `Start of Week Past Due (Std Hrs)`, `End of Week Backlog/Buffer (Std Hrs)`, `Overtime Hours`.
   - 49 data rows (Weeks 5..53), ascending order, no gaps/duplicates.
   - Write numeric values (not formulas) to avoid #NAME? errors.

4. **Write `/root/ship_block_summary.txt`** with exactly 3 lines:
   - `First_Week_5_Days: <week>` — the first week where Days Worked = 5 (or `N/A`).
   - `First_Week_4_Days: <week>` — the first week where Days Worked = 4 (or `N/A`).
   - `Summary: <text>` — a manager-facing summary of ≤60 words and ≤3 sentences mentioning both step-down week numbers (or N/A).

5. After writing both files, print the first 5 and last 5 rows of the plan for verification, and print the summary file contents.

Run the script with `python /root/solve.py` and confirm both output files exist and look correct.

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