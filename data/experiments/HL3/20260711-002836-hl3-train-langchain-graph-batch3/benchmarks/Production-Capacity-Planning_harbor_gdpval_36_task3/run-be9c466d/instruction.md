# Task Instruction

Write and execute a Python script `/root/solve.py` that performs the following steps:

1. **Read the source data**: Open `/root/assembly_schedule.xlsx`, sheet `Assembly`. Read all rows. Identify the column labeled `PCB Assembly Demand (Std Hrs)` and the `Phase` column. Drop any rows where Phase is outside 6–54 inclusive. Among remaining rows, keep only the first occurrence of each phase (drop duplicates on Phase, keeping first). Sort by Phase ascending. You should end up with exactly 49 phases (6, 7, 8, …, 54).

2. **Compute the plan** using this deterministic logic:

   - For Phase 6:
     - `demand` = the scheduled demand value for Phase 6 from the data.
     - `calc_start` = 469.59 - demand  (because initial condition says Start of Phase Past Due + Scheduled Demand = 469.59, and calc_start for phase 6 is the signed carryover that when added to demand gives the initial backlog condition, i.e., calc_start = 469.59 - demand so that calc_start + demand = 469.59).
     
     **WAIT — re-read the instructions carefully.** The initial condition says: "Initial condition at Phase 6: Start of Phase Past Due + Scheduled Demand = 469.59". This means `Start of Phase Past Due` for Phase 6 = 469.59 - demand_phase6. But `calc_start` for Phase 6 equals `Start of Phase Past Due` for Phase 6 (since there's no prior phase; the initial condition seeds it). So:
     - `start_past_due_phase6 = 469.59 - demand_phase6`
     - `calc_start_phase6 = start_past_due_phase6`
     
     Actually, re-read again: "Calc Start = prior phase End of Phase Backlog/Buffer (Phase 6 starts from the initial condition)." The initial condition is that the total work at Phase 6 start is 469.59. So `calc_start` for Phase 6 should produce: `calc_start + demand = 469.59`, meaning `calc_start = 469.59 - demand`. And `Start of Phase Past Due = max(0, calc_start)`. Let me verify: if calc_start is positive, then start_past_due = calc_start, and total = calc_start + demand = 469.59. If calc_start were negative (unlikely), start_past_due = 0, but we still use calc_start for calculations. This is consistent.

   - For each phase (starting at Phase 6), in order:
     1. `start_past_due = max(0, calc_start)`
     2. Determine `days_worked`:
        - If `start_past_due > 0.01`: choose the smallest value in {5, 6} such that `calc_start + demand - (20 * days_worked) <= 0`. If neither works, choose 6.
        - Else (start_past_due <= 0.01): choose 4 if demand <= 80, else 5.
     3. `weekly_capacity = 20 * days_worked`
     4. `end_backlog = calc_start + demand - weekly_capacity`
     5. `overtime = 10 * max(0, days_worked - 4)`
     6. For the next phase: `calc_start_next = end_backlog` (the signed value).

3. **Write `/root/assembly_plan.xlsx`**: Create a single sheet named `Plan` with exactly these headers in row 1:
   - `Phase`
   - `Days Worked`
   - `Scheduled Demand (Std Hrs)`
   - `Weekly Capacity (Std Hrs)`
   - `Start of Phase Past Due (Std Hrs)`
   - `End of Phase Backlog/Buffer (Std Hrs)`
   - `Overtime Hours`
   
   Write 49 data rows (Phases 6–54 ascending), one per phase. Use openpyxl or xlsxwriter. Ensure numeric values are stored as numbers, not strings.

4. **Write `/root/assembly_summary.txt`**: Exactly 3 lines:
   - `First_Week_5_Days: <phase>` — the first phase where Days Worked == 5 (or `N/A`).
   - `First_Week_4_Days: <phase>` — the first phase where Days Worked == 4 (or `N/A`).
   - `Summary: <text>` — a manager-facing summary, at most 60 words and at most 3 sentences. It must mention both step-down phase numbers (the phase where the plan first drops to 5-day weeks and the phase where it first drops to 4-day weeks), using `N/A` if either is not reached.

5. **Validate**: After generating both files, re-read `/root/assembly_plan.xlsx` and print:
   - Number of rows (should be 49)
   - First 5 rows and last 5 rows of data
   - All unique values of Days Worked
   - Phase 6 values (start_past_due, demand, calc_start, days_worked, end_backlog)
   - Verify that 469.59 == start_past_due_phase6 + demand_phase6 (within tolerance)
   - Print contents of `/root/assembly_summary.txt`

Use `pandas` and `openpyxl` for reading/writing Excel files. Run the script and confirm the output looks correct.

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
Task metadata: author_email=codex@openai.com, author_name=Codex, category=manufacturing-planning, difficulty=medium, tags=[xlsx, operations, capacity-planning, pcb, backlog].
Verifier config: timeout_sec=900.0.