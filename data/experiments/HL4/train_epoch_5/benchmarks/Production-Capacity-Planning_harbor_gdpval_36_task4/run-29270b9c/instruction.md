# Task Instruction

Execute the following steps in order to produce the two deliverables.

## Step 1 – Read the demand data

Open `/root/hvac_demand_sheet.xlsx`, sheet `Install`. Locate the row whose label is `HVAC Ductwork Demand (Std Hrs)`. Extract the demand values for every phase (column) from 8 through 56 inclusive. Store them in a dict/list keyed by phase number. Print them so you can verify.

## Step 2 – Read the existing plan (for reference only)

Open `/root/hvac_existing_plan.xlsx`, sheet `Plan`. Print its contents. You will overwrite this file later; the only thing you keep is the correct header row (verify it matches the seven headers below).

## Step 3 – Compute the schedule (phases 8–56)

Initialize:
- `calc_start` for Phase 8: The initial condition says `Start of Phase Past Due + Scheduled Demand = 1138.66` at Phase 8. This means `calc_start_8 = 1138.66 - demand_8`. But note: `Start of Phase Past Due` at Phase 8 is `max(0, calc_start_8)`, and the initial condition equates that reported past-due plus demand to 1138.66. So if `calc_start_8 >= 0`, then `calc_start_8 + demand_8 = 1138.66` → `calc_start_8 = 1138.66 - demand_8`. If `calc_start_8 < 0`, then `0 + demand_8 = 1138.66` → `demand_8 = 1138.66`, and `calc_start_8` could be anything ≤ 0 — but the simplest consistent interpretation is that `calc_start_8 = 1138.66 - demand_8`.

For each phase p from 8 to 56:

1. `demand_p` = scheduled demand for phase p (from Step 1).
2. If p == 8: `calc_start = 1138.66 - demand_p`. Else: `calc_start = end_of_phase_backlog[p-1]` (the previous phase's End of Phase Backlog/Buffer, signed, i.e., can be negative).
3. `start_past_due = max(0, calc_start)` — for reporting only.
4. Choose `days_worked`:
   - If `start_past_due > 0.01`:
     - Try 5 first: if `calc_start + demand_p - 35*5 <= 0`, choose 5.
     - Else try 6: if `calc_start + demand_p - 35*6 <= 0`, choose 6.
     - Else choose 6 (even if inequality not satisfied).
   - Else (start_past_due <= 0.01):
     - If `demand_p <= 140`: choose 4.
     - Else: choose 5.
5. `weekly_capacity = 35 * days_worked`
6. `end_backlog = calc_start + demand_p - weekly_capacity`
7. `overtime = 10 * max(0, days_worked - 4)`

Store all seven columns per phase.

## Step 4 – Write `/root/hvac_existing_plan.xlsx`

Overwrite the file with a single sheet named `Plan`. Row 1 must have exactly these headers:
```
Phase | Days Worked | Scheduled Demand (Std Hrs) | Weekly Capacity (Std Hrs) | Start of Phase Past Due (Std Hrs) | End of Phase Backlog/Buffer (Std Hrs) | Overtime Hours
```
Rows 2–50 contain phases 8–56 in ascending order (49 data rows). No extra rows.

Use `openpyxl` (or pandas with `ExcelWriter` using openpyxl engine) to write the file. Make sure numeric cells are stored as numbers, not strings. Round floats to 2 decimal places for display but keep full precision in the computation chain.

## Step 5 – Copy to `/root/hvac_schedule_plan.xlsx`

Make an identical copy: `shutil.copy('/root/hvac_existing_plan.xlsx', '/root/hvac_schedule_plan.xlsx')`.

## Step 6 – Determine summary values

- `First_Week_5_Days`: the first phase (lowest number) where `Days Worked == 5`. If none, `N/A`.
- `First_Week_4_Days`: the first phase where `Days Worked == 4`. If none, `N/A`.

## Step 7 – Write `/root/hvac_schedule_summary.txt`

Exactly 3 lines, no trailing blank line:
```
First_Week_5_Days: <value>
First_Week_4_Days: <value>
Summary: <manager-facing summary, ≤60 words, ≤3 sentences, mentioning both step-down phase numbers or N/A>
```

## Step 8 – Validate

1. Re-read `/root/hvac_schedule_plan.xlsx` and print all rows. Confirm 49 data rows, phases 8–56, headers match exactly, numeric types.
2. Re-read `/root/hvac_existing_plan.xlsx` and confirm identical content.
3. Print contents of `/root/hvac_schedule_summary.txt` and confirm format (3 lines, word count ≤ 60, sentence count ≤ 3).
4. Verify Phase 8 initial condition: `start_past_due_8 + demand_8` should equal `1138.66`.
5. Spot-check a few phases' arithmetic.

If any validation fails, fix and re-run from the failing step.

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