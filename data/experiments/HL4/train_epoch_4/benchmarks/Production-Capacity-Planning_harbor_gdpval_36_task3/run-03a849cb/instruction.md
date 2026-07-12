# Task Instruction

Create a Python script `/root/solve.py` and execute it to produce `/root/assembly_plan.xlsx` and `/root/assembly_summary.txt`.

## Step-by-step instructions

### 1. Read the source data
- Open `/root/assembly_schedule.xlsx`, sheet `Assembly`.
- Find the column labeled `PCB Assembly Demand (Std Hrs)` and the `Phase` column.
- Keep only the first occurrence of each phase (drop duplicate phase rows).
- Filter to phases 6 through 54 inclusive (49 phases).
- Store as a dict mapping phase number → scheduled demand (float).

### 2. Implement the deterministic policy

The initial condition states: at Phase 6, `Start of Phase Past Due + Scheduled Demand = 469.59`.

This means for Phase 6:
- `start_past_due = 469.59 - demand[6]`
- `calc_start = start_past_due` (since start_past_due is the carryover into Phase 6)

IMPORTANT: Do NOT set `calc_start = 469.59`. The initial condition tells us the SUM of past-due and demand equals 469.59, so `calc_start` (which equals `start_past_due` for Phase 6) is `469.59 - demand[6]`.

For each phase from 6 to 54 in order:

```
if phase == 6:
    start_past_due = 469.59 - demand[phase]
    calc_start = start_past_due
else:
    calc_start = prev_end_backlog
    start_past_due = max(0, calc_start)
```

Then choose `days_worked`:
```
if start_past_due > 0.01:
    # Try 5 first, then 6
    if calc_start + demand[phase] - (20 * 5) <= 0:
        days_worked = 5
    elif calc_start + demand[phase] - (20 * 6) <= 0:
        days_worked = 6
    else:
        days_worked = 6
else:
    if demand[phase] <= 80:
        days_worked = 4
    else:
        days_worked = 5
```

Then compute:
```
weekly_capacity = 20 * days_worked
end_backlog = calc_start + demand[phase] - weekly_capacity
overtime = 10 * max(0, days_worked - 4)
```

Store `prev_end_backlog = end_backlog` for the next phase.

### 3. Write `/root/assembly_plan.xlsx`
- Use openpyxl to create a workbook with a single sheet named `Plan`.
- Row 1 headers (exactly, in order):
  1. `Phase`
  2. `Days Worked`
  3. `Scheduled Demand (Std Hrs)`
  4. `Weekly Capacity (Std Hrs)`
  5. `Start of Phase Past Due (Std Hrs)`
  6. `End of Phase Backlog/Buffer (Std Hrs)`
  7. `Overtime Hours`
- 49 data rows (phases 6–54), ascending order, no gaps, no duplicates.
- Write numeric values (not strings). Round floats to 2 decimal places for display.

### 4. Write `/root/assembly_summary.txt`
- Scan the results to find:
  - `First_Week_5_Days`: the first phase where `days_worked == 5` (among phases that had a transition, i.e., after being at 6). More precisely, just find the first phase where days_worked is exactly 5.
  - `First_Week_4_Days`: the first phase where `days_worked == 4`.
  - Use `N/A` if either never occurs.
- Write exactly 3 lines:
  ```
  First_Week_5_Days: <phase-number-or-N/A>
  First_Week_4_Days: <phase-number-or-N/A>
  Summary: <manager-facing summary, ≤60 words, ≤3 sentences, mentioning both step-down phase numbers or N/A>
  ```
- No trailing newline after the third line (or a single trailing newline is fine, but no blank lines).

### 5. Validation
- After generating, re-read `/root/assembly_plan.xlsx` and print:
  - Number of data rows (must be 49)
  - The headers (must match exactly)
  - First 3 rows and last row of data
  - Phase 6: verify that `start_past_due + demand == 469.59` (within tolerance)
- Read `/root/assembly_summary.txt` and print its contents.
- Confirm no duplicate phases, phases are 6–54 contiguous.

### 6. Run the script
```bash
cd /root && python solve.py
```

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