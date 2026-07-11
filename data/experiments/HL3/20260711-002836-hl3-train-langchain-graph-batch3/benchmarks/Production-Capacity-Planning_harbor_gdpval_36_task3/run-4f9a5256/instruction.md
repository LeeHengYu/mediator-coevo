# Task Instruction

Execute the following steps in order to produce `/root/assembly_plan.xlsx` and `/root/assembly_summary.txt`.

---

### Step 1 – Inspect the source workbook

```python
import openpyxl
wb = openpyxl.load_workbook('/root/assembly_schedule.xlsx', data_only=True)
print(wb.sheetnames)
ws = wb['Assembly']
for row in ws.iter_rows(min_row=1, max_row=min(ws.max_row, 70), values_only=False):
    print([(c.column, c.value) for c in row])
```

Identify:
- Which column holds the phase number.
- Which column is labeled `PCB Assembly Demand (Std Hrs)`.
- Whether there are duplicate phase rows (keep only the first occurrence of each phase).

---

### Step 2 – Build the plan data with Python

Using the information from Step 1, write and run a single Python script that:

1. Reads the `Assembly` sheet from `/root/assembly_schedule.xlsx` using `openpyxl` (data_only=True).
2. Extracts `(phase, demand)` pairs for phases 6–54 inclusive, keeping only the **first** occurrence of each phase.
3. Implements the deterministic policy described below **exactly**:

```
Phases: 6 through 54 inclusive, processed in ascending order.

Initial condition for Phase 6:
  calc_start = 469.59   # this is "Calc Start" for phase 6
                         # (it equals Start of Phase Past Due + Scheduled Demand... 
                         #  NO — re-read: "Initial condition at Phase 6: 
                         #  Start of Phase Past Due + Scheduled Demand = 469.59")
  Actually this means:  past_due_6 + demand_6 = 469.59
  So calc_start_6 = past_due_6 + demand_6 ... but we need to figure out past_due_6.
  Wait — let me re-read more carefully.

  "Calc Start = prior phase End of Phase Backlog/Buffer"
  "Phase 6 starts from the initial condition."
  And "Start of Phase Past Due + Scheduled Demand = 469.59" for Phase 6.

  The initial condition tells us the total work to clear in Phase 6:
    Start of Phase Past Due (phase 6) + Scheduled Demand (phase 6) = 469.59
  
  For Phase 6:
    calc_start = <some prior backlog that equals past_due_6>
    past_due_6 = max(0, calc_start)  ... but calc_start IS the prior backlog
    So: calc_start_6 + demand_6 = 469.59  →  calc_start_6 = 469.59 - demand_6
    Also: past_due_6 = max(0, calc_start_6)
```

**IMPORTANT**: Derive `calc_start` for Phase 6 as `469.59 - demand_phase6`. For all subsequent phases, `calc_start = prior phase End of Phase Backlog/Buffer`.

For each phase i (6..54):
```
past_due_i = max(0, calc_start_i)          # for reporting

if past_due_i > 0.01:
    # try days_worked = 5 first
    if calc_start_i + demand_i - (20 * 5) <= 0:
        days_worked = 5
    elif calc_start_i + demand_i - (20 * 6) <= 0:
        days_worked = 6
    else:
        days_worked = 6
else:
    if demand_i <= 80:
        days_worked = 4
    else:
        days_worked = 5

capacity_i = 20 * days_worked
end_backlog_i = calc_start_i + demand_i - capacity_i
overtime_i = 10 * max(0, days_worked - 4)

# next phase:
calc_start_{i+1} = end_backlog_i
```

4. Writes `/root/assembly_plan.xlsx` with a single sheet named `Plan`.
   - Row 1 headers (exactly, in order):
     `Phase`, `Days Worked`, `Scheduled Demand (Std Hrs)`, `Weekly Capacity (Std Hrs)`, `Start of Phase Past Due (Std Hrs)`, `End of Phase Backlog/Buffer (Std Hrs)`, `Overtime Hours`
   - Rows 2–50: one row per phase 6..54 (49 rows), ascending order, no gaps, no duplicates.
   - All numeric values stored as numbers (int or float), not strings.

5. Determines:
   - `first_5_day`: the first phase where `Days Worked == 5` (report phase number, or `N/A`).
   - `first_4_day`: the first phase where `Days Worked == 4` (report phase number, or `N/A`).

6. Writes `/root/assembly_summary.txt` with exactly 3 lines:
   ```
   First_Week_5_Days: <phase-number-or-N/A>
   First_Week_4_Days: <phase-number-or-N/A>
   Summary: <manager-facing summary, ≤60 words, ≤3 sentences, mentioning both step-down phase numbers or N/A>
   ```

---

### Step 3 – Validate

1. Re-open `/root/assembly_plan.xlsx` and print:
   - The header row.
   - The first 5 data rows and last 3 data rows.
   - Total row count (should be 49 data rows).
   - Confirm Phase column goes 6..54 with no gaps or duplicates.
2. Print the contents of `/root/assembly_summary.txt`.
3. Verify:
   - Phase 6 `calc_start + demand = 469.59`.
   - All `Days Worked` values are in {4, 5, 6}.
   - `End of Phase Backlog/Buffer` = `calc_start + demand - capacity` for every row.
   - `Overtime Hours` = `10 * max(0, days_worked - 4)` for every row.
   - Summary line is ≤ 60 words and ≤ 3 sentences.

If any validation fails, fix and re-run.

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