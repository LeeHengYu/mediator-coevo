# Task Instruction

You are an HVAC installation supervisor building a staffing schedule for ductwork crew capacity.

## Step-by-step plan

### 1. Inspect input files

```bash
cd /root
python3 -c "
import openpyxl
wb = openpyxl.load_workbook('hvac_demand_sheet.xlsx', data_only=True)
print('Sheets:', wb.sheetnames)
ws = wb['Install']
for row in ws.iter_rows(min_row=1, max_row=ws.max_row, max_col=ws.max_column, values_only=False):
    vals = [(c.value, c.column) for c in row]
    print(vals)
"
```

Also inspect the existing plan:
```bash
python3 -c "
import openpyxl
wb = openpyxl.load_workbook('hvac_existing_plan.xlsx', data_only=True)
print('Sheets:', wb.sheetnames)
ws = wb['Plan']
for row in ws.iter_rows(min_row=1, max_row=min(ws.max_row, 25), max_col=ws.max_column, values_only=True):
    print(row)
"
```

Goal: identify which row contains `HVAC Ductwork Demand (Std Hrs)` and extract demand values for phases 8–56. The phases are likely column headers in some row, and the demand row has values beneath them.

### 2. Extract demand data and compute the schedule

Write a Python script `/root/build_schedule.py` that:

a) Reads `hvac_demand_sheet.xlsx` sheet `Install` using openpyxl with `data_only=True`.
b) Locates the row whose first non-empty cell contains (case-insensitive) `HVAC Ductwork Demand` or similar.
c) Identifies column positions for phases 8 through 56 (look in a header row for integer values 8..56, or use adjacent row structure).
d) Extracts scheduled demand for each phase.
e) Implements the deterministic policy:

```
Phases: 8 through 56 inclusive (49 phases)
Initial condition: At Phase 8, Calc Start + Scheduled Demand = 1138.66
  => Calc Start for Phase 8 = 1138.66 - Scheduled_Demand[Phase 8]
  Wait — re-read: "Start of Phase Past Due + Scheduled Demand = 1138.66" at Phase 8.
  This means the INITIAL Calc Start (prior phase End of Phase Backlog/Buffer) is such that
  Start of Phase Past Due = max(0, prior_backlog) and
  Start of Phase Past Due + Scheduled Demand = 1138.66.
  So Start of Phase Past Due for Phase 8 = 1138.66 - Scheduled_Demand[Phase 8].
  And Calc Start for Phase 8 = that same value (since it's the first phase, and past due > 0 means prior backlog was positive).
  Actually: Calc Start = prior phase End of Phase Backlog/Buffer. For Phase 8, there is no prior phase.
  The initial condition says Start of Phase Past Due + Scheduled Demand = 1138.66.
  Start of Phase Past Due = max(0, prior_backlog). Since this is the first phase, prior_backlog is the initial backlog.
  So prior_backlog = 1138.66 - Scheduled_Demand[Phase 8] (assuming it's positive).
  And Calc Start for Phase 8 = prior_backlog = 1138.66 - Scheduled_Demand[Phase 8].
```

For each phase i (starting at Phase 8):
1. `start_past_due = max(0, prior_backlog)`  — for reporting
2. `calc_start = prior_backlog`  — the signed value
3. Choose `days_worked`:
   - If `start_past_due > 0.01`:
     - Try 5: if `calc_start + demand - 35*5 <= 0`, use 5
     - Else try 6: if `calc_start + demand - 35*6 <= 0`, use 6
     - Else use 6
   - Else (start_past_due <= 0.01):
     - If `demand <= 140`: days = 4
     - Else: days = 5
4. `weekly_capacity = 35 * days_worked`
5. `end_backlog = calc_start + demand - weekly_capacity`
6. `overtime = 10 * max(0, days_worked - 4)`
7. Set `prior_backlog = end_backlog` for next phase.

f) Collect results into a list of dicts.

### 3. Write output files

Using openpyxl:

a) Create `/root/hvac_existing_plan.xlsx` with sheet named `Plan`, headers in row 1:
   `Phase`, `Days Worked`, `Scheduled Demand (Std Hrs)`, `Weekly Capacity (Std Hrs)`, `Start of Phase Past Due (Std Hrs)`, `End of Phase Backlog/Buffer (Std Hrs)`, `Overtime Hours`
   Then 49 data rows (phases 8..56), ascending order.

b) Copy that file to `/root/hvac_schedule_plan.xlsx` (use shutil.copy or write identically).

c) Create `/root/hvac_schedule_summary.txt` with exactly 3 lines:
   - `First_Week_5_Days: <phase>` — the first phase where Days Worked = 5 (or N/A)
   - `First_Week_4_Days: <phase>` — the first phase where Days Worked = 4 (or N/A)
   - `Summary: <text>` — a manager-facing summary, ≤60 words, ≤3 sentences, mentioning both step-down phase numbers.

### 4. Validate

After writing, re-read both xlsx files and the txt file to confirm:
- Sheet name is `Plan`
- Headers match exactly
- 49 data rows, phases 8–56
- Phase 8 row: `Start of Phase Past Due + Scheduled Demand == 1138.66` (within tolerance)
- All Days Worked values are in {4, 5, 6}
- End of Phase Backlog/Buffer = Calc Start + Demand - Capacity for each row
- Overtime = 10 * max(0, days - 4)
- Summary file has exactly 3 lines with correct prefixes
- Summary ≤ 60 words, ≤ 3 sentences

Print validation results.

### Important notes
- Use `data_only=True` when reading the demand sheet to get computed values, not formulas.
- Round numeric values to 2 decimal places for display but keep full precision in calculations.
- The initial condition interpretation: `prior_backlog_before_phase8 = 1138.66 - demand[phase8]`. This makes `start_past_due[phase8] = max(0, prior_backlog_before_phase8)` and `start_past_due[phase8] + demand[phase8] = 1138.66` when prior_backlog is positive.
- If demand values appear as None for some phases, check if the sheet uses merged cells or different layout, and adapt accordingly.
- Store numeric values as numbers (int or float) in Excel cells, not strings.

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