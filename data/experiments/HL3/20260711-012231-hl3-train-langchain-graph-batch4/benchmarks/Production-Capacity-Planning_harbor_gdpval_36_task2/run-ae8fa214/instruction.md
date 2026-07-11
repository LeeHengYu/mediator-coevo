# Task Instruction

Create a Python script `/root/solve.py` and execute it to produce both deliverables. Follow these steps precisely:

## Step 1: Read the input data

```python
import openpyxl
from openpyxl import Workbook

wb = openpyxl.load_workbook('/root/mill_demand_sheet.xlsx', data_only=True)
ws = wb['Mill']
```

Scan the sheet to find the row whose first non-empty cell contains the text `CNC Mill Demand (Hrs)` (case-insensitive, stripped). Then extract the demand values for Periods 1–52. The periods are typically in columns after the label column. Print the row content and the number of demand values found to verify you have exactly 52 values. If the sheet layout is ambiguous, print the first 5 rows and first 60 columns to understand the structure before proceeding.

## Step 2: Implement the deterministic simulation

Variables:
- `demands`: list of 52 floats (Scheduled Demand for periods 1..52)
- Initial condition: Period 1's `Calc Start + Scheduled Demand = 538.08`, so `Calc Start for Period 1 = 538.08 - demands[0]`

For each period `t` (1..52):
1. If `t == 1`: `calc_start = 538.08 - demands[0]` and then `past_due = max(0, calc_start)`, `calc_start_with_demand = 538.08` (i.e., `calc_start + demands[0]`).
   Actually, let me be precise:
   - `calc_start = 538.08 - demands[0]`  (this is the signed carryover before period 1)
   - `start_past_due = max(0, calc_start)`
   - `scheduled_demand = demands[0]`
   - Then choose Days Worked using the policy below with `calc_start` and `scheduled_demand`.
2. If `t > 1`: `calc_start = end_backlog[t-2]` (prior period's End of Period Backlog/Buffer), `start_past_due = max(0, calc_start)`, `scheduled_demand = demands[t-1]`.

Days Worked policy:
- If `start_past_due > 0.01`:
  - Try 5: if `calc_start + scheduled_demand - 125 <= 0`, choose 5.
  - Else try 6: if `calc_start + scheduled_demand - 150 <= 0`, choose 6.
  - Else choose 6.
- Else (`start_past_due <= 0.01`):
  - If `scheduled_demand <= 125`, choose 4. Else choose 5.

Then:
- `weekly_capacity = 25 * days_worked`
- `end_backlog = calc_start + scheduled_demand - weekly_capacity`
- `overtime = 10 * max(0, days_worked - 4)`

Store all results.

## Step 3: Write `/root/mill_catch_up_plan.xlsx`

Create a workbook with a single sheet named `Plan`. Row 1 headers exactly:
`Period`, `Days Worked`, `Scheduled Demand (Std Hrs)`, `Weekly Capacity (Std Hrs)`, `Start of Period Past Due (Std Hrs)`, `End of Period Backlog/Buffer (Std Hrs)`, `Overtime Hours`

Then 52 data rows (periods 1–52). Use numeric types (int for Period/Days Worked/Overtime Hours, float for the rest). Save to `/root/mill_catch_up_plan.xlsx`.

## Step 4: Write `/root/mill_catch_up_summary.txt`

Scan the results to find:
- `First_Week_5_Days`: the first period where Days Worked is exactly 5 (could be period 1 if backlog forces it). If none, `N/A`.
- `First_Week_4_Days`: the first period where Days Worked is exactly 4. If none, `N/A`.

Write exactly 3 lines:
```
First_Week_5_Days: <value>
First_Week_4_Days: <value>
Summary: <manager-facing summary, ≤60 words, ≤3 sentences, mentioning both step-down period numbers or N/A>
```

## Step 5: Validate

After generating both files:
1. Re-read the Excel file and print the first 5 and last 3 rows to verify correctness.
2. Print the contents of the summary text file.
3. Verify Period 1: `Start of Period Past Due + Scheduled Demand` should equal `538.08`.
4. Count total rows (should be 52), verify no gaps/duplicates in Period column.
5. Print the sum of all Overtime Hours and the period where backlog first goes to 0 or negative.

Run the script with `python3 /root/solve.py` and confirm both output files are correct.

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
Task metadata: author_email=codex@openai.com, author_name=Codex, category=manufacturing-planning, difficulty=medium, tags=[xlsx, operations, capacity-planning, cnc, backlog].
Verifier config: timeout_sec=900.0.