# Task Instruction

Execute the following steps in order:

## Step 1: Inspect the source data

Open `/root/copy_of_capacity_sheet.xlsx`, sheet `Weld`. Identify the row labeled `MIG weld Demand Total`. Extract the scheduled demand values for weeks 4 through 52 (columns corresponding to those weeks). Print the first few and last few values to confirm correct extraction.

## Step 2: Implement the catch-up plan logic in Python

Write a Python script that:

1. Reads the `MIG weld Demand Total` row from the `Weld` sheet for weeks 4–52.
2. Initializes Week 4 with `Calc Start` such that `Calc Start + Scheduled Demand for Week 4 = 438.81`, i.e., `Calc Start = 438.81 - Demand[Week4]`.
   - IMPORTANT: The initial condition says "Start of Week Past Due + Scheduled Demand = 438.81" at Week 4. This means the `Calc Start` for Week 4 = 438.81 - Week4's Scheduled Demand. The `Start of Week Past Due` for Week 4 = max(0, Calc Start).
   - Wait — re-read the instruction more carefully: "Initial condition at Week 4: Start of Week Past Due + Scheduled Demand = 438.81". And the policy says "Calc Start = prior week End of Week Backlog/Buffer (Week 4 starts from the initial condition)". So for Week 4, Calc Start is derived from the initial condition. Since there's no prior week, interpret: Calc Start for Week 4 = 438.81 - Scheduled Demand for Week 4? No — re-read again.
   - Actually the simplest reading: at Week 4, the total work to be done = 438.81 std hrs. That is: Start of Week Past Due + Scheduled Demand = 438.81. The Calc Start for Week 4 should be set so that Calc Start + Scheduled Demand = 438.81, meaning Calc Start = 438.81 - Demand[Week4]. And Start of Week Past Due = max(0, Calc Start).

3. For each week 4..52 in order, compute:
   - `Start of Week Past Due = max(0, Calc Start)` (for reporting)
   - Choose `Days Worked`:
     - If `Start of Week Past Due > 0.01`:
       - Try days=5: if `Calc Start + Demand - 30*5 <= 0`, use 5
       - Else try days=6: if `Calc Start + Demand - 30*6 <= 0`, use 6
       - Else use 6
     - Else (Past Due <= 0.01):
       - If `Demand <= 120`, use 4; else use 5
   - `Weekly Capacity = 30 * Days Worked`
   - `End of Week Backlog/Buffer = Calc Start + Demand - Weekly Capacity`
   - `Overtime Hours = 10 * max(0, Days Worked - 4)`
   - Next week's `Calc Start = End of Week Backlog/Buffer` (the signed value, not clamped)

4. Track `First_Week_5_Days` = the first week where Days Worked is exactly 5 (considering the step-down from 6 to 5, or the first occurrence of 5).
   - Actually, re-read: the summary asks for step-down week numbers. `First_Week_5_Days` likely means the first week the plan steps down to 5 days (from 6). `First_Week_4_Days` means the first week it steps down to 4 days (from 5 or 6). If a value never occurs, use `N/A`.
   - More precisely: find the first week where Days Worked == 5 (this is the step-down from 6 to 5), and the first week where Days Worked == 4 (step-down to normal). Report these week numbers.

## Step 3: Write `/root/catch_up_plan.xlsx`

Using openpyxl, create a workbook with a single sheet named `Plan`. Row 1 has exactly these headers:
- `Week`
- `Days Worked`
- `Scheduled Demand (Std Hrs)`
- `Weekly Capacity (Std Hrs)`
- `Start of Week Past Due (Std Hrs)`
- `End of Week Backlog/Buffer (Std Hrs)`
- `Overtime Hours`

Rows 2–50 contain weeks 4–52 (49 data rows), in ascending order, no gaps, no duplicates. Values should be numeric (not strings). Round floating point values to 2 decimal places for cleanliness.

## Step 4: Write `/root/catch_up_summary.txt`

Exactly 3 lines:
```
First_Week_5_Days: <week>
First_Week_4_Days: <week>
Summary: <text>
```
The summary must be ≤60 words, ≤3 sentences, and mention both step-down week numbers (or N/A).

## Step 5: Validate

1. Re-read `/root/catch_up_plan.xlsx` and print the first 5 rows and last 3 rows to confirm correctness.
2. Verify 49 data rows exist.
3. Verify Week column goes 4..52 with no gaps.
4. Verify all Days Worked values are in {4, 5, 6}.
5. Verify End of Week Backlog/Buffer for Week 4 = 438.81 - Demand[4] - 30*DaysWorked[4] + (438.81 - Demand[4]) ... actually just spot-check Week 4 manually.
6. Print `/root/catch_up_summary.txt` contents and verify format (3 lines, correct prefixes, word count ≤60).
7. Confirm no extra sheets in the workbook and sheet name is exactly `Plan`.

## Important Notes
- Use `openpyxl` for both reading and writing Excel files.
- When reading the source file, be careful about how weeks map to columns — inspect the header row to find the column indices for weeks 4–52.
- The `MIG weld Demand Total` row label might have slight formatting differences; search for a row containing that text.
- Do NOT leave any file handles open; close workbooks after use.
- If any demand value is missing or zero for a week, treat it as 0.

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
Task metadata: author_email=codex@openai.com, author_name=Codex, category=manufacturing-planning, difficulty=medium, tags=[xlsx, operations, capacity-planning, backlog].
Verifier config: timeout_sec=900.0.