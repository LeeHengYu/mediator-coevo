# Task Instruction

Execute the following steps in order to produce the two deliverables for the glass furnace catch-up plan.

## Step 1 – Inspect the source data

1. Open `/root/glass_demand_sheet.xlsx`, sheet `Glass`.
2. Locate the row labeled `Glass Furnace Demand (Std Hrs)`.
3. Extract the demand values for Weeks 2 through 50 (49 values). Print them so you can verify they were read correctly.

## Step 2 – Compute the plan using the deterministic policy

Write a Python script (using `openpyxl` and/or `pandas`) that does the following:

### Initial condition for Week 2
- The problem states: `Start of Week Past Due + Scheduled Demand = 910.80` at Week 2.
- Therefore for Week 2: `Calc Start = 910.80 - Demand[Week2]`.
- `Start of Week Past Due (Week 2) = max(0, Calc Start)`.

### For each week w = 2..50 (in order):

1. **Scheduled Demand** = demand value read from the spreadsheet for week w.
2. **Calc Start**:
   - Week 2: `910.80 - Demand[Week2]`
   - Week w > 2: `End of Week Backlog/Buffer` from week w-1 (signed, can be negative).
3. **Start of Week Past Due (Std Hrs)** = `max(0, Calc Start)` — for display/reporting only.
4. **Days Worked** decision:
   - If `Start of Week Past Due > 0.01`:
     - Try 5: if `Calc Start + Demand - 22*5 <= 0`, choose 5.
     - Else try 6: if `Calc Start + Demand - 22*6 <= 0`, choose 6.
     - Else (neither clears it): choose 6.
   - Else (`Start of Week Past Due <= 0.01`):
     - If `Demand <= 110`: choose 4.
     - Else: choose 5.
5. **Weekly Capacity (Std Hrs)** = `22 * Days Worked`.
6. **End of Week Backlog/Buffer (Std Hrs)** = `Calc Start + Demand - Weekly Capacity`.
7. **Overtime Hours** = `10 * max(0, Days Worked - 4)`.

Store all 49 rows of results.

## Step 3 – Identify transition weeks

- `First_Week_5_Days`: the first week where Days Worked drops from 6 to 5 (i.e., the first week with Days Worked == 5). If no week has 5, output `N/A`.
- `First_Week_4_Days`: the first week where Days Worked == 4. If no week has 4, output `N/A`.

## Step 4 – Write `/root/glass_furnace_plan.xlsx`

Create a workbook with a single sheet named `Plan`. Row 1 must have exactly these headers (in this order, verbatim):

1. `Week`
2. `Days Worked`
3. `Scheduled Demand (Std Hrs)`
4. `Weekly Capacity (Std Hrs)`
5. `Start of Week Past Due (Std Hrs)`
6. `End of Week Backlog/Buffer (Std Hrs)`
7. `Overtime Hours`

Rows 2–50 contain the data for Weeks 2..50 in ascending order. All numeric values should be stored as numbers (not strings). Round floats to 2 decimal places where appropriate.

## Step 5 – Write `/root/glass_furnace_summary.txt`

Exactly 3 lines, no trailing blank lines:

```
First_Week_5_Days: <week-number-or-N/A>
First_Week_4_Days: <week-number-or-N/A>
Summary: <manager-facing summary, ≤60 words, ≤3 sentences, mentioning both step-down week numbers or N/A>
```

Based on prior successful execution, expect Week 2 for the 5-day step-down and Week 36 for the 4-day step-down, but compute them fresh from the data.

## Step 6 – Validate

1. Re-read `/root/glass_furnace_plan.xlsx` and print the first 5 rows and last 3 rows to confirm correctness.
2. Confirm the sheet name is exactly `Plan`.
3. Confirm there are exactly 49 data rows (Weeks 2–50).
4. Confirm all 7 headers match exactly.
5. Print the contents of `/root/glass_furnace_summary.txt`.
6. Verify the summary is ≤ 3 lines, the Summary line is ≤ 60 words and ≤ 3 sentences.
7. If any check fails, fix and re-validate before finishing.

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
Task metadata: author_email=codex@openai.com, author_name=Codex, category=manufacturing-planning, difficulty=medium, tags=[xlsx, operations, capacity-planning, glass, backlog].
Verifier config: timeout_sec=900.0.