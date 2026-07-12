# Task Instruction

Perform the following steps to complete the task:

## Step 0: Inspect the workbook
1. Copy `/root/data/workbook.xlsx` to `/root/output/result.xlsx`.
2. Using `openpyxl`, open `/root/output/result.xlsx` and inspect:
   - Sheet `Task`: read the layout of rows 10-50, especially columns D and H-L. Print row 10 (the year headers in H10:L10), column D rows 12-31 (the series codes), rows 35-40 column D (region names or codes for Net reliability gap), rows 42-47 column A or D (stat labels), and row 50.
   - Sheet `Data`: read rows 21-38 to understand the data layout (which row has which series, which columns have which years).
   - Note the exact column letters/numbers used on the Data sheet so formulas can reference them correctly.

## Step 1: Populate lookup formulas in H12:L17, H19:L24, H26:L31

For each cell in these ranges, write a formula that looks up the value from sheet `Data` rows 21:38 using:
- The series code from column D of that row on the Task sheet
- The year from row 10 of that column on the Task sheet

Use `INDEX(MATCH,MATCH)` pattern. The formula pattern should be something like:
```
=INDEX(Data!$B$21:$XX$38, MATCH($D12, Data!$A$21:$A$38, 0), MATCH(H$10, Data!$B$20:$XX$20, 0))
```
Adjust the exact ranges after inspecting the Data sheet layout (identify which column contains the series codes and which row contains the year headers, and what the data range is).

IMPORTANT: Use absolute references for the data range and series code column on Data sheet; use mixed references ($D12 for the series code column, H$10 for the year row) so formulas can be filled across the range.

## Step 2: Net reliability gap in H35:L40 and statistics in H42:L47

For H35:L40, write formulas that calculate:
`(Successful API Requests - Failed API Requests) / Compute Capacity * 100`

You need to identify which rows in the Task sheet contain "Successful API Requests", "Failed API Requests", and "Compute Capacity" for each region. Based on the three blocks:
- H12:L17 is likely one metric (6 regions)
- H19:L24 is likely another metric (6 regions)
- H26:L31 is likely another metric (6 regions)

After inspecting, construct the formula. For example, if rows 12-17 are Successful API Requests, rows 19-24 are Failed API Requests, and rows 26-31 are Compute Capacity, then:
`H35 = (H12 - H19) / H26 * 100`

Adjust based on actual inspection.

For H42:L47, write column-wise statistics of H35:L40:
- H42: `=MIN(H35:H40)`
- H43: `=MAX(H35:H40)`
- H44: `=MEDIAN(H35:H40)`
- H45: `=AVERAGE(H35:H40)`
- H46: `=PERCENTILE(H35:H40, 0.25)`
- H47: `=PERCENTILE(H35:H40, 0.75)`

Check the labels in column A/D for rows 42-47 to confirm the order (min, max, median, mean, 25th, 75th).

## Step 3: Weighted mean in H50:L50

For each column (H through L), write:
`=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)`

This calculates the weighted mean of the Net reliability gap percentages weighted by Compute Capacity.

## Final Steps
1. Save the workbook.
2. Re-open and verify that:
   - Formulas are present in the expected cells
   - No extra sheets were added
   - The file is saved at `/root/output/result.xlsx`

## Critical Notes
- Use `openpyxl` for all operations.
- When writing formulas, use `cell.value = '=FORMULA...'` (string starting with `=`).
- Do NOT use `data_only=True` when opening for writing.
- Preserve all existing formatting, merged cells, styles.
- Do not delete or add any sheets.
- After inspection, print your findings clearly before writing formulas so you can verify correctness.

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
Task metadata: author_email=catpaw@meituan.com, author_name=CatPaw Task Engineer, category=spreadsheet-formula-reuse, difficulty=easy, tags=[excel, formulas, lookup, statistics, weighted-mean].
Verifier config: timeout_sec=600.0.