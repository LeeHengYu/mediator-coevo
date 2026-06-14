# Task Instruction

Execute the following two-phase plan to produce /root/output/result.xlsx.

## Phase 0 – Inspect the workbook

1. `mkdir -p /root/output`
2. Open `/root/data/workbook.xlsx` with openpyxl (data_only=False) and inspect:
   a. Sheet names.
   b. **Task sheet**: Print rows 10-11 (to see year headers in H10:L10), rows 12-31 column D (series codes), rows 35-50 labels, and any existing content/formulas in the yellow target ranges.
   c. **Data sheet**: Print rows 21-38 entirely (all columns) to see how series codes and year data are laid out. Note which row contains headers (series codes) and which column contains years, or vice-versa. Identify the exact layout: are series codes in a column and years across columns, or years in a column and series codes across columns?
   d. Print your findings before proceeding.

## Phase 1 – Write formulas

Using the layout discovered in Phase 0, write a Python script (openpyxl, data_only=False) that:

### Step 1 – Lookup formulas in H12:L17, H19:L24, H26:L31

For each cell in these three 6-row × 5-column blocks, insert an `INDEX/MATCH` formula that:
- Uses the series code from column D of the same row on the Task sheet.
- Uses the year from row 10 of the same column on the Task sheet.
- Looks up the value from the Data sheet rows 21:38.
- The exact INDEX/MATCH pattern depends on the Data sheet layout discovered in Phase 0. If Data has series codes in a column (say column A or B) and years in a header row, use:
  `=INDEX(Data!<data_range>,MATCH(D{row},Data!<series_code_column>,0),MATCH({col}10,Data!<year_row>,0))`
  Adjust references based on actual layout.

### Step 2 – Net reliability gap (H35:L40)

For each of the 6 regions (rows 35-40) and 5 year-columns (H-L):
- Identify which rows in the Step 1 blocks correspond to "Successful API Requests", "Failed API Requests", and "Compute Capacity". These should be in the three blocks H12:L17, H19:L24, H26:L31 respectively (check the labels in column D or nearby to confirm).
- Insert formula: `=(<Successful_cell> - <Failed_cell>) / <Capacity_cell> * 100`
  where Successful is from the first block, Failed from the second block, and Capacity from the third block, matching the same region (same relative row position within each block) and same column.

### Step 2 continued – Summary statistics (H42:L47)

For each year-column H through L, in rows 42-47 insert:
- Row 42: `=MIN(H35:H40)` (adjust column letter)
- Row 43: `=MAX(H35:H40)`
- Row 44: `=MEDIAN(H35:H40)`
- Row 45: `=AVERAGE(H35:H40)`
- Row 46: `=PERCENTILE(H35:H40,0.25)`
- Row 47: `=PERCENTILE(H35:H40,0.75)`

Check the labels in column D for rows 42-47 to confirm the correct order (min, max, median, mean, 25th, 75th). Adjust row assignments if the labels differ.

### Step 3 – Weighted mean (H50:L50)

For each year-column H through L:
`=SUMPRODUCT(H35:H40,H26:H31)/SUM(H26:H31)`

This computes the weighted mean of the Net reliability gap values (H35:H40) weighted by Compute Capacity (H26:L31).

### Save

Save the workbook to `/root/output/result.xlsx`. Do NOT change formatting, do NOT add sheets.

## Phase 2 – Validate

1. Reopen `/root/output/result.xlsx` with openpyxl (data_only=False).
2. Print formulas in cells H12, L17, H19, L24, H26, L31, H35, L40, H42, H47, H50, L50.
3. Confirm all target cells contain formulas (strings starting with '='), not None or bare values.
4. If any cell is None or missing a formula, diagnose and fix before finishing.

## Critical Notes
- Use `data_only=False` when loading so existing formulas are preserved.
- Do not overwrite any cells outside the specified target ranges.
- Do not add macros, VBA, external links, helper tabs, or new sheets.
- The avoid-artifact from weighted-hospital-bedflow-calc failed because target cells were left empty (None). Ensure every target cell gets a formula.

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