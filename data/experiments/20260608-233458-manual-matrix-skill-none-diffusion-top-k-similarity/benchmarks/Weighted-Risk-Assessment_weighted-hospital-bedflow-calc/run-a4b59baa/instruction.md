# Task Instruction

## Task: Weighted Hospital Bedflow Calculation

You need to update `/root/data/workbook.xlsx` by populating formulas in the `Task` sheet, then save the result to `/root/output/result.xlsx`.

### Step 0: Inspect the workbook
1. Create `/root/output/` directory if it doesn't exist.
2. Open `/root/data/workbook.xlsx` using openpyxl (with `data_only=False` so you can write formulas).
3. Inspect the `Task` sheet carefully:
   - Read row 10 to find the years in columns H through L.
   - Read column D for rows 12-17, 19-24, 26-31 to find the series codes.
   - Read the structure of rows 35-40 (hospital names, any labels).
   - Read rows 42-47 labels (min, max, median, mean, 25th percentile, 75th percentile).
   - Read row 50 label.
4. Inspect the `Data` sheet:
   - Read rows 21-38 to understand the data layout: which row has which column headers, where series codes are, where years are, and how data is organized.
   - Determine whether data is arranged with series codes in a column and years across columns, or vice versa.
5. Print all findings so you understand the exact layout before writing any formulas.

### Step 1: Populate lookup formulas in H12:L17, H19:L24, H26:L31

For each cell in these ranges, write a spreadsheet formula that looks up data from `Data!$21:$38` using:
- The series code from column D of the current row on the `Task` sheet
- The year from row 10 of the `Task` sheet

Use `INDEX`/`MATCH` (most reliable pattern). The exact formula structure depends on the Data sheet layout you discovered in Step 0. For example, if the Data sheet has series codes in a column (say column A) and years in a header row (say row 21), the formula pattern would be:
```
=INDEX(Data!$B$22:$ZZ$38, MATCH($D12, Data!$A$22:$A$38, 0), MATCH(H$10, Data!$B$21:$ZZ$21, 0))
```
Adjust the exact ranges based on what you find in the Data sheet. The key contract:
- First MATCH finds the row by series code (column D of current row, absolute column reference with $D)
- Second MATCH finds the column by year (row 10 of Task sheet, absolute row reference with $10)
- INDEX returns the intersection

IMPORTANT: Use absolute references where needed so formulas work correctly across the range. Column D reference should be `$D` (absolute column), row 10 reference should be `$10` (absolute row).

### Step 2: Net patient flow and summary statistics in H35:L40 and H42:L47

For H35:L40, calculate Net patient flow for each hospital:
```
=(H12 - H19) / H26 * 100
```
where row 12-17 = Patient Admissions, row 19-24 = Patient Discharges, row 26-31 = Effective Bed Capacity. Match each hospital row correctly (row 35 uses rows 12, 19, 26; row 36 uses rows 13, 20, 27; etc.).

For H42:L47, calculate column-wise statistics. Based on the labels in column D/E for rows 42-47, assign:
- Minimum: `=MIN(H35:H40)` (or `=MIN(H$35:H$40)`)
- Maximum: `=MAX(H35:H40)`
- Median: `=MEDIAN(H35:H40)`
- Simple Mean: `=AVERAGE(H35:H40)`
- 25th Percentile: `=PERCENTILE(H35:H40, 0.25)` or `=PERCENTILE.INC(H35:H40, 0.25)`
- 75th Percentile: `=PERCENTILE(H35:H40, 0.75)` or `=PERCENTILE.INC(H35:H40, 0.75)`

Read the actual labels in rows 42-47 to assign the correct formula to each row.

### Step 3: Weighted mean in H50:L50

For each column H through L:
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```
This computes the weighted mean of Net patient flow percentages weighted by Effective Bed Capacity.

### Step 4: Save and Validate
1. Save the workbook to `/root/output/result.xlsx`.
2. Re-open the saved file and verify:
   - Cells H12:L17, H19:L24, H26:L31 all contain formulas (not None/empty)
   - Cells H35:L40 contain formulas
   - Cells H42:L47 contain formulas
   - Cells H50:L50 contain formulas
   - No extra sheets were added
   - Print a sample of formulas to confirm correctness

### Critical Notes
- Use `openpyxl` with `load_workbook(filename, data_only=False)` to preserve and write formulas.
- Do NOT use `data_only=True` as that strips formulas.
- Do NOT add new sheets, macros, VBA, or external links.
- Preserve all existing formatting (do not clear cells, do not change fonts/colors/borders).
- When writing formulas, just assign the formula string to `cell.value`.
- Make sure the `Data` sheet reference in formulas uses the exact sheet name as it appears in the workbook (check `wb.sheetnames`).

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