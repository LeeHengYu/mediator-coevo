# Task Instruction

Execute the following steps precisely to complete the task.

## 0. Setup
```bash
mkdir -p /root/output
pip install openpyxl
```

## 1. Inspect the workbook structure
Open `/root/data/workbook.xlsx` with openpyxl and inspect:
- Sheet names (confirm `Task` and `Data` exist)
- On sheet `Task`: read cells D12:D17 (series codes for block 1), D19:D24 (block 2), D26:D31 (block 3), and H10:L10 (years in the header row). Also read row 35-40 labels, row 42-47 labels, and row 50 label. Print all of these so you understand the layout.
- On sheet `Data`: read rows 21 through 38 to understand the data table structure — specifically which row is the header, which column has series codes, and how the year columns are laid out. Print the first few columns of each row and the full header row.
- Also inspect the yellow-highlighted cells to confirm the target ranges.

Print everything clearly before proceeding.

## 2. Write a Python script using openpyxl to populate formulas

Use openpyxl to open the workbook and write formulas into the cells. Key rules:
- Use `data_only=False` (default) so existing formulas are preserved.
- Do NOT overwrite any cells outside the specified ranges.
- Preserve all existing formatting by not touching style attributes.

### Step 1: Lookup formulas in H12:L17, H19:L24, H26:L31

For each cell in these three blocks, write a formula that looks up data from `Data!$21:$38`. The formula must use the series code from column D of the same row on sheet `Task` and the year from row 10 of the same column on sheet `Task`.

Based on what you discover about the Data sheet layout, choose the most appropriate lookup pattern. The most likely suitable pattern is INDEX-MATCH-MATCH:

```
=INDEX(Data!<data_range>, MATCH($D{row}, Data!<series_code_column>, 0), MATCH(H$10, Data!<year_header_row>, 0))
```

Adjust the exact references based on what you find in the inspection step. The `$D{row}` should use a mixed reference (column absolute, row relative to current). The `H$10` should use mixed reference (column relative, row absolute).

IMPORTANT: Confirm the exact row range and column range on the Data sheet. The series code column is likely column A or B on Data. The year headers are likely in a specific row (maybe row 21 or row 20). The data area should span from the first data column to the last year column.

### Step 2: Net reliability gap formulas in H35:L40

The three blocks correspond to:
- H12:L17 = one metric (e.g., Successful API Requests)
- H19:L24 = another metric (e.g., Failed API Requests)  
- H26:L31 = Compute Capacity

Verify which block is which by reading the labels (likely in column B or C near rows 11, 18, 25).

The formula for each cell in H35:L40 is:
```
=(H{row_successful} - H{row_failed}) / H{row_capacity} * 100
```

where the row offsets correspond to the same region (row 1 of each block = same region). For example, if row 12 is region 1 in block 1, row 19 is region 1 in block 2, and row 26 is region 1 in block 3, then H35 = (H12 - H19) / H26 * 100.

### Step 2 continued: Summary statistics in H42:L47

For each column H through L, based on the labels in rows 42-47 (which should be min, max, median, mean, 25th percentile, 75th percentile — verify the exact order from the labels):
- Minimum: `=MIN(H35:H40)`
- Maximum: `=MAX(H35:H40)`
- Median: `=MEDIAN(H35:H40)`
- Mean: `=AVERAGE(H35:H40)`
- 25th percentile: `=PERCENTILE(H35:H40, 0.25)` or `=PERCENTILE.INC(H35:H40, 0.25)`
- 75th percentile: `=PERCENTILE(H35:H40, 0.75)` or `=PERCENTILE.INC(H35:H40, 0.75)`

Match each formula to the correct row based on the actual label text.

### Step 3: Weighted mean in H50:L50

For each column H through L:
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```

This computes the weighted mean of the Net reliability gap percentages using Compute Capacity as weights.

## 3. Save and verify
- Save to `/root/output/result.xlsx`
- Reopen the saved file and print the formulas in all modified cells to confirm they are correctly written.
- Verify no extra sheets were added.
- Verify the file is valid by reopening it without errors.

## Critical Notes
- Do NOT use `data_only=True` when loading — this would strip formulas.
- When writing formulas with openpyxl, assign the formula string (starting with `=`) directly to `cell.value`.
- Do NOT modify any cell formatting, styles, or conditional formatting.
- Do NOT add any new sheets.
- Double-check all cell references by inspecting the actual workbook structure first.

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