# Task Instruction

Execute the following steps to produce /root/output/result.xlsx.

## 0 – Preparation
```bash
mkdir -p /root/output
```
Open and inspect the workbook:
```python
import openpyxl
wb = openpyxl.load_workbook('/root/data/workbook.xlsx')
print(wb.sheetnames)
ts = wb['Task']
ds = wb['Data']
# Inspect Task sheet layout
for r in ts.iter_rows(min_row=1, max_row=55, max_col=13, values_only=False):
    print([(c.coordinate, c.value) for c in r])
# Inspect Data sheet rows 1-5 and 18-40
for r in ds.iter_rows(min_row=1, max_row=5, values_only=False):
    print([(c.coordinate, c.value) for c in r])
for r in ds.iter_rows(min_row=18, max_row=40, values_only=False):
    print([(c.coordinate, c.value) for c in r])
```
Study the output carefully before writing any formulas. Identify:
- The series codes in column D for rows 12-17, 19-24, 26-31.
- The years in row 10 for columns H-L.
- The layout of the Data sheet rows 21-38: which row holds headers, which column holds series codes, and which columns hold year values.
- Confirm the exact column letters and row numbers in Data that hold the lookup table.

## 1 – Step 1: Lookup formulas in H12:L17, H19:L24, H26:L31

For each cell in these three blocks, write an INDEX/MATCH formula that:
- Uses the series code from column D of the same row on Task sheet.
- Uses the year from row 10 of the same column on Task sheet.
- Looks up the value from the Data sheet rows 21:38.

The formula pattern (adjust column/row references based on your inspection):
```
=INDEX(Data!<data_range>, MATCH($D12, Data!<series_code_column>, 0), MATCH(H$10, Data!<year_header_row>, 0))
```
Where:
- `<data_range>` is the rectangular block on Data that contains the numeric values (e.g., Data!B21:Z38 or similar – determine from inspection).
- `<series_code_column>` is the column in Data holding series codes (e.g., Data!A21:A38 or Data!B21:B38).
- `<year_header_row>` is the row in Data holding years (e.g., Data!B20:Z20).

Use absolute row references for the year row ($10) and absolute column references for the series code column ($D) so the formula copies correctly across the 5 columns and 6 rows of each block.

Populate all 3 blocks (18 cells each = 54 cells? No: 5 cols × 6 rows = 30 cells per block, 90 total). Actually: columns H through L = 5 columns, rows per block = 6. So 30 cells per block, 90 cells total.

Write these using openpyxl by assigning formula strings to each cell.

## 2 – Step 2: Net capacity headroom in H35:L40

For each cell in H35:L40, compute:
```
=(H12 - H19) / H26 * 100
```
where H12 corresponds to Available Care Slots (rows 12-17), H19 to Occupied Care Slots (rows 19-24), H26 to Staffed Bed Capacity (rows 26-31). The row offset within each block should match: row 35 uses rows 12, 19, 26; row 36 uses 13, 20, 27; etc.

So for cell in row r (35-40), column c (H-L):
```
=(<c><r-23> - <c><r-16>) / <c><r-9> * 100
```
Verify: r=35 → 35-23=12, 35-16=19, 35-9=26. ✓

## 3 – Step 2 continued: Statistics in H42:L47

For each column c in H:L:
- Row 42: `=MIN(c35:c40)`
- Row 43: `=MAX(c35:c40)`
- Row 44: `=MEDIAN(c35:c40)`
- Row 45: `=AVERAGE(c35:c40)`
- Row 46: `=PERCENTILE(c35:c40, 0.25)`
- Row 47: `=PERCENTILE(c35:c40, 0.75)`

IMPORTANT: Check the Task sheet labels in column D or nearby for rows 42-47 to confirm which row is which statistic. Adjust the order accordingly. Do NOT assume the order above – verify from the sheet.

## 4 – Step 3: Weighted mean in H50:L50

For each column c in H:L:
```
=SUMPRODUCT(c35:c40, c26:c31) / SUM(c26:c31)
```

## 5 – Save
```python
wb.save('/root/output/result.xlsx')
```

## 6 – Validate
Reopen the saved file (without data_only) and verify:
- Cells in H12:L17, H19:L24, H26:L31 contain formula strings starting with '='.
- Cells in H35:L40 contain formula strings.
- Cells in H42:L47 contain formula strings.
- Cells in H50:L50 contain formula strings.
- No new sheets were added.
- Print a sample of formulas to confirm correctness.

Also open with data_only=True and check that values are not None (they will be None since openpyxl can't evaluate, but at least confirm the formulas parse).

CRITICAL NOTES:
- Do NOT use `data_only=True` when loading the workbook for editing.
- Do NOT add sheets, macros, VBA, external links, or helper tabs.
- Keep existing formatting unchanged – only write to the specified cells.
- Before writing formulas, carefully inspect the Data sheet to get exact cell references. The formulas must reference the correct cells on the Data sheet.
- The avoid/recheck artifact warns that empty cells (None values) result from incorrect formula references. Double-check every reference.

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
Task metadata: author_email=catpaw@meituan.com, author_name=CatPaw Task Engineer, category=spreadsheet-formula-reuse, difficulty=hard, tags=[excel, formulas, lookup, statistics, weighted-mean].
Verifier config: timeout_sec=600.0.