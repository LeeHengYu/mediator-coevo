# Task Instruction

Execute the following steps precisely to complete the task.

## 0. Preparation

```bash
mkdir -p /root/output
pip install openpyxl
```

Inspect the workbook to understand the exact layout:

```python
import openpyxl
wb = openpyxl.load_workbook('/root/data/workbook.xlsx')
for name in wb.sheetnames:
    print(f'=== Sheet: {name} ===')
    ws = wb[name]
    for row in ws.iter_rows(min_row=1, max_row=ws.max_row, max_col=ws.max_column, values_only=False):
        for cell in row:
            if cell.value is not None:
                print(f'  {cell.coordinate}: {repr(cell.value)}')
```

Run this first and read the output carefully. Pay special attention to:
- The series codes in column D of sheet `Task` for rows 12-17, 19-24, 26-31, 35-40
- The year values in row 10 of sheet `Task` (columns H through L)
- The structure of sheet `Data` rows 21-38 (what columns hold what, where series codes and years are)
- What is already in cells H35:L40, H42:L47, H50:L50
- The region names and their order

## 1. Understand the Data sheet layout

After inspecting, determine:
- Which column on `Data` contains the series code (the lookup key)
- Which row on `Data` contains the year headers
- Where the data values begin (the table range for lookups)
- The exact range to use in VLOOKUP/INDEX/MATCH formulas

## 2. Write formulas using openpyxl

Use a Python script with openpyxl to open the workbook and write formulas into the cells. Important: use `openpyxl.load_workbook('/root/data/workbook.xlsx')` and save to `/root/output/result.xlsx`.

### Step 1: Lookup formulas in H12:L17, H19:L24, H26:L31

For each cell in these ranges, write an INDEX/MATCH formula. The formula pattern should be:

```
=INDEX(Data!<data_range>, MATCH(<series_code_cell>, Data!<series_code_column>, 0), MATCH(<year_cell>, Data!<year_header_row>, 0))
```

Where:
- `<series_code_cell>` is the absolute reference to column D of the current row on Task sheet (e.g., `$D12`)
- `<year_cell>` is the absolute reference to the year in row 10 of the current column (e.g., `H$10`)
- `<data_range>` is the data block on the Data sheet (rows 21:38, covering the value columns)
- `<series_code_column>` is the column on Data that holds series codes (same rows 21:38)
- `<year_header_row>` is the row on Data that holds year headers

Adapt the exact ranges based on what you find in the inspection. The key constraint is: use INDEX with MATCH (or one of the other allowed patterns: VLOOKUP+MATCH, HLOOKUP+MATCH, XLOOKUP+MATCH).

Make sure references are properly mixed (row-absolute or column-absolute) so they work when conceptually "copied" across the range. Use `$D` for the series code column and `$10` for the year row.

### Step 2: Net reliability gap in H35:L40

For each cell in H35:L40, the formula is:
```
=(H12 - H19) / H26 * 100
```
(Adjust row references for each row: row 35 uses rows 12,19,26; row 36 uses 13,20,27; etc.)

The mapping is:
- Row 35 -> Successful=Row12, Failed=Row19, Capacity=Row26
- Row 36 -> Successful=Row13, Failed=Row20, Capacity=Row27
- Row 37 -> Successful=Row14, Failed=Row21, Capacity=Row28
- Row 38 -> Successful=Row15, Failed=Row22, Capacity=Row29
- Row 39 -> Successful=Row16, Failed=Row23, Capacity=Row30
- Row 40 -> Successful=Row17, Failed=Row24, Capacity=Row31

### Step 2 continued: Summary stats in H42:L47

For each column (H through L):
- Row 42: `=MIN(H35:H40)`
- Row 43: `=MAX(H35:H40)`
- Row 44: `=MEDIAN(H35:H40)`
- Row 45: `=AVERAGE(H35:H40)`
- Row 46: `=PERCENTILE(H35:H40, 0.25)`
- Row 47: `=PERCENTILE(H35:H40, 0.75)`

Verify from the inspection which row is which statistic. The instruction says: minimum, maximum, median, simple mean, 25th percentile, 75th percentile — in that order for rows 42-47.

### Step 3: Weighted mean in H50:L50

For each column (H through L):
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```

This uses the Net reliability gap values as the "values" and Compute Capacity as weights.

## 3. Save and validate

Save to `/root/output/result.xlsx`. Then re-open and verify:
- All formula cells contain formula strings (not None or numeric values)
- The formulas reference the correct cells
- No extra sheets were added
- Print a sample of formulas to confirm correctness

```python
wb2 = openpyxl.load_workbook('/root/output/result.xlsx')
ws = wb2['Task']
for coord in ['H12','L17','H19','L24','H26','L31','H35','L40','H42','L47','H50','L50']:
    print(f'{coord}: {ws[coord].value}')
```

## CRITICAL NOTES

- Before writing any formulas, you MUST inspect the workbook to understand the exact layout. The row/column references above are based on the task description but the Data sheet layout needs to be discovered.
- Do NOT add any new sheets, macros, VBA, or external links.
- Do NOT change any existing formatting.
- Use `data_only=False` when loading (which is the default) to preserve existing formulas.
- Write formula strings (starting with '=') into cells, not computed values.
- The inspection step is mandatory — do not skip it.

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