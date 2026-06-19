# Task Instruction

Execute the following steps in a single Python script to produce /root/output/result.xlsx.

## Step 0: Inspect the workbook

```python
import openpyxl, os, shutil

os.makedirs('/root/output', exist_ok=True)
shutil.copy('/root/data/workbook.xlsx', '/root/output/result.xlsx')

wb = openpyxl.load_workbook('/root/output/result.xlsx')
ws_task = wb['Task']
ws_data = wb['Data']

# Print row 10 headers (years) in columns H-L
for col in range(8, 13):  # H=8 .. L=12
    print(f"Task row10 col{col}: {ws_task.cell(row=10, column=col).value}")

# Print series codes in column D for rows 12-17, 19-24, 26-31, 35-40, 42-47, 50
for r in list(range(12,18)) + list(range(19,25)) + list(range(26,32)) + list(range(35,41)) + list(range(42,48)) + [50]:
    print(f"Task D{r}: {ws_task.cell(row=r, column=4).value}")

# Print Data sheet structure: row 21 headers and a few rows
for r in range(20, 40):
    row_vals = [ws_data.cell(row=r, column=c).value for c in range(1, 20)]
    print(f"Data row{r}: {row_vals}")

wb.close()
```

Run this first and read the output carefully before proceeding.

## Step 1: Populate lookup formulas in H12:L17, H19:L24, H26:L31

After inspecting the output, construct INDEX/MATCH formulas. The pattern for each cell at (row, col) should be:

```
=INDEX(Data!$A$21:$Z$38, MATCH(D{row}, Data!$A$21:$A$38, 0), MATCH(Task!{col_letter}$10, Data!$A$21:$Z$21, 0))
```

Adjust the column range based on what you see in the Data sheet (use the actual extent of data columns). The series code is in column D of the Task sheet row, and the year is in row 10 of the Task sheet.

Write these formulas for every cell in the three blocks (H12:L17, H19:L24, H26:L31). Use absolute references for the Data range and the header row/column, and a relative reference for D{row} and the year column header.

## Step 2: Net production slack in H35:L40

For each cell in H35:L40, the formula is:
```
=({corresponding cell from H12:L17} - {corresponding cell from H19:L24}) / {corresponding cell from H26:L31} * 100
```

The offset mapping: row 35 maps to rows 12, 19, 26; row 36 maps to rows 13, 20, 27; etc. Columns H-L stay the same.

Verify the D column labels in rows 35-40 match the plant names in rows 12-17 (they should correspond to the same six plants).

## Step 3: Statistics in H42:L47

Read the labels in D42:D47 to determine which statistic goes in which row. Based on the task description, expect: minimum, maximum, median, simple mean, 25th percentile, 75th percentile. Map each label to the appropriate Excel function:

- Minimum → `=MIN(H35:H40)` (or whichever column)
- Maximum → `=MAX(H35:H40)`
- Median → `=MEDIAN(H35:H40)`
- Simple mean → `=AVERAGE(H35:H40)`
- 25th percentile → `=PERCENTILE(H35:H40, 0.25)`
- 75th percentile → `=PERCENTILE(H35:H40, 0.75)`

Apply across columns H through L, adjusting the column letter.

## Step 4: Weighted mean in H50:L50

For each column (H through L):
```
=SUMPRODUCT({col}35:{col}40, {col}26:{col}31) / SUM({col}26:{col}31)
```

This computes the weighted mean of the Net production slack percentages using Rated Production Capacity as weights.

## Step 5: Save

Save the workbook to `/root/output/result.xlsx`. Do NOT use `data_only=True` when loading. Make sure to preserve existing formatting.

## Important constraints
- Use openpyxl to write Excel formulas (strings starting with '=') into cells. Do NOT compute values in Python.
- Do NOT add sheets, macros, VBA, external links, or helper tabs.
- Do NOT modify any cells outside the specified ranges.
- After writing all formulas, re-read a sample of cells to confirm formulas were written correctly.
- The previous successful run used this exact approach (INDEX/MATCH formulas via openpyxl) and scored 1.0.

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