# Task Instruction

Execute the following steps carefully and in order.

## 0. Inspect the workbook

```python
import openpyxl, pprint
wb = openpyxl.load_workbook('/root/data/workbook.xlsx', data_only=True)
print('Sheet names:', wb.sheetnames)

task = wb['Task']
data = wb['Data']

# Print Task sheet: series codes in column D, rows 10-31
print('\n--- Task sheet column D (rows 10-31) ---')
for r in range(10, 32):
    print(f'  D{r}: {repr(task.cell(r, 4).value)}')

# Print Task sheet: year headers in row 10, columns H-L
print('\n--- Task sheet row 10, cols H-L ---')
for c in range(8, 13):
    print(f'  {chr(64+c)}{10}: {repr(task.cell(10, c).value)}')

# Print Task sheet rows 33-50 col A-D to understand layout
print('\n--- Task sheet col A-G rows 33-52 ---')
for r in range(33, 53):
    vals = [task.cell(r, c).value for c in range(1, 8)]
    print(f'  Row {r}: {vals}')

# Print Data sheet structure
print('\n--- Data sheet rows 1-5 (headers) ---')
for r in range(1, 6):
    vals = [data.cell(r, c).value for c in range(1, 30)]
    print(f'  Row {r}: {vals}')

# Print Data sheet rows 19-40
print('\n--- Data sheet rows 19-40 ---')
for r in range(19, 41):
    vals = [data.cell(r, c).value for c in range(1, 30)]
    print(f'  Row {r}: {vals}')

wb.close()
```

Study the output carefully. Identify:
- The exact series code strings in Task!D12:D17, D19:D24, D26:D31.
- The exact year values in Task!H10:L10.
- The exact layout of Data sheet rows 21:38 — which column holds the series code, and which columns/rows hold the year headers and values.
- Whether the Data sheet is arranged with series codes in rows and years in columns, or vice versa.

## 1. Build the lookup formulas (Step 1)

Open the workbook with openpyxl (NOT data_only) and write formulas.

For each cell in H12:L17, H19:L24, H26:L31, write an INDEX/MATCH formula that:
- Uses the series code from column D of the same row on the Task sheet
- Uses the year from row 10 of the same column on the Task sheet
- Looks up in the Data sheet rows 21:38

IMPORTANT: Determine the correct Data range from step 0 output. The formula pattern should be something like:
```
=INDEX(Data!$B$21:$Z$38, MATCH($D12, Data!$A$21:$A$38, 0), MATCH(H$10, Data!$B$20:$Z$20, 0))
```
But adjust the exact ranges based on what you found in step 0:
- The row range for MATCH on series codes (first column of data area)
- The column range for MATCH on years (header row of data area)
- The data area for INDEX

Use absolute references for the data ranges ($) and mixed references so the formula can vary by row (series code from $D12) and column (year from H$10).

Verify that the series code strings in Task!D column EXACTLY match those in the Data sheet (check for extra spaces, different casing, etc.).

## 2. Net patient flow formulas (Step 2)

In H35:L40, write formulas for:
`(Patient Admissions - Patient Discharges) / Effective Bed Capacity * 100`

Determine which rows in the Task sheet correspond to:
- Patient Admissions (likely H12:L17)
- Patient Discharges (likely H19:L24)
- Effective Bed Capacity (likely H26:L31)

The formula for H35 should be something like:
`=(H12-H19)/H26*100`
with the row offsets matching the hospital ordering.

Verify that the hospital order is the same across all three blocks and the Net patient flow block.

## 3. Statistics formulas (Step 2 continued)

In H42:L47, write column-wise statistics over H35:L40:
- Row 42: MIN  → `=MIN(H35:H40)`
- Row 43: MAX  → `=MAX(H35:H40)`
- Row 44: MEDIAN → `=MEDIAN(H35:H40)`
- Row 45: AVERAGE → `=AVERAGE(H35:H40)`
- Row 46: PERCENTILE (25th) → `=PERCENTILE(H35:H40,0.25)`
- Row 47: PERCENTILE (75th) → `=PERCENTILE(H35:H40,0.75)`

Check the labels in column D/E/F/G of rows 42-47 to confirm the correct order of statistics. Use `PERCENTILE` (not `PERCENTILE.INC` or `PERCENTILE.EXC`) to avoid #NAME? errors — but verify which function name works. If the workbook is .xlsx format, `PERCENTILE.INC` should work, but `PERCENTILE` is safer for compatibility.

ACTUALLY: Use `PERCENTILE` for maximum compatibility. The cross-task failure artifact shows #NAME? errors from the statistics section, likely from using an unsupported function name.

## 4. Weighted mean (Step 3)

In H50:L50, write a SUMPRODUCT formula:
`=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)`

This computes the weighted mean of Net patient flow percentages using Effective Bed Capacity as weights.

## 5. Save and verify

Save to `/root/output/result.xlsx`. Then reopen with data_only=False and print all formula cells to confirm they are present. Also reopen with data_only=True (after saving from a formula-aware engine if needed) or use a spot check.

Actually, openpyxl cannot evaluate formulas. Instead:
1. Save the file.
2. Reopen it and print the formula strings for a sample of cells (e.g., H12, L17, H35, L40, H42, H47, H50, L50) to confirm formulas were written correctly.
3. Verify no cells in the target ranges are None or empty.

## Critical checks
- Ensure all formulas start with '='
- Ensure the Data sheet name in formulas matches exactly (case-sensitive in references)
- Ensure no extra sheets are created
- Ensure existing formatting is preserved (use openpyxl load without data_only, write formulas, save)
- Create /root/output/ directory if it doesn't exist

```python
import os
os.makedirs('/root/output', exist_ok=True)
```

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