# Task Instruction

Execute the following steps to produce `/root/output/result.xlsx` from `/root/data/workbook.xlsx`.

## Step 0 – Inspect the workbook layout

```python
import openpyxl, os
wb = openpyxl.load_workbook('/root/data/workbook.xlsx')
for s in wb.sheetnames:
    print(f'--- {s} ---')
    ws = wb[s]
    print(f'  dimensions: {ws.dimensions}')
    # Print key reference cells
    if s == 'Task':
        # Row 10 (year headers), column D (series codes), sample yellow cells
        print('  Row 10 (H-L):', [ws.cell(10, c).value for c in range(8, 13)])
        print('  Col D rows 12-17:', [ws.cell(r, 4).value for r in range(12, 18)])
        print('  Col D rows 19-24:', [ws.cell(r, 4).value for r in range(19, 25)])
        print('  Col D rows 26-31:', [ws.cell(r, 4).value for r in range(26, 32)])
        print('  Col A rows 35-40:', [ws.cell(r, 1).value for r in range(35, 41)])
        print('  Col A rows 42-47:', [ws.cell(r, 1).value for r in range(42, 48)])
        print('  Col A row 50:', ws.cell(50, 1).value)
        # Check what's in H12 already
        print('  H12 current:', ws.cell(12, 8).value)
    if s == 'Data':
        # Rows 21-38: print col A and col B to understand lookup structure
        for r in range(20, 39):
            print(f'  Data row {r}: A={ws.cell(r,1).value}, B={ws.cell(r,2).value}, C={ws.cell(r,3).value}')
        # Also check row 20 for headers
        print('  Data row 20 H-L:', [ws.cell(20, c).value for c in range(8, 13)])
        # Check column structure of Data sheet
        print('  Data cols A-G row 21:', [ws.cell(21, c).value for c in range(1, 8)])
wb.close()
```

Read and understand the layout carefully before proceeding. The Data sheet rows 21:38 contain the source records; identify which column holds the series codes and which row/column holds the year values so you can build correct MATCH references.

## Step 1 – Populate lookup formulas in H12:L17, H19:L24, H26:L31

Using `INDEX/MATCH` with locked references:
- Row anchor for years: row 10 on sheet Task (use `H$10`, `I$10`, etc., or a single formula with `H$10` that copies across).
- Column anchor for series codes: column D on sheet Task (use `$D12`, `$D13`, etc.).
- The MATCH for the series code should search the appropriate column in `Data!$A$21:$A$38` (or whichever column holds the codes — confirm in Step 0).
- The MATCH for the year should search the appropriate header row in the Data sheet.
- The INDEX range should be the data block in Data rows 21:38.

Use the pattern that worked before:
```
=INDEX(Data!<data_range>, MATCH($D12, Data!<code_column>, 0), MATCH(H$10, Data!<year_header_row>, 0))
```

Write formulas into every cell in the three blocks (6 rows × 5 columns = 30 cells each, 90 total). Use openpyxl to write formula strings. Make sure the `$` anchors are correct so each cell references its own row's series code and its own column's year.

## Step 2 – Net capacity headroom (H35:L40) and summary statistics (H42:L47)

For H35:L40, the formula per cell is:
```
=(H12 - H19) / H26 * 100
```
where row 12 = Available Care Slots, row 19 = Occupied Care Slots, row 26 = Staffed Bed Capacity. Adjust row references for each of the 6 hospital clusters (rows 12-17 map to 35-40, rows 19-24 map to 35-40, rows 26-31 map to 35-40).

For H42:L47 (column-wise statistics over H35:L40):
- Row 42: `=MIN(H35:H40)`
- Row 43: `=MAX(H35:H40)`
- Row 44: `=MEDIAN(H35:H40)`
- Row 45: `=AVERAGE(H35:H40)`
- Row 46: `=PERCENTILE(H35:H40, 0.25)`
- Row 47: `=PERCENTILE(H35:H40, 0.75)`

Verify the order of these statistics by checking the labels in column A/B of rows 42-47. Adjust the order to match the labels.

## Step 3 – Weighted mean in H50:L50

For each column (H through L):
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```

## Step 4 – Save

```python
os.makedirs('/root/output', exist_ok=True)
wb.save('/root/output/result.xlsx')
```

## Validation

After saving, reload the file and confirm:
1. H12 contains a formula string (not None or a plain value).
2. H35 contains a formula string.
3. H42 contains a formula string.
4. H50 contains a formula string.
5. No new sheets were added.
6. Run `pytest /root/test_output.py -v` if the test file exists.

IMPORTANT: Throughout, do NOT use `data_only=True` when loading. Always write formula strings (starting with `=`), never computed values. Do not add sheets, macros, VBA, external links, or helper tabs. Preserve all existing formatting.

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