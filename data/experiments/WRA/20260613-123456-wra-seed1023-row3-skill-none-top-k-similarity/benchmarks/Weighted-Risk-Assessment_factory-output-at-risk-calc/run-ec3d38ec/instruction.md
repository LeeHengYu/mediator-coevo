# Task Instruction

Execute the following steps to produce /root/output/result.xlsx.

## 0 – Preparation
```bash
mkdir -p /root/output
```
Inspect the workbook to understand its layout:
```python
import openpyxl
wb = openpyxl.load_workbook('/root/data/workbook.xlsx')
for s in wb.sheetnames:
    ws = wb[s]
    print(f'--- {s} (rows={ws.max_row}, cols={ws.max_column}) ---')
    for row in ws.iter_rows(min_row=1, max_row=min(ws.max_row, 50), values_only=False):
        vals = [(c.coordinate, c.value) for c in row if c.value is not None]
        if vals:
            print(vals)
```
Pay close attention to:
- The series codes in column D for rows 12-17, 19-24, 26-31, 35-40.
- The years in row 10 (columns H-L).
- The Data sheet layout, especially rows 21-38: which row holds which series, and which column holds which year. Determine whether Data is organized with years in rows or columns.
- Identify the exact column/row offsets needed for VLOOKUP/HLOOKUP/INDEX-MATCH.

## 1 – Write the formulas

Open the workbook with openpyxl (data_only=False so formulas are preserved), and write formulas into the Task sheet.

### Step 1: Lookup formulas in H12:L17, H19:L24, H26:L31

For each cell in those ranges, write an INDEX-MATCH-MATCH or equivalent formula. The formula must:
- Use the series code from column D of that row.
- Use the year from row 10 of that column.
- Reference the Data sheet rows 21:38.

Determine the correct formula pattern by inspecting the Data sheet structure:
- If Data has series codes in a column and years in a header row, use INDEX/MATCH/MATCH.
- Example pattern (adjust ranges after inspection):
  `=INDEX(Data!$B$22:$F$38, MATCH($D12, Data!$A$22:$A$38, 0), MATCH(H$10, Data!$B$21:$F$21, 0))`
  Adjust column letters and row numbers to match the actual layout.

### Step 2: Net production slack in H35:L40

The formula for each cell is:
`= (FinishedOutput - ScrapAndRework) / RatedProductionCapacity * 100`

where FinishedOutput, ScrapAndRework, and RatedProductionCapacity come from the three blocks above (rows 12-17, 19-24, 26-31 respectively — verify which block maps to which metric by reading the labels in the Task sheet).

For example, if rows 12-17 are Finished Output, rows 19-24 are Scrap And Rework, and rows 26-31 are Rated Production Capacity, then for cell H35:
`= (H12 - H19) / H26 * 100`
And similarly for the rest of H35:L40.

Verify the mapping by reading the block labels before writing.

### Step 2 continued: Summary statistics in H42:L47

For each column (H through L), compute six statistics over the 6 cells in that column's rows 35-40:
- Row 42: MIN  → `=MIN(H35:H40)`
- Row 43: MAX  → `=MAX(H35:H40)`
- Row 44: MEDIAN → `=MEDIAN(H35:H40)`
- Row 45: AVERAGE → `=AVERAGE(H35:H40)`
- Row 46: 25th percentile → `=PERCENTILE(H35:H40,0.25)`
- Row 47: 75th percentile → `=PERCENTILE(H35:H40,0.75)`

**CRITICAL**: Verify the row labels to confirm the order (min, max, median, mean, 25th, 75th). Read cells in column A/B/C/D for rows 42-47.

**CRITICAL**: For the percentile functions, use `PERCENTILE` (not `PERCENTILE.INC`, not `_xlfn.PERCENTILE.INC`). The plain `PERCENTILE` function is the safest for openpyxl compatibility. However, if the verifier still rejects it, the fallback is `_xlfn.PERCENTILE.INC`. But try `PERCENTILE` first.

Actually — based on the previous failure feedback about #NAME? errors: openpyxl may write `PERCENTILE` but modern .xlsx files sometimes need the `_xlfn.` prefix for certain functions. To be safe, **test both approaches**: first try writing with plain `PERCENTILE`. Then open the saved file, read back the cells, and check if they contain the expected formula text. The real issue is that when Excel (or the verifier's evaluation engine) opens the file, `PERCENTILE` must resolve.

The safest approach: use `PERCENTILE` (without prefix). If the verifier evaluates formulas using an engine that doesn't recognize it, switch to `_xlfn.PERCENTILE.INC`.

Let me be more specific: **Use `PERCENTILE` for the formula text.** openpyxl will write it as-is. Standard Excel recognizes `PERCENTILE`. The #NAME? error in the previous run was likely from using `PERCENTILE.INC` without the `_xlfn.` prefix, or some other function name issue.

Also double-check: for AVERAGE (not MEAN), MIN, MAX, MEDIAN — these are all standard and should work fine.

### Step 3: Weighted mean in H50:L50

For each column H through L:
`=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)`

Wait — re-read the instruction: "using the Step 2 percentages as values and the Rated Production Capacity block in H26:L31 as weights". A weighted mean with SUMPRODUCT is:
`=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)`

This computes the weighted average of the net production slack percentages, weighted by rated production capacity.

## 2 – Save and Validate

Save to `/root/output/result.xlsx`.

Then validate:
```python
wb2 = openpyxl.load_workbook('/root/output/result.xlsx')
ws = wb2['Task']
# Check a sample of cells to confirm formulas were written
for cell in ['H12', 'L17', 'H19', 'L24', 'H26', 'L31', 'H35', 'L40', 'H42', 'H46', 'H47', 'L47', 'H50', 'L50']:
    print(f'{cell}: {ws[cell].value}')
```

Confirm:
- Lookup cells contain INDEX/MATCH formulas referencing Data sheet
- Calculation cells contain arithmetic formulas
- Statistics cells contain MIN/MAX/MEDIAN/AVERAGE/PERCENTILE formulas
- Weighted mean cells contain SUMPRODUCT formulas
- No #NAME?, no empty cells in the target ranges

If any cell is None or unexpected, fix and re-save.

## 3 – Run the verifier if available
```bash
cd /root && find . -name 'test_output*' -o -name 'test_result*' | head -5
```
If tests exist, run them:
```bash
cd /root && python -m pytest tests/ -x -v 2>&1 | tail -40
```
Fix any failures and re-save.

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