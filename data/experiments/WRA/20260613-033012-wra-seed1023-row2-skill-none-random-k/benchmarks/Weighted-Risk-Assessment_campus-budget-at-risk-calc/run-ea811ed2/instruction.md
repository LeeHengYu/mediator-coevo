# Task Instruction

Execute the following steps to produce `/root/output/result.xlsx`.

## 1 – Inspect the workbook layout

```python
import openpyxl, os
wb = openpyxl.load_workbook('/root/data/workbook.xlsx')
for name in wb.sheetnames:
    print(f'--- {name} ---')
    ws = wb[name]
    for row in ws.iter_rows(min_row=1, max_row=ws.max_row, max_col=ws.max_column, values_only=False):
        vals = [(c.coordinate, c.value) for c in row if c.value is not None]
        if vals:
            print(vals)
```

Record:
- The series codes in column D for rows 12-17, 19-24, 26-31 on sheet `Task`.
- The years in row 10 for columns H-L on sheet `Task`.
- The layout of sheet `Data` rows 21-38 (which column holds the series code, which row holds years, where numeric data starts).
- The labels/structure of rows 35-40, 42-47, 50 on sheet `Task`.

## 2 – Write formulas into the workbook

Open the workbook **without** `data_only` so existing formulas and formatting are preserved:

```python
wb = openpyxl.load_workbook('/root/data/workbook.xlsx')
ts = wb['Task']
```

### Step 1 – Lookup formulas in H12:L17, H19:L24, H26:L31

For each cell in these three 6×5 blocks, write an `INDEX/MATCH/MATCH` formula that:
- Looks up the series code from column D of the current row against the series-code column on `Data` (rows 21-38).
- Looks up the year from row 10 of the current column against the year row on `Data`.
- Returns the intersecting value from the Data table body.

Use mixed references so the formula can be filled across the block:
- `$D12` (absolute column, relative row) for the series code.
- `H$10` (relative column, absolute row) for the year.
- Absolute references for the Data ranges.

Example pattern (adjust column letters and row numbers based on your inspection):
```
=INDEX(Data!$B$22:$<lastcol>$38, MATCH($D12, Data!$A$22:$A$38, 0), MATCH(H$10, Data!$B$21:$<lastcol>$21, 0))
```
Adjust `$A`, `$B`, `<lastcol>` etc. to match the actual layout you observed.

Loop over the three blocks and assign the formula string to each cell.

### Step 2a – Net budget buffer (H35:L40)

The three lookup blocks correspond to three metrics. Based on the task description and typical ordering:
- Block 1 (rows 12-17): Committed Funding
- Block 2 (rows 19-24): Operating Spend  
- Block 3 (rows 26-31): Approved Budget Base

**Verify this mapping** by checking the block headers/labels you recorded in step 1. If the ordering differs, adjust accordingly.

For each cell in H35:L40 (6 departments × 5 years), write:
```
=(H12-H19)/H26*100
```
with the row offsets matching: row 35↔(12,19,26), row 36↔(13,20,27), … row 40↔(17,24,31). Use relative references so filling works naturally, or compute the offset in the loop.

### Step 2b – Summary statistics (H42:L47)

For each column H through L:
- Row 42: `=MIN(H35:H40)`
- Row 43: `=MAX(H35:H40)`
- Row 44: `=MEDIAN(H35:H40)`
- Row 45: `=AVERAGE(H35:H40)`
- Row 46: `=PERCENTILE(H35:H40,0.25)`
- Row 47: `=PERCENTILE(H35:H40,0.75)`

Use `PERCENTILE` (not `PERCENTILE.INC`) for compatibility.

### Step 3 – Weighted mean (H50:L50)

For each column H through L:
```
=SUMPRODUCT(H35:H40,H26:H31)/SUM(H26:H31)
```

## 3 – Save

```python
os.makedirs('/root/output', exist_ok=True)
wb.save('/root/output/result.xlsx')
```

## 4 – Validate

Reload the saved file and spot-check:
- Cells H12, L17, H19, L24, H26, L31 contain formula strings (start with `=`).
- Cells H35, L40 contain formula strings.
- Cells H42, H47 contain formula strings.
- Cell H50 contains a formula string.
- No cells in the target ranges are None.
- Sheet count is unchanged; no new sheets added.

If any target cell is None or missing a formula, debug and fix before finishing.

## 5 – Run verifier if available

```bash
cd /root && python -m pytest test_output.py -v 2>&1 | head -80
```

If tests fail, read the error messages, identify which cells have wrong values, re-inspect the workbook layout, fix the formulas, re-save, and re-run the tests. Common pitfalls:
- Wrong row/column ranges on the Data sheet.
- Block-to-metric mapping is different from assumed order.
- Off-by-one in INDEX/MATCH ranges.
- The summary stat row order (min/max/median/mean/p25/p75) might differ from assumed — check the labels in column D or G of those rows.

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