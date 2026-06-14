# Task Instruction

Execute the following steps exactly in order.

## 0 — Inspect the source workbook

```python
import openpyxl, pprint
wb = openpyxl.load_workbook('/root/data/workbook.xlsx', data_only=True)

# --- Task sheet ---
ts = wb['Task']
print('=== Task sheet: years row 10, cols H-L ===')
for c in range(8, 13):            # H=8 … L=12
    print(f'  col {c} ({ts.cell(10,c).coordinate}): {ts.cell(10,c).value!r}')

print('\n=== Task sheet: series codes D12:D31 ===')
for r in range(12, 32):
    print(f'  D{r}: {ts.cell(r, 4).value!r}')

print('\n=== Task sheet: region labels (for Step 2) ===')
for r in range(35, 41):
    print(f'  row {r}: col C={ts.cell(r,3).value!r}  col D={ts.cell(r,4).value!r}  col E={ts.cell(r,5).value!r}')

print('\n=== Task sheet: row 50 labels ===')
for c in range(1, 8):
    print(f'  ({50},{c}): {ts.cell(50,c).value!r}')

print('\n=== Task sheet: stat labels rows 42-47 ===')
for r in range(42, 48):
    print(f'  row {r}: col C={ts.cell(r,3).value!r}  col D={ts.cell(r,4).value!r}  col E={ts.cell(r,5).value!r}  col F={ts.cell(r,6).value!r}  col G={ts.cell(r,7).value!r}')

# --- Data sheet ---
ds = wb['Data']
print('\n=== Data sheet: header row (row 1) sample ===')
for c in range(1, min(ds.max_column+1, 30)):
    print(f'  col {c}: {ds.cell(1,c).value!r}')

print('\n=== Data sheet: rows 21-38 column A (series codes) ===')
for r in range(21, 39):
    vals = [ds.cell(r, c).value for c in range(1, min(ds.max_column+1, 10))]
    print(f'  row {r}: {vals}')

print('\n=== Data sheet: first 3 rows to understand layout ===')
for r in range(1, 4):
    vals = [ds.cell(r, c).value for c in range(1, min(ds.max_column+1, 30))]
    print(f'  row {r}: {vals}')

print('\n=== Data sheet dimensions:', ds.min_row, ds.max_row, ds.min_column, ds.max_column)
wb.close()
```

Print ALL output. Read it carefully before proceeding.

## 1 — Determine exact layout of Data rows 21-38

From the printed output, identify:
- Which column holds the series codes in the Data sheet (call it SC_COL).
- Which row holds the year headers in the Data sheet (call it YEAR_ROW).
- Which columns hold the numeric data years.

Also confirm the exact text of the series codes in Task!D12:D31 and Data rows 21-38. Check for leading/trailing spaces, different quote characters, or number-vs-string mismatches.

## 2 — Write formulas using openpyxl (formula mode)

Open the workbook with `openpyxl.load_workbook('/root/data/workbook.xlsx')` (NOT data_only). Do NOT create or delete any sheets.

### Step 1 — Lookup formulas in H12:L17, H19:L24, H26:L31

For each cell in these three 6×5 blocks, write an INDEX/MATCH formula that:
- Uses the series code from column D of the current row (e.g., `$D12`).
- Uses the year from row 10 of the current column (e.g., `H$10`).
- Looks up in the Data sheet rows 21-38.

Construct the formula referencing the exact columns you discovered in Step 0. Use absolute references for the Data ranges. Example pattern (adapt column letters and row numbers to match actual layout):

```
=INDEX(Data!$B$21:$F$38, MATCH($D12, Data!$A$21:$A$38, 0), MATCH(H$10, Data!$B$20:$F$20, 0))
```

Adjust `$A$21:$A$38` to the actual series-code column, `$B$20:$F$20` to the actual year-header row and columns, and `$B$21:$F$38` to the actual data range.

IMPORTANT: If the years in Data are stored as numbers (e.g., 2019) and the years in Task row 10 are also numbers, the MATCH will work directly. If one is text and the other numeric, wrap the lookup value in VALUE() or TEXT() as needed.

### Step 2 — Net reliability gap (H35:L40)

The three blocks correspond to:
- Successful API Requests: rows 12-17 (H12:L17)
- Failed API Requests: rows 19-24 (H19:L24)
- Compute Capacity: rows 26-31 (H26:L31)

For each cell in H35:L40:
```
=(H12-H19)/H26*100
```
(adjusting row offsets: row 35→uses rows 12,19,26; row 36→13,20,27; etc.)

### Statistics (H42:L47)

For each column (H through L), in rows 42-47, place:
- Row 42: `=MIN(H35:H40)`
- Row 43: `=MAX(H35:H40)`
- Row 44: `=MEDIAN(H35:H40)`
- Row 45: `=AVERAGE(H35:H40)`
- Row 46: `=PERCENTILE(H35:H40, 0.25)`
- Row 47: `=PERCENTILE(H35:H40, 0.75)`

Check the labels in rows 42-47 from Step 0 output to match the correct statistic to the correct row. The order above is: min, max, median, mean, 25th, 75th. Adjust if the labels say otherwise.

### Step 3 — Weighted mean (H50:L50)

For each column:
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```

## 3 — Save

Save to `/root/output/result.xlsx`. Create the output directory if needed.

```python
import os
os.makedirs('/root/output', exist_ok=True)
wb.save('/root/output/result.xlsx')
```

## 4 — Validate

Reopen the saved file in data_only=False mode and print the formulas in a sample of cells (H12, L17, H19, H26, H35, H40, H42, H47, H50, L50) to confirm they are present and correctly structured.

Then, if a test script exists at `/root/test_output.py` or similar, run it:
```bash
cd /root && python -m pytest test_output.py -v
```

If the test fails, read the error output carefully, diagnose the issue, fix the formulas, re-save, and re-test. Common pitfalls:
- Off-by-one in row/column references
- Series code column mismatch
- Year header row mismatch
- Text vs number type mismatch in MATCH lookup values
- Wrong statistic assigned to wrong row

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