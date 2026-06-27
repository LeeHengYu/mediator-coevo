# Task Instruction

Execute the following steps exactly:

## 0. Inspect the workbook and data layout
```bash
cd /root && pip install openpyxl 2>/dev/null
python3 - <<'PYEOF'
import openpyxl
wb = openpyxl.load_workbook('data/workbook.xlsx')
for name in wb.sheetnames:
    print(f'Sheet: {name}')
ws_task = wb['Task']
print('\n--- Task sheet, rows 1-55, cols A-M ---')
for row in ws_task.iter_rows(min_row=1, max_row=55, min_col=1, max_col=13, values_only=False):
    vals = [(c.coordinate, c.value) for c in row if c.value is not None]
    if vals:
        print(vals)
ws_data = wb['Data']
print('\n--- Data sheet, rows 1-5 (headers) ---')
for row in ws_data.iter_rows(min_row=1, max_row=5, min_col=1, max_col=20, values_only=False):
    vals = [(c.coordinate, c.value) for c in row if c.value is not None]
    if vals:
        print(vals)
print('\n--- Data sheet, rows 19-40 ---')
for row in ws_data.iter_rows(min_row=19, max_row=40, min_col=1, max_col=20, values_only=False):
    vals = [(c.coordinate, c.value) for c in row if c.value is not None]
    if vals:
        print(vals)
PYEOF
```

## 1. Inspect the test file to understand the verifier contract
```bash
cat /root/tests/test_output*.py 2>/dev/null || cat /root/tests/*.py 2>/dev/null | head -200
```

## 2. Build and save the result workbook

After inspecting the layout (column positions, series codes in column D, years in row 10, Data sheet row 21-38 structure), write a Python script that:

### Step 1 – Lookup formulas in H12:L17, H19:L24, H26:L31
For each cell in these blocks, write an INDEX/MATCH formula that:
- Uses the series code from column D of the same row on Task sheet
- Uses the year from row 10 of the same column on Task sheet  
- Looks up data from the Data sheet rows 21:38
- Pattern: `=INDEX(Data!<data_range>, MATCH($D{row}, Data!<series_col>, 0), MATCH(H$10, Data!<year_row>, 0))`
- Adjust the exact ranges based on what you see in the inspection output.

### Step 2 – Net budget buffer in H35:L40
For each cell, compute: `(<Committed Funding cell> - <Operating Spend cell>) / <Approved Budget Base cell> * 100`
- Committed Funding is in H12:L17, Operating Spend is in H19:L24, Approved Budget Base is in H26:L31 (confirm from inspection).
- The formula for H35 would be something like: `=(H12-H19)/H26*100`

### Step 2 continued – Summary statistics in H42:L47
For each column (H through L):
- Row 42: `=MIN(H35:H40)`
- Row 43: `=MAX(H35:H40)`
- Row 44: `=MEDIAN(H35:H40)`
- Row 45: `=AVERAGE(H35:H40)`
- Row 46: `=PERCENTILE(H35:H40,0.25)`  ← Use PERCENTILE (legacy), NOT PERCENTILE.INC
- Row 47: `=PERCENTILE(H35:H40,0.75)`  ← Use PERCENTILE (legacy), NOT PERCENTILE.INC

**CRITICAL**: Use exactly `PERCENTILE` — not `PERCENTILE.INC`, not `PERCENTILE.EXC`. The openpyxl/evaluation engine may not recognize the dotted variants. This was the exact cause of the previous failure (#NAME? errors in H46:L47).

### Step 3 – Weighted mean in H50:L50
For each column: `=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)`

### Save
- Save to `/root/output/result.xlsx`
- Create the output directory if needed.
- Do NOT change formatting, do NOT add sheets.

## 3. Validate
```bash
python3 - <<'PYEOF'
import openpyxl
wb = openpyxl.load_workbook('/root/output/result.xlsx')
ws = wb['Task']
# Check a sample of cells for formulas (not None, not #NAME?)
check_cells = ['H12','L17','H19','L24','H26','L31','H35','L40','H42','H45','H46','H47','H50','L50']
for c in check_cells:
    val = ws[c].value
    print(f'{c}: {val}')
    assert val is not None, f'{c} is empty!'
    if isinstance(val, str):
        assert '#NAME' not in val.upper(), f'{c} has #NAME? error'
        assert 'PERCENTILE.INC' not in val.upper(), f'{c} uses PERCENTILE.INC — use PERCENTILE instead'
        assert 'PERCENTILE.EXC' not in val.upper(), f'{c} uses PERCENTILE.EXC — use PERCENTILE instead'
print('All checks passed.')
PYEOF
```

## 4. Run the verifier
```bash
cd /root && python -m pytest tests/ -v 2>&1 | tail -40
```

If the verifier fails, read the error output carefully, re-inspect the workbook, and fix the specific issue. Pay special attention to:
- Whether the block labels (Committed Funding, Operating Spend, Approved Budget Base) match the row ranges assumed
- Whether the Data sheet lookup ranges are correct
- Whether PERCENTILE formulas are spelled correctly

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