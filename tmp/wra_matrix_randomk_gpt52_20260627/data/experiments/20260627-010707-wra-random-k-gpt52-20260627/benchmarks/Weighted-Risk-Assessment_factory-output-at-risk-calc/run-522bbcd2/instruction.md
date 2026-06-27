# Task Instruction

Execute the following steps exactly, in order.

## 0. Inspect the workbook
```bash
cd /root/data
python3 - <<'PYEOF'
import openpyxl
wb = openpyxl.load_workbook('workbook.xlsx')
for s in wb.sheetnames:
    print(f'=== Sheet: {s} ===')
    ws = wb[s]
    print(f'  Dimensions: {ws.dimensions}')
    for row in ws.iter_rows(min_row=1, max_row=min(ws.max_row,50), values_only=False):
        vals = [(c.coordinate, c.value) for c in row if c.value is not None]
        if vals:
            print(f'  {vals}')
PYEOF
```
Read the output carefully. Identify:
- The series codes in column D (rows 12-17, 19-24, 26-31).
- The years in row 10 (columns H-L).
- The layout of the Data sheet rows 21-38: which row holds series codes, which row holds years, orientation of the table.
- The labels for the three blocks (Finished Output, Scrap And Rework, Rated Production Capacity) — confirm which row ranges map to which.
- The structure of rows 35-40 (Net production slack), 42-47 (stats), and 50 (weighted mean).

## 1. Write the formulas
Create a Python script that uses openpyxl to open `/root/data/workbook.xlsx`, populate the formulas, and save to `/root/output/result.xlsx`.

Key rules:
- Use `INDEX(MATCH,MATCH)` pattern for lookups. The MATCH functions should reference the series code in column D of the current row and the year in row 10.
- For the Data sheet lookup range in rows 21:38, use absolute references for the data block, the row header (series codes column), and the column header (years row). Make sure the MATCH for the series code searches the series-code column of Data!21:38, and the MATCH for the year searches the years row of Data!21:38.
- Use classic Excel function names: `PERCENTILE` (not `PERCENTILE.INC`), `AVERAGE` (not `MEAN`), `MEDIAN`, `MIN`, `MAX`.
- For Step 2 (H35:L40): `(block1 - block2) / block3 * 100` where block1 = Finished Output (H12:L17), block2 = Scrap And Rework (H19:L24), block3 = Rated Production Capacity (H26:L31). Use direct cell references, e.g., for H35: `=(H12-H19)/H26*100`.
- For Step 2 stats (H42:L47): For each column (H through L):
  - Row 42: `=MIN(H35:H40)`
  - Row 43: `=MAX(H35:H40)`
  - Row 44: `=MEDIAN(H35:H40)`
  - Row 45: `=AVERAGE(H35:H40)`
  - Row 46: `=PERCENTILE(H35:H40,0.25)`
  - Row 47: `=PERCENTILE(H35:H40,0.75)`
  Check the actual labels in column D/E/F/G of rows 42-47 to confirm the correct order (min, max, median, mean, 25th, 75th). Adjust the row assignments to match whatever labels are actually present.
- For Step 3 (H50:L50): `=SUMPRODUCT(H35:H40,H26:H31)/SUM(H26:H31)` for each column.
- Do NOT change any existing cell values, formatting, or styles. Only write formulas into the yellow target cells.
- Do NOT add sheets, macros, VBA, or external links.

## 2. Validate
```bash
mkdir -p /root/output
python3 - <<'PYEOF'
import openpyxl
wb = openpyxl.load_workbook('/root/output/result.xlsx')
ws = wb['Task']
# Check that formula cells are populated
for r in range(12,18):
    for c in ['H','I','J','K','L']:
        v = ws[f'{c}{r}'].value
        assert v is not None, f'{c}{r} is None'
        assert str(v).startswith('='), f'{c}{r} not a formula: {v}'
for r in range(19,25):
    for c in ['H','I','J','K','L']:
        v = ws[f'{c}{r}'].value
        assert v is not None and str(v).startswith('='), f'{c}{r} bad: {v}'
for r in range(26,32):
    for c in ['H','I','J','K','L']:
        v = ws[f'{c}{r}'].value
        assert v is not None and str(v).startswith('='), f'{c}{r} bad: {v}'
for r in range(35,41):
    for c in ['H','I','J','K','L']:
        v = ws[f'{c}{r}'].value
        assert v is not None and str(v).startswith('='), f'{c}{r} bad: {v}'
for r in range(42,48):
    for c in ['H','I','J','K','L']:
        v = ws[f'{c}{r}'].value
        assert v is not None and str(v).startswith('='), f'{c}{r} bad: {v}'
for c in ['H','I','J','K','L']:
    v = ws[f'{c}50'].value
    assert v is not None and str(v).startswith('='), f'{c}50 bad: {v}'
print('All formula cells populated correctly.')
PYEOF
```

If any assertion fails, re-read the workbook structure and fix.

## 3. Run the verifier if available
```bash
cd /root && python3 -m pytest test_output.py -v 2>&1 || true
```
If tests fail, read the error messages, fix the formulas, re-save, and re-run until all pass.

IMPORTANT: Before writing any formulas, carefully inspect the Data sheet to understand its exact layout — particularly which column contains series codes and which row contains years. The INDEX/MATCH references must exactly match the Data sheet structure. A common failure mode is targeting wrong rows/columns in MATCH.

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