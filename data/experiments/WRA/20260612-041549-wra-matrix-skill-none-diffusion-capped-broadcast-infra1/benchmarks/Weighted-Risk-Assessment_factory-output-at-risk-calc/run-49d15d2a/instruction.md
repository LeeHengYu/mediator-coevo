# Task Instruction

Execute the following steps to produce /root/output/result.xlsx from /root/data/workbook.xlsx.

## 0 – Inspect the workbook
```bash
pip install openpyxl
```
Then in Python:
```python
import openpyxl, pprint
wb = openpyxl.load_workbook('/root/data/workbook.xlsx')
for s in wb.sheetnames:
    print(f'=== {s} ===')
    ws = wb[s]
    print(f'  dims: {ws.dimensions}')
    for row in ws.iter_rows(min_row=1, max_row=min(50, ws.max_row), values_only=False):
        for c in row:
            if c.value is not None:
                print(f'  {c.coordinate}: {repr(c.value)}')
```
Study:
- Sheet `Task`: column D series codes, row 10 year headers (H10:L10), the three lookup blocks (H12:L17, H19:L24, H26:L31), the derived block labels, and the stat labels in rows 42-47.
- Sheet `Data`: rows 21-38 layout – which column holds the series code, which row holds years, how data is arranged.

Print everything you find so you understand the exact layout before writing any formulas.

## 1 – Write lookup formulas (Step 1)

For every cell in the three yellow blocks (H12:L17, H19:L24, H26:L31):
- The series code is in column D of the same row.
- The year is in the same column's row 10.
- Data lives on sheet `Data` rows 21:38.

Use INDEX/MATCH (safest cross-engine pattern):
```
=INDEX(Data!<data_columns>, MATCH($D12, Data!<series_column>, 0), MATCH(H$10, Data!<year_row>, 0))
```
Adjust the exact ranges after inspecting the Data sheet layout. Lock row/column references appropriately ($D12 for series, H$10 for year) so formulas can fill across the block.

Write these formulas using openpyxl by setting cell.value to the formula string (no leading space, starts with `=`).

## 2 – Derived block: Net production slack (H35:L40)

Identify which of the three lookup blocks corresponds to `Finished Output`, `Scrap And Rework`, and `Rated Production Capacity` by reading the block labels (likely around rows 11, 18, 25 or in column A/B). Then for each cell, e.g. H35:
```
=(H12 - H19) / H26 * 100
```
(Adjust row offsets based on actual block positions for Finished Output, Scrap And Rework, Rated Production Capacity.)

## 3 – Statistics block (H42:L47)

For each column (H through L), rows 42-47 contain MIN, MAX, MEDIAN, AVERAGE (simple mean), 25th percentile, 75th percentile of the six values in H35:H40 (same column).

**CRITICAL (from prior failure):** The PERCENTILE function caused #NAME? errors. Use `PERCENTILE` without `.INC` suffix first. If the evaluation engine still rejects it, try `_xlfn.PERCENTILE.INC` (the prefixed form that openpyxl/modern Excel uses). To be safe, use the `_xlfn.` prefix:

Row 42 (MIN):    `=MIN(H35:H40)`
Row 43 (MAX):    `=MAX(H35:H40)`
Row 44 (MEDIAN): `=MEDIAN(H35:H40)`
Row 45 (MEAN):   `=AVERAGE(H35:H40)`
Row 46 (25th %): `=_xlfn.PERCENTILE.INC(H35:H40,0.25)`
Row 47 (75th %): `=_xlfn.PERCENTILE.INC(H35:H40,0.75)`

Also try reading what the verifier/test expects. If `_xlfn.PERCENTILE.INC` still fails, fall back to `PERCENTILE(H35:H40,0.25)` without prefix. **Test both variants if needed by checking what the evaluation library (likely xlcalc or formulas) recognizes.** You can also check if the test file or verifier script is present at /root/ and inspect it.

## 4 – Weighted mean (H50:L50)

For each column H-L:
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```
This computes the weighted mean of the Net production slack percentages weighted by Rated Production Capacity.

## 5 – Save
```python
import os
os.makedirs('/root/output', exist_ok=True)
wb.save('/root/output/result.xlsx')
```

## 6 – Validate

1. Re-open /root/output/result.xlsx and print all formula cells to confirm they are set correctly.
2. Check for any test or verifier script:
```bash
find /root -name 'test_output.py' -o -name 'verify*' 2>/dev/null
```
If found, read it to understand what it checks, then run it:
```bash
cd /root && python -m pytest test_output.py -v 2>&1 | head -80
```
3. If PERCENTILE formulas cause #NAME? errors, switch between `PERCENTILE`, `_xlfn.PERCENTILE.INC`, and `PERCENTILE.INC` and re-run until tests pass.
4. Confirm all cells in the checked ranges return numeric values (not None, not #NAME?, not #REF!).

## Key Warnings
- Do NOT add sheets, macros, VBA, external links, or helper tabs.
- Do NOT change existing formatting.
- Inspect before writing – verify exact row/column positions from the actual workbook.
- The prior failure was specifically PERCENTILE → #NAME?. Prioritize getting the right function name variant.
- The avoid-recheck artifact warns about cells returning None – make sure every formula cell is actually populated (not skipped due to off-by-one errors in loops).

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