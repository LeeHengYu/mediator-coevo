# Task Instruction

Execute the following steps to produce /root/output/result.xlsx from /root/data/workbook.xlsx.

## 0. Inspect the workbook
```python
import openpyxl, os, shutil
wb = openpyxl.load_workbook('/root/data/workbook.xlsx')
for name in wb.sheetnames:
    print(f'--- {name} ---')
    ws = wb[name]
    for row in ws.iter_rows(min_row=1, max_row=ws.max_row, max_col=ws.max_column, values_only=False):
        for cell in row:
            if cell.value is not None:
                print(f'  {cell.coordinate}: {repr(cell.value)}')
```
Run this first to understand:
- The layout of sheet `Task`: column D series codes, row 10 year headers, yellow target ranges.
- The layout of sheet `Data` rows 21-38: how series codes and years are arranged (row vs column orientation).
- The exact cell references for Finished Output, Scrap And Rework, and Rated Production Capacity blocks.
- Any existing labels in rows 35-50.

## 1. Populate lookup formulas in H12:L17, H19:L24, H26:L31

For each cell in these ranges, write an INDEX+MATCH formula that:
- Matches the series code in column D of that row against the series-code column on sheet `Data` (rows 21:38).
- Matches the year in row 10 of the current column against the year row on sheet `Data`.
- Returns the intersecting value.

Concrete pattern (adjust absolute references after inspecting Data layout):
```
=INDEX(Data!$B$21:$Z$38, MATCH($D12,Data!$A$21:$A$38,0), MATCH(H$10,Data!$B$20:$Z$20,0))
```
Adjust the data range, lookup column/row references to match what you observe in step 0. The key contract: two MATCH calls (one for series code, one for year) feeding INDEX.

## 2. Net production slack in H35:L40

Identify which of the three populated blocks corresponds to:
- Finished Output (likely H12:L17)
- Scrap And Rework (likely H19:L24)
- Rated Production Capacity (likely H26:L31)

Confirm by reading labels in the Task sheet (look at column A or nearby cells for block titles).

For each cell in H35:L40, write:
```
=(H12-H19)/H26*100
```
(Adjust row offsets so each of the 6 plants lines up across the three blocks and the result block.)

## 3. Statistics in H42:L47

For each column (H through L), write these six formulas in rows 42-47. Use only legacy function names (no .INC/.EXC suffixes):
- Row 42 (Min): `=MIN(H35:H40)`
- Row 43 (Max): `=MAX(H35:H40)`
- Row 44 (Median): `=MEDIAN(H35:H40)`
- Row 45 (Mean): `=AVERAGE(H35:H40)`
- Row 46 (25th pctl): `=PERCENTILE(H35:H40,0.25)`
- Row 47 (75th pctl): `=PERCENTILE(H35:H40,0.75)`

**IMPORTANT**: Use `PERCENTILE` not `PERCENTILE.INC` or `PERCENTILE.EXC` — the latter cause #NAME? errors in the verifier. Confirm the order of statistics by checking labels in column A/D/E of rows 42-47; reorder if the labels differ from min/max/median/mean/25th/75th.

## 4. Weighted mean in H50:L50

For each column (H through L):
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```
This computes the weighted mean of the net-production-slack percentages using Rated Production Capacity as weights.

## 5. Save

```python
os.makedirs('/root/output', exist_ok=True)
wb.save('/root/output/result.xlsx')
```

## 6. Validate

Reload the saved workbook and print all formula cells in the Task sheet to confirm:
- H12:L17, H19:L24, H26:L31 contain INDEX+MATCH formulas.
- H35:L40 contain the net slack formulas.
- H42:L47 contain MIN, MAX, MEDIAN, AVERAGE, PERCENTILE (not .INC/.EXC).
- H50:L50 contain SUMPRODUCT formulas.
- No other sheets were added; no macros.

If any test script exists at /root/tests/ or similar, run it:
```bash
cd /root && python -m pytest tests/ -v
```

## Critical Reminders
- Do NOT use PERCENTILE.INC or PERCENTILE.EXC — use PERCENTILE.
- Do NOT add sheets, macros, VBA, external links, or helper tabs.
- Preserve all existing formatting.
- Adapt all cell references to what you actually observe in step 0; do not blindly copy the template references above if the actual layout differs.

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