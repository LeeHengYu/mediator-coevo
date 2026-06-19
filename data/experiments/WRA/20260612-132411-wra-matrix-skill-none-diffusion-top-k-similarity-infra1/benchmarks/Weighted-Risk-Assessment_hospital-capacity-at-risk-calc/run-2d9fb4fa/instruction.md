# Task Instruction

Execute the following steps to produce /root/output/result.xlsx from /root/data/workbook.xlsx.

## 0 — Inspect the workbook
```python
import openpyxl, os, json
wb = openpyxl.load_workbook('/root/data/workbook.xlsx')
for name in wb.sheetnames:
    print(f'--- {name} ---')
    ws = wb[name]
    print(f'  dims: {ws.dimensions}')
    # Print first 55 rows of Task, first 40 rows of Data (columns A-M)
    if name == 'Task':
        for row in ws.iter_rows(min_row=1, max_row=55, max_col=13, values_only=False):
            print([(c.coordinate, c.value) for c in row])
    if name == 'Data':
        for row in ws.iter_rows(min_row=1, max_row=40, max_col=20, values_only=False):
            print([(c.coordinate, c.value) for c in row])
```
Read and understand:
- The series codes in column D (rows 12-17, 19-24, 26-31).
- The years in row 10 (columns H-L).
- The Data sheet layout in rows 21-38 (which column holds the series code, which columns hold year data, or which row holds years).
- The labels in rows 35-50 to confirm what goes where.

## 1 — Write the lookup formulas (H12:L17, H19:L24, H26:L31)

Use `INDEX(MATCH,MATCH)` pattern referencing the Data sheet rows 21:38. The exact references depend on what you find in step 0, but the pattern should be:

```
=INDEX(Data!<data_range>, MATCH($D12, Data!<series_code_column>, 0), MATCH(H$10, Data!<year_row>, 0))
```

Adjust the ranges after inspecting the Data sheet. The row lookup key is the series code in column D of the current row (use $D with absolute column). The column lookup key is the year in row 10 (use absolute row H$10). Fill all 5 columns × 18 rows (three blocks of 6 rows each).

## 2 — Net capacity headroom (H35:L40)

For each of the 6 hospital clusters (rows 35-40) and each year column (H-L):
```
=(H12 - H19) / H26 * 100
```
where row 12→Available Care Slots, row 19→Occupied Care Slots, row 26→Staffed Bed Capacity. Adjust row references per cluster (row 35 uses rows 12,19,26; row 36 uses rows 13,20,27; etc.).

## 3 — Summary statistics (H42:L47)

For each column H through L, in the six rows 42-47 place:
- Row 42: `=MIN(H35:H40)`
- Row 43: `=MAX(H35:H40)`
- Row 44: `=MEDIAN(H35:H40)`
- Row 45: `=AVERAGE(H35:H40)`
- Row 46: `=PERCENTILE(H35:H40,0.25)`
- Row 47: `=PERCENTILE(H35:H40,0.75)`

**Important**: Use `PERCENTILE` (not `PERCENTILE.INC` or `PERCENTILE.EXC`) to avoid #NAME? errors. Check the labels in column D/E/F/G of rows 42-47 to confirm which row is which statistic and adjust accordingly.

## 4 — Weighted mean (H50:L50)

For each column H through L:
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```

## 5 — Save
```python
os.makedirs('/root/output', exist_ok=True)
wb.save('/root/output/result.xlsx')
```

## 6 — Validate
Reload /root/output/result.xlsx with data_only=False. Print cells in the formula regions to confirm formulas are present (not None, not hardcoded values). Spot-check a few cells. Then if there is a test script in /root or /tests, run it:
```bash
find /root -name 'test_output*' -o -name 'test_outputs*' 2>/dev/null
# then run: python -m pytest <path> -v
```

## Key Constraints
- Do NOT add sheets, macros, VBA, external links, or helper tabs.
- Do NOT change existing formatting.
- Use openpyxl to write formulas as strings (e.g., cell.value = '=INDEX(...)').
- Use PERCENTILE (not PERCENTILE.INC/EXC) for the percentile functions.
- Adapt all cell references based on what you actually observe in step 0. Do not assume — inspect first.

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