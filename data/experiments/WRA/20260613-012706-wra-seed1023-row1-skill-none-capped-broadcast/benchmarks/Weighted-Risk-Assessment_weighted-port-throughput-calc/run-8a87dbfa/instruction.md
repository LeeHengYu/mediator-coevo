# Task Instruction

Produce /root/output/result.xlsx from /root/data/workbook.xlsx by following these precise steps.

## 0 – Inspect the workbook
```bash
cd /root && python3 - <<'PY'
import openpyxl, pprint
wb = openpyxl.load_workbook('data/workbook.xlsx', data_only=False)
for name in wb.sheetnames:
    print(f'=== {name} ===')
    ws = wb[name]
    for row in ws.iter_rows(min_row=1, max_row=ws.max_row, max_col=ws.max_column):
        vals = [(c.coordinate, c.value) for c in row if c.value is not None]
        if vals:
            print(vals)
PY
```
Study the output carefully. Identify:
- The series codes in column D for rows 12-17, 19-24, 26-31.
- The years in row 10 (columns H-L).
- The layout of Data!A21:?38 (column headers, row labels).
- What labels appear in rows 35-40, 42-47, 50.

## 1 – Populate lookup formulas (H12:L17, H19:L24, H26:L31)

For every yellow cell in those ranges, write a formula that looks up the value from the Data sheet using the series code in column D of that row and the year in row 10 of that column.

Use the INDEX/MATCH pattern:
```
=INDEX(Data!$B$21:$<lastcol>$38, MATCH($D<row>,Data!$A$21:$A$38,0), MATCH(H$10,Data!$B$20:$<lastcol>$20,0))
```
Adjust the exact range references (columns, header row for years) after inspecting the Data sheet layout. Lock row/column references with $ so the formula copies correctly across the 5 columns and down the rows.

## 2 – Net container flow (H35:L40)

For each of the 6 ports (rows 35-40) and each year column (H-L), write:
```
=(H12-H19)/H26*100
```
where H12 is the Loaded Containers Inbound cell, H19 is the Loaded Containers Outbound cell, and H26 is the Terminal Throughput Capacity cell for the same port and year. Adjust row references per port (row 35 uses rows 12,19,26; row 36 uses 13,20,27; etc.).

## 3 – Statistics block (H42:L47)

For each year column (H through L), write exactly these formulas. **Use only the legacy function names below** (no `.INC`/`.EXC` suffixes, no `_xlfn.` prefix):

| Row | Statistic | Formula (column H example) |
|-----|-----------|----------------------------|
| 42  | Minimum   | `=MIN(H35:H40)` |
| 43  | Maximum   | `=MAX(H35:H40)` |
| 44  | Median    | `=MEDIAN(H35:H40)` |
| 45  | Mean      | `=AVERAGE(H35:H40)` |
| 46  | 25th pctl | `=PERCENTILE(H35:H40,0.25)` |
| 47  | 75th pctl | `=PERCENTILE(H35:H40,0.75)` |

**Critical**: Do NOT use `PERCENTILE.INC` — use `PERCENTILE`. This avoids #NAME? errors in the evaluation harness.

## 4 – Weighted mean for CPA (H50:L50)

For each year column, write:
```
=SUMPRODUCT(H35:H40,H26:H31)/SUM(H26:H31)
```
This computes the weighted mean of the net-container-flow percentages using Terminal Throughput Capacity as weights.

## 5 – Save

```python
import os, shutil, openpyxl
os.makedirs('/root/output', exist_ok=True)
# (after all formulas are written)
wb.save('/root/output/result.xlsx')
```

## 6 – Validate

After saving, re-open the file and print every formula in the key ranges to confirm:
- No `_xlfn.` prefixes appear.
- No `.INC` or `.EXC` suffixes on PERCENTILE.
- All INDEX/MATCH references point to valid Data sheet ranges.
- Net container flow formulas reference the correct inbound/outbound/capacity rows.
- SUMPRODUCT weighted mean formulas are present in H50:L50.

Run the test suite if available:
```bash
cd /root && python3 -m pytest tests/ -v 2>&1 | head -80
```

If any cell shows #NAME? or tests fail, diagnose and fix before finishing. Do NOT use PERCENTILE.INC, PERCENTILE.EXC, XLOOKUP (unless you verify it works), or any function with `_xlfn.` prefix.

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
Task metadata: author_email=catpaw@meituan.com, author_name=CatPaw Task Engineer, category=spreadsheet-formula-reuse, difficulty=medium, tags=[excel, formulas, lookup, statistics, weighted-mean].
Verifier config: timeout_sec=600.0.