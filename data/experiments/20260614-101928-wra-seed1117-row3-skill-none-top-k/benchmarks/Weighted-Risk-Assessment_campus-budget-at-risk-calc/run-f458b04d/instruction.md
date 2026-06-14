# Task Instruction

Execute the following steps exactly, in order.

## 0 — Inspect the workbook
```bash
cd /root
pip install openpyxl 2>/dev/null
python3 << 'PYEOF'
import openpyxl, json
wb = openpyxl.load_workbook('data/workbook.xlsx', data_only=False)
for s in wb.sheetnames:
    print(f'=== Sheet: {s} ===')
    ws = wb[s]
    print(f'  dims: {ws.dimensions}')
    # Print first 55 rows, cols A-M for Task sheet
    if s == 'Task':
        for r in range(1, 56):
            vals = []
            for c in range(1, 14):
                cell = ws.cell(r, c)
                v = cell.value
                vals.append(str(v) if v is not None else '')
            print(f'  R{r:02d}: {vals}')
    if s == 'Data':
        for r in range(1, 45):
            vals = []
            for c in range(1, 14):
                cell = ws.cell(r, c)
                v = cell.value
                vals.append(str(v) if v is not None else '')
            print(f'  R{r:02d}: {vals}')
wb.close()
PYEOF
```

## 1 — Read the test file
```bash
cat /root/test_output*.py 2>/dev/null || cat /root/tests/test_output*.py 2>/dev/null || find /root -name 'test_output*' -exec cat {} \;
```

## 2 — Understand the Data sheet layout
Note which column holds series codes and which rows (21-38) hold the source data. Identify the column layout (years in which columns, values in which columns).

## 3 — Write the formulas
Create a Python script that:

a) Opens `/root/data/workbook.xlsx` preserving formatting (use openpyxl, do NOT use data_only).

b) **Step 1 — Lookup formulas in H12:L17, H19:L24, H26:L31:**
   For each cell in these ranges, write an `INDEX(MATCH,MATCH)` formula that:
   - Uses the series code from column D of the same row
   - Uses the year from row 10 of the same column
   - Looks up in the Data sheet rows 21:38
   - Use absolute references for the Data range, relative for the row's series code and column's year
   - Pattern: `=INDEX(Data!$B$21:$<lastcol>$38,MATCH($D<row>,Data!$A$21:$A$38,0),MATCH(<colref>10,Data!$B$20:$<lastcol>$20,0))`
   - Adjust the exact column letters after inspecting the Data sheet layout in step 0.

c) **Step 2 — Net budget buffer in H35:L40:**
   Formula: `=(<CommittedFunding> - <OperatingSpend>) / <ApprovedBudgetBase> * 100`
   where CommittedFunding is from H12:L17, OperatingSpend from H19:L24, ApprovedBudgetBase from H26:L31, matching by row offset (row 35 uses row 12, 19, 26; row 36 uses row 13, 20, 27; etc.).

d) **Step 2 — Statistics in H42:L47:**
   - Row 42: `=MIN(H35:H40)` (column-wise)
   - Row 43: `=MAX(H35:H40)`
   - Row 44: `=MEDIAN(H35:H40)`
   - Row 45: `=AVERAGE(H35:H40)`
   - Row 46: `=PERCENTILE(H35:H40,0.25)` — use `PERCENTILE` not `PERCENTILE.INC`
   - Row 47: `=PERCENTILE(H35:H40,0.75)` — use `PERCENTILE` not `PERCENTILE.INC`
   Adjust column letter for each column H through L.

e) **Step 3 — Weighted mean in H50:L50:**
   `=SUMPRODUCT(H35:H40,H26:H31)/SUM(H26:H31)` for each column H-L.

f) Save to `/root/output/result.xlsx`. Create `/root/output/` if needed.

**CRITICAL**: Use the function name `PERCENTILE` (not `PERCENTILE.INC` or `PERCENTILE.EXC`) for the 25th and 75th percentile rows. This was the cause of the previous `#NAME?` failure.

**CRITICAL**: Do NOT add new sheets, macros, VBA, or external links. Do NOT change existing formatting or cell values outside the specified ranges.

## 4 — Validate
```bash
python3 << 'PYEOF'
import openpyxl
wb = openpyxl.load_workbook('/root/output/result.xlsx', data_only=False)
ws = wb['Task']
for r in [12,13,17,19,24,26,31,35,40,42,43,44,45,46,47,50]:
    vals = []
    for c in range(8, 13):  # H=8 to L=12
        v = ws.cell(r, c).value
        vals.append(str(v)[:60] if v is not None else 'NONE')
    print(f'Row {r}: {vals}')
wb.close()
PYEOF
```
Verify that:
- Rows 12-17, 19-24, 26-31 contain INDEX/MATCH formulas
- Rows 35-40 contain arithmetic formulas referencing the blocks above
- Rows 42-47 contain MIN/MAX/MEDIAN/AVERAGE/PERCENTILE formulas
- Row 50 contains SUMPRODUCT formulas
- No cell shows NONE

## 5 — Run the test
```bash
cd /root && python -m pytest test_output*.py -v 2>/dev/null || python -m pytest tests/test_output*.py -v 2>/dev/null || echo 'Could not find test file'
```

If any test fails, read the error carefully, fix the formulas, and re-run. Pay special attention to:
- Exact Data sheet range references (row/column boundaries)
- Whether the Data sheet uses row 20 for headers or a different row
- Whether series codes match exactly between Task column D and Data column A
- Function name spelling

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