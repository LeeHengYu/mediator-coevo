# Task Instruction

Execute the following steps exactly:

## 1. Inspect the workbook
```bash
pip install openpyxl
python3 << 'PYEOF'
import openpyxl, json
wb = openpyxl.load_workbook('/root/data/workbook.xlsx', data_only=False)
for name in wb.sheetnames:
    print(f'Sheet: {name}')
ws_task = wb['Task']
ws_data = wb['Data']

# Print Task sheet layout: rows 1-55, columns A-M
print('\n=== Task sheet ===')
for row in ws_task.iter_rows(min_row=1, max_row=55, min_col=1, max_col=13, values_only=False):
    vals = []
    for c in row:
        v = c.value
        if v is not None:
            vals.append(f'{c.coordinate}: {repr(v)}')
    if vals:
        print('; '.join(vals))

# Print Data sheet rows 1-5 and 18-40 to understand structure
print('\n=== Data sheet (rows 1-5) ===')
for row in ws_data.iter_rows(min_row=1, max_row=5, min_col=1, max_col=20, values_only=False):
    vals = []
    for c in row:
        v = c.value
        if v is not None:
            vals.append(f'{c.coordinate}: {repr(v)}')
    if vals:
        print('; '.join(vals))

print('\n=== Data sheet (rows 18-40) ===')
for row in ws_data.iter_rows(min_row=18, max_row=40, min_col=1, max_col=20, values_only=False):
    vals = []
    for c in row:
        v = c.value
        if v is not None:
            vals.append(f'{c.coordinate}: {repr(v)}')
    if vals:
        print('; '.join(vals))

wb.close()
PYEOF
```

## 2. After inspecting the output, write the formulas

Based on the inspection, write a Python script that:

a) Opens `/root/data/workbook.xlsx` with openpyxl (NOT data_only).

b) **Step 1 — Lookup formulas in H12:L17, H19:L24, H26:L31:**
   For each cell in these ranges, write an `INDEX(MATCH,MATCH)` formula that:
   - Uses the series code from column D of the same row
   - Uses the year from row 10 of the same column
   - Looks up in the Data sheet rows 21:38
   - Determine the exact Data sheet layout from inspection: identify which column has the series codes and which row has the years, then construct the INDEX/MATCH formula accordingly.
   - Example pattern: `=INDEX(Data!$B$21:$Z$38,MATCH($D12,Data!$A$21:$A$38,0),MATCH(H$10,Data!$B$20:$Z$20,0))` — adjust column/row references based on actual Data sheet structure.

c) **Step 2 — Net budget buffer in H35:L40:**
   The formula is: `(Committed Funding - Operating Spend) / Approved Budget Base * 100`
   - Committed Funding is in H12:L17
   - Operating Spend is in H19:L24  
   - Approved Budget Base is in H26:L31
   (Verify the actual block labels from the inspection and adjust if needed.)
   For each cell, e.g. H35: `=(H12-H19)/H26*100` (mapping row 35→12,19,26; row 36→13,20,27; etc.)

d) **Step 2 — Summary statistics in H42:L47:**
   For each column H through L:
   - Row 42 (MIN): `=MIN(H35:H40)`
   - Row 43 (MAX): `=MAX(H35:H40)`
   - Row 44 (MEDIAN): `=MEDIAN(H35:H40)`
   - Row 45 (MEAN): `=AVERAGE(H35:H40)`
   - Row 46 (25th percentile): `=PERCENTILE(H35:H40,0.25)` — use the legacy `PERCENTILE` function, NOT `PERCENTILE.INC` or `PERCENTILE.EXC`
   - Row 47 (75th percentile): `=PERCENTILE(H35:H40,0.75)` — use the legacy `PERCENTILE` function

e) **Step 3 — Weighted mean in H50:L50:**
   For each column H through L:
   `=SUMPRODUCT(H35:H40,H26:H31)/SUM(H26:H31)`

f) Save to `/root/output/result.xlsx`. Create `/root/output/` directory if needed.

## 3. Validate the output
```bash
python3 << 'PYEOF'
import openpyxl
wb = openpyxl.load_workbook('/root/output/result.xlsx', data_only=False)
ws = wb['Task']

# Check a sample of cells have formulas
for coord in ['H12','L17','H19','L24','H26','L31','H35','L40','H42','H47','H46','L46','H50','L50']:
    c = ws[coord]
    print(f'{coord}: {repr(c.value)}')

wb.close()
PYEOF
```

## Critical Notes
- **PERCENTILE function**: Use `PERCENTILE(range, k)` — the legacy form. Do NOT use `PERCENTILE.INC` or `PERCENTILE.EXC`. The previous run failed because of #NAME? errors from using a non-recognized percentile variant.
- Do NOT add new sheets, macros, VBA, external links, or helper tabs.
- Do NOT change existing formatting.
- Verify the exact layout of the Data sheet before writing formulas. The row/column references for INDEX/MATCH must match the actual data structure.
- The label-to-block mapping (which rows are Committed Funding, Operating Spend, Approved Budget Base) must be confirmed from the Task sheet inspection. Adjust the formula row offsets accordingly if labels differ from the assumed order.
- After writing the script, re-read a few cells to confirm formulas were written correctly.

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