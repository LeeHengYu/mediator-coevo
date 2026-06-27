# Task Instruction

Execute the following steps exactly, in order.

## 0. Inspect the workbook
```bash
cd /root && python3 - <<'PY'
import openpyxl, json
wb = openpyxl.load_workbook('data/workbook.xlsx')
for name in wb.sheetnames:
    ws = wb[name]
    print(f'=== Sheet: {name}  rows={ws.max_row} cols={ws.max_column} ===')
    # Print first 50 rows (or all if fewer)
    for r in ws.iter_rows(min_row=1, max_row=min(ws.max_row, 55), values_only=False):
        vals = [(c.coordinate, c.value) for c in r if c.value is not None]
        if vals:
            print(vals)
PY
```
Read the output carefully. Identify:
- The series codes in column D of the Task sheet (rows 12-17, 19-24, 26-31).
- The years in row 10 (columns H-L).
- The Data sheet layout, especially rows 21-38.
- The exact labels/layout of rows 35-47 and row 50 on the Task sheet.

## 1. Write the workbook with formulas
```bash
python3 - <<'PYEOF'
import openpyxl
from copy import copy

wb = openpyxl.load_workbook('/root/data/workbook.xlsx')
ts = wb['Task']
ds = wb['Data']

# ── Step 1: Populate H12:L17, H19:L24, H26:L31 with lookup formulas ──
# Each cell uses the series code in col D of same row and the year in row 10.
# Data is on sheet Data rows 21:38.
# Use INDEX/MATCH pattern:
#   =INDEX(Data!$B$21:$Z$38, MATCH(D12,Data!$A$21:$A$38,0), MATCH(H$10,Data!$B$20:$Z$20,0))
# Adjust the column range based on inspection. We'll figure out the Data layout first.

# Determine Data sheet dimensions for the lookup range
# Find the header row for years on Data sheet (likely row 20 or the row just above 21)
# Let's find the last used column on Data sheet
data_max_col = ds.max_column
from openpyxl.utils import get_column_letter
last_col_letter = get_column_letter(data_max_col)

# Identify the year-header row on Data sheet (row above 21, i.e., row 20)
# Check what's in row 20
print('Data row 20:', [ds.cell(row=20, column=c).value for c in range(1, data_max_col+1)])
print('Data row 21:', [ds.cell(row=21, column=c).value for c in range(1, min(data_max_col+1, 10))])

# Series codes column on Data = column A (or column 1)
# Values start from column B (or column 2)
# Year headers in row 20 starting from column B

# Build the lookup formula using INDEX+MATCH
# Data range for values: Data!$B$21:${last_col}$38
# Series code range: Data!$A$21:$A$38
# Year header range: Data!$B$20:${last_col}$20

val_range = f"Data!$B$21:${last_col_letter}$38"
code_range = f"Data!$A$21:$A$38"
year_range = f"Data!$B$20:${last_col_letter}$20"

row_blocks = list(range(12,18)) + list(range(19,25)) + list(range(26,32))
col_letters = ['H','I','J','K','L']

for row in row_blocks:
    for cl in col_letters:
        formula = f'=INDEX({val_range},MATCH($D{row},{code_range},0),MATCH({cl}$10,{year_range},0))'
        ts[f'{cl}{row}'] = formula

# ── Step 2: Net capacity headroom H35:L40 ──
# (Available Care Slots - Occupied Care Slots) / Staffed Bed Capacity * 100
# Available Care Slots = rows 12:17, Occupied Care Slots = rows 19:24, Staffed Bed Capacity = rows 26:31
# Row mapping: cluster 1 → row 12/19/26 mapped to row 35, etc.
for i in range(6):
    src_avail = 12 + i
    src_occup = 19 + i
    src_staff = 26 + i
    dest_row = 35 + i
    for cl in col_letters:
        formula = f'=({cl}{src_avail}-{cl}{src_occup})/{cl}{src_staff}*100'
        ts[f'{cl}{dest_row}'] = formula

# ── Summary statistics H42:L47 ──
# Row 42: MIN, 43: MAX, 44: MEDIAN, 45: AVERAGE, 46: 25th percentile, 47: 75th percentile
# Check the labels in column D/E/F/G for rows 42-47 to confirm ordering
print('Task rows 42-47 labels:')
for r in range(42, 48):
    vals = [ts.cell(row=r, column=c).value for c in range(1, 8)]
    print(f'  Row {r}: {vals}')

for cl in col_letters:
    rng = f'{cl}35:{cl}40'
    ts[f'{cl}42'] = f'=MIN({rng})'
    ts[f'{cl}43'] = f'=MAX({rng})'
    ts[f'{cl}44'] = f'=MEDIAN({rng})'
    ts[f'{cl}45'] = f'=AVERAGE({rng})'
    # Use PERCENTILE.INC for 25th and 75th - standard Excel function
    ts[f'{cl}46'] = f'=PERCENTILE.INC({rng},0.25)'
    ts[f'{cl}47'] = f'=PERCENTILE.INC({rng},0.75)'

# ── Step 3: Weighted mean H50:L50 using SUMPRODUCT ──
for cl in col_letters:
    ts[f'{cl}50'] = f'=SUMPRODUCT({cl}35:{cl}40,{cl}26:{cl}31)/SUM({cl}26:{cl}31)'

# ── Save ──
import os
os.makedirs('/root/output', exist_ok=True)
wb.save('/root/output/result.xlsx')
print('Saved /root/output/result.xlsx')
PYEOF
```

After saving, re-read the output file to confirm formulas are present:
```bash
python3 - <<'PY'
import openpyxl
wb = openpyxl.load_workbook('/root/output/result.xlsx')
ts = wb['Task']
# Spot-check formulas
for coord in ['H12','L17','H19','L24','H26','L31','H35','L40','H42','H43','H44','H45','H46','H47','H50','L50']:
    print(f'{coord}: {ts[coord].value}')
PY
```

## 2. Verify the row-label ordering
Before finalizing, verify the label ordering for rows 42-47 from the initial inspection output. If the ordering is different from MIN/MAX/MEDIAN/AVERAGE/25th/75th, adjust the formulas accordingly by re-reading the labels and re-writing only the affected cells. The key concern from previous failure is that rows 46-47 must contain valid percentile formulas.

Also verify:
- If the labels say "25th percentile" is in a different row than 46, move the formula.
- If `PERCENTILE.INC` causes issues, try `PERCENTILE` as a fallback (but `PERCENTILE.INC` is the standard modern Excel function and should work).

## 3. Run the verifier if available
```bash
cd /root && ls tests/ 2>/dev/null && python3 -m pytest tests/ -v 2>&1 | head -80
```
If any test fails, read the error carefully and fix only the failing cells, then re-save and re-run.

## Critical Notes
- Do NOT use `PERCENTILE.EXC` — use `PERCENTILE.INC` (or plain `PERCENTILE` if `.INC` fails).
- The previous failure was `#NAME?` errors in percentile rows. `PERCENTILE.INC` is standard Excel and should evaluate correctly. If the test environment's formula evaluator doesn't support `.INC`, fall back to `=PERCENTILE(range, 0.25)`.
- Do not add sheets, macros, VBA, external links, or helper tabs.
- Preserve all existing formatting.
- The final file must be at `/root/output/result.xlsx`.

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