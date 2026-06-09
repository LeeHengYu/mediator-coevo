# Task Instruction

Execute the following steps exactly:

## 1. Inspect the workbook structure

```bash
cd /root && python3 << 'PYEOF'
import openpyxl
wb = openpyxl.load_workbook('data/workbook.xlsx', data_only=True)
for s in wb.sheetnames:
    print(f'Sheet: {s}')
    ws = wb[s]
    print(f'  Dimensions: {ws.dimensions}')
    print(f'  Max row: {ws.max_row}, Max col: {ws.max_column}')
PYEOF
```

## 2. Inspect the Task sheet layout

```bash
python3 << 'PYEOF'
import openpyxl
wb = openpyxl.load_workbook('data/workbook.xlsx')
ws = wb['Task']

# Print rows 1-55 for columns A-M to understand the layout
for row in range(1, 56):
    vals = []
    for col in range(1, 14):  # A-M
        cell = ws.cell(row=row, column=col)
        v = cell.value
        vals.append(str(v) if v is not None else '')
    print(f'Row {row:2d}: {" | ".join(vals)}')
PYEOF
```

## 3. Inspect the Data sheet layout (especially rows 21-38)

```bash
python3 << 'PYEOF'
import openpyxl
wb = openpyxl.load_workbook('data/workbook.xlsx')
ws = wb['Data']

# Print header area and rows 1-40
for row in range(1, 41):
    vals = []
    for col in range(1, 20):  # A-S
        cell = ws.cell(row=row, column=col)
        v = cell.value
        vals.append(str(v) if v is not None else '')
    print(f'Row {row:2d}: {" | ".join(vals)}')
PYEOF
```

## 4. Build the formulas and save

After inspecting the layouts, write a Python script that:

### Step 1 – Lookup formulas in H12:L17, H19:L24, H26:L31
- Each block has 6 rows (ports) × 5 columns (years).
- Column D of each row has a series code; row 10 has the years.
- Use INDEX/MATCH to look up values from Data!$A$21:$S$38 (adjust range after inspection).
- Formula pattern per cell (e.g., H12): `=INDEX(Data!$B$21:$S$38,MATCH($D12,Data!$A$21:$A$38,0),MATCH(H$10,Data!$B$20:$S$20,0))` — adjust row/column references based on actual Data sheet layout discovered in step 3.
- IMPORTANT: Use the actual column letters and row numbers from the Data sheet. The MATCH for the year should reference the header row of the Data sheet's data area. The MATCH for the series code should reference column A (or whichever column holds the codes).

### Step 2 – Net container flow in H35:L40
- Formula: `=(H12-H19)/H26*100` (adjust row references based on actual block positions: first block = Loaded Containers Inbound, second = Loaded Containers Outbound, third = Terminal Throughput Capacity). Verify which block is which from the Task sheet labels.
- Statistics in H42:L47 (column-wise over H35:L40):
  - H42: `=MIN(H35:H40)`
  - H43: `=MAX(H35:H40)`
  - H44: `=MEDIAN(H35:H40)`
  - H45: `=AVERAGE(H35:H40)`
  - H46: `=PERCENTILE.INC(H35:H40,0.25)` — **CRITICAL**: Use `PERCENTILE.INC` (with the dot). If the previous run got #NAME? errors, the likely cause is openpyxl or the formula string. When writing with openpyxl, the formula must be written exactly as a string starting with `=`. Do NOT let openpyxl translate function names. Verify after writing that the cell.value starts with `=PERCENTILE.INC(`.
  - H47: `=PERCENTILE.INC(H35:H40,0.75)`

  **Alternative if PERCENTILE.INC causes issues**: Use `=PERCENTILE(H35:H40,0.25)` instead (the legacy function name that is universally recognized).

### Step 3 – Weighted mean in H50:L50
- `=SUMPRODUCT(H35:H40,H26:H31)/SUM(H26:H31)` for each column H-L.

### Saving
- Use openpyxl to load the workbook (NOT data_only), write formulas, and save to `/root/output/result.xlsx`.
- Do NOT use `data_only=True` when loading for writing.
- Create `/root/output/` directory if it doesn't exist.

## 5. Verify the output

```bash
python3 << 'PYEOF'
import openpyxl
wb = openpyxl.load_workbook('/root/output/result.xlsx')
ws = wb['Task']

# Check lookup formulas
for r in [12, 19, 26]:
    cell = ws.cell(row=r, column=8)  # H
    print(f'Row {r}, Col H: {cell.value}')

# Check net flow
print(f'H35: {ws.cell(row=35, column=8).value}')

# Check stats - especially percentiles
for r in range(42, 48):
    cell = ws.cell(row=r, column=8)
    print(f'Row {r}, Col H: {cell.value}')

# Check weighted mean
print(f'H50: {ws.cell(row=50, column=8).value}')
PYEOF
```

Confirm that:
- All formula cells contain formula strings (starting with `=`), not None or literal values.
- PERCENTILE formulas do NOT show as `#NAME?`. Use `PERCENTILE` (legacy) if `PERCENTILE.INC` caused issues in the previous run.
- No new sheets were added.
- The file is saved at `/root/output/result.xlsx`.

## Key lessons from previous failure
1. **PERCENTILE.INC might not be recognized** — try `PERCENTILE` (without `.INC`) as the function name. This is the legacy Excel function and is universally supported.
2. **Formulas returning None** — this happens when loading with `data_only=True` or when formulas aren't written correctly. Always load without `data_only=True` for writing.
3. **Cross-sheet references** — ensure `Data!` prefix is used and ranges match the actual data layout.
4. **Row/column mapping** — carefully verify which rows in the Task sheet correspond to which metric (Inbound, Outbound, Capacity) before writing the net flow formula.
5. **Verify the row labels** in column C or D of the Task sheet for rows 42-47 to know the exact order of statistics expected (min, max, median, mean, 25th pct, 75th pct).
6. **Use `$` for absolute references** appropriately in INDEX/MATCH formulas so they work when spanning multiple columns/rows.

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