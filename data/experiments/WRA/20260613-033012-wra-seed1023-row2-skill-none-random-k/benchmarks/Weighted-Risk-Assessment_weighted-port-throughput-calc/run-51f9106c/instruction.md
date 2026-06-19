# Task Instruction

Execute the following steps exactly, in order.

## 0. Inspect the workbook
```bash
cd /root && python3 - <<'PYEOF'
import openpyxl, json
wb = openpyxl.load_workbook('data/workbook.xlsx', data_only=True)
for s in wb.sheetnames:
    print(f'=== Sheet: {s} ===')
    ws = wb[s]
    print(f'  Dimensions: {ws.dimensions}')
    # Print first 50 rows, columns A-M
    for row in ws.iter_rows(min_row=1, max_row=50, max_col=13, values_only=False):
        vals = [(c.coordinate, c.value) for c in row if c.value is not None]
        if vals:
            print(f'  {vals}')
PYEOF
```
Read the output carefully. Identify:
- The series codes in column D for rows 12-17, 19-24, 26-31.
- The years in row 10 (columns H-L).
- The structure of the Data sheet rows 21-38 (which column holds the series code, which row holds years, how data is laid out).
- The port names in rows 35-40 and what blocks correspond to Loaded Containers Inbound, Loaded Containers Outbound, Terminal Throughput Capacity.
- The weights row for CPA in row 50.

## 1. Also inspect the Data sheet more thoroughly
```bash
python3 - <<'PYEOF'
import openpyxl
wb = openpyxl.load_workbook('data/workbook.xlsx', data_only=True)
ws = wb['Data']
for row in ws.iter_rows(min_row=1, max_row=45, max_col=20, values_only=False):
    vals = [(c.coordinate, c.value) for c in row if c.value is not None]
    if vals:
        print(vals)
PYEOF
```

## 2. Write the solution script

After inspecting, write a Python script `/root/solve.py` that:

a) Loads `/root/data/workbook.xlsx` with openpyxl (NOT data_only).

b) For Step 1 – Populates H12:L17, H19:L24, H26:L31 on sheet `Task` with INDEX/MATCH formulas.
   - Each formula uses two inputs: the series code from column D of the current row, and the year from row 10 of the current column.
   - The lookup source is sheet `Data` rows 21:38.
   - Use the pattern: `=INDEX(Data!$B$21:$<lastcol>$38, MATCH($D<row>,Data!$A$21:$A$38,0), MATCH(<colref>$10,Data!$B$20:$<lastcol>$20,0))`
   - Adjust column references based on what you found in the Data sheet inspection. The key is that column A of Data holds series codes and row 20 (or whichever header row) holds years.
   - Use absolute references for the data range and relative references for the lookup values (series code cell and year cell) so the formula works across the block.

c) For Step 2 – Net container flow in H35:L40:
   - The three blocks are: Loaded Containers Inbound (rows 12-17), Loaded Containers Outbound (rows 19-24), Terminal Throughput Capacity (rows 26-31).
   - Formula for each cell: `=(H12-H19)/H26*100` (adjusting row/column references for each port and year).
   - For H42:L42 (MIN): `=MIN(H35:H40)` across each column.
   - For H43:L43 (MAX): `=MAX(H35:H40)` across each column.
   - For H44:L44 (MEDIAN): `=MEDIAN(H35:H40)` across each column.
   - For H45:L45 (AVERAGE): `=AVERAGE(H35:H40)` across each column.
   - For H46:L46 (25th percentile): `=PERCENTILE(H35:H40,0.25)` — use exactly `PERCENTILE` (the legacy function name, NOT `PERCENTILE.INC` or `PERCENTILE.EXC`).
   - For H47:L47 (75th percentile): `=PERCENTILE(H35:H40,0.75)` — again use exactly `PERCENTILE`.

d) For Step 3 – Weighted mean in H50:L50:
   - `=SUMPRODUCT(H35:H40,H26:H31)/SUM(H26:H31)` for each column H through L.

e) Save to `/root/output/result.xlsx`. Create the output directory if needed.

IMPORTANT: Do NOT modify any existing cell values, formatting, or sheet structure. Only write formulas into the specified empty/yellow cells.

## 3. Run the script
```bash
mkdir -p /root/output
python3 /root/solve.py
```

## 4. Validate the output
```bash
python3 - <<'PYEOF'
import openpyxl
wb = openpyxl.load_workbook('/root/output/result.xlsx')
ws = wb['Task']
# Check that formula cells are populated
for r in range(12, 18):
    for c in ['H','I','J','K','L']:
        cell = ws[f'{c}{r}']
        print(f'{c}{r}: {cell.value}')
for r in [35,36,37,38,39,40]:
    for c in ['H','I','J','K','L']:
        cell = ws[f'{c}{r}']
        print(f'{c}{r}: {cell.value}')
for r in [42,43,44,45,46,47]:
    for c in ['H','I','J','K','L']:
        cell = ws[f'{c}{r}']
        print(f'{c}{r}: {cell.value}')
for c in ['H','I','J','K','L']:
    cell = ws[f'{c}50']
    print(f'{c}50: {cell.value}')
print('--- Checking PERCENTILE function name ---')
for c in ['H']:
    print(f'{c}46: {ws[f"{c}46"].value}')
    print(f'{c}47: {ws[f"{c}47"].value}')
PYEOF
```
Confirm:
- All formula cells contain formulas (strings starting with '=').
- PERCENTILE cells use `PERCENTILE(` not `PERCENTILE.INC(` or `PERCENTILE.EXC(`.
- No cells are None.

## 5. Run the verifier if available
```bash
if [ -f /root/test_output.py ]; then cd /root && python3 -m pytest test_output.py -v; fi
```

## Key warnings from prior failures:
- Do NOT use `PERCENTILE.INC` or `PERCENTILE.EXC` — these cause #NAME? errors. Use the legacy `PERCENTILE` function.
- Do NOT leave any target cells empty/None — all yellow cells must have formulas.
- Inspect the Data sheet structure carefully before writing formulas — get the exact row/column layout right.

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