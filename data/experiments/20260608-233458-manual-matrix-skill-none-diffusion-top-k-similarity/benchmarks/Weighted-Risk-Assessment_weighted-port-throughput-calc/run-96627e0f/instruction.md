# Task Instruction

Execute the following steps exactly, in order.

## 0 – Inspect the workbook
```bash
cd /root/data
python3 - <<'PY'
import openpyxl, json
wb = openpyxl.load_workbook('workbook.xlsx', data_only=False)
for name in wb.sheetnames:
    print(f'--- Sheet: {name} ---')
    ws = wb[name]
    print(f'  Dimensions: {ws.dimensions}')
    # Print first 50 rows, columns A-M
    for row in ws.iter_rows(min_row=1, max_row=50, min_col=1, max_col=13, values_only=False):
        vals = [(c.coordinate, c.value) for c in row if c.value is not None]
        if vals:
            print(' ', vals)
PY
```
Read the output carefully. Identify:
- The series codes in column D for rows 12-17, 19-24, 26-31.
- The years in row 10 for columns H-L.
- The structure of the Data sheet rows 21-38 (which column holds the series code, which row holds years, and where the numeric data lives).
- The port names in rows 35-40 and their correspondence to the three blocks above.
- Any existing formulas or values already present.

## 1 – Write the Python script that populates the workbook

Create `/root/solve.py` with the logic below. Adjust column letters, row numbers, and Data-sheet layout based on what you discovered in Step 0.

```python
import openpyxl
import shutil, os

src = '/root/data/workbook.xlsx'
dst = '/root/output/result.xlsx'
os.makedirs('/root/output', exist_ok=True)
shutil.copy2(src, dst)

wb = openpyxl.load_workbook(dst)
ws = wb['Task']

# ---------- Step 0: discover layout ----------
# (Fill in after inspection.  Example placeholders below.)

# Columns H-L = columns 8-12
col_letters = ['H','I','J','K','L']

# ---- Step 1: Lookup formulas in H12:L17, H19:L24, H26:L31 ----
# Use INDEX/MATCH referencing Data!$A$21:$A$38 for series codes
# and Data!<first_data_col>$20:<last_data_col>$20 for year headers.
# Adapt the exact Data-sheet column range after inspection.

blocks = [
    range(12, 18),   # rows 12-17
    range(19, 25),   # rows 19-24
    range(26, 32),   # rows 26-31
]

# We need to know:
#   - Data sheet: which column has the series code (e.g. A)
#   - Data sheet: which row has the year headers (e.g. row 20)
#   - Data sheet: the rectangular data range
# Build INDEX(MATCH,MATCH) formulas accordingly.

for block in blocks:
    for r in block:
        for col_letter in col_letters:
            # series_code_cell = D{r} on Task sheet
            # year_cell = {col_letter}10 on Task sheet
            # Formula pattern (adjust after inspection):
            formula = (
                f"=INDEX(Data!$B$21:$XX$38,"
                f"MATCH($D{r},Data!$A$21:$A$38,0),"
                f"MATCH({col_letter}$10,Data!$B$20:$XX$20,0))"
            )
            ws[f'{col_letter}{r}'] = formula

# ---- Step 2a: Net container flow in H35:L40 ----
# Formula: (Loaded Inbound - Loaded Outbound) / Terminal Throughput Capacity * 100
# Rows 12-17 = one block, 19-24 = another, 26-31 = third.
# Identify which block is Inbound, Outbound, Capacity from the labels.
# Then for each port i (0..5):
#   H35+i = (H12+i - H19+i) / H26+i * 100   (adjust block mapping after inspection)

for i in range(6):
    r_out = 35 + i
    for col_letter in col_letters:
        # Adjust row offsets after inspection
        inbound_row  = 12 + i  # placeholder
        outbound_row = 19 + i  # placeholder
        capacity_row = 26 + i  # placeholder
        formula = (
            f"=({col_letter}{inbound_row}-{col_letter}{outbound_row})"
            f"/{col_letter}{capacity_row}*100"
        )
        ws[f'{col_letter}{r_out}'] = formula

# ---- Step 2b: Statistics in H42:L47 ----
# Row 42 = MIN, 43 = MAX, 44 = MEDIAN, 45 = AVERAGE, 46 = 25th %ile, 47 = 75th %ile
stat_formulas = [
    (42, 'MIN({c}35:{c}40)'),
    (43, 'MAX({c}35:{c}40)'),
    (44, 'MEDIAN({c}35:{c}40)'),
    (45, 'AVERAGE({c}35:{c}40)'),
    (46, 'PERCENTILE.INC({c}35:{c}40,0.25)'),
    (47, 'PERCENTILE.INC({c}35:{c}40,0.75)'),
]
for row, tmpl in stat_formulas:
    for col_letter in col_letters:
        ws[f'{col_letter}{row}'] = '=' + tmpl.format(c=col_letter)

# ---- Step 3: Weighted mean in H50:L50 ----
for col_letter in col_letters:
    ws[f'{col_letter}50'] = (
        f'=SUMPRODUCT({col_letter}35:{col_letter}40,'
        f'{col_letter}26:{col_letter}31)/SUM({col_letter}26:{col_letter}31)'
    )

wb.save(dst)
print('Saved', dst)
```

## 2 – Run the inspection, then adapt and run the script

1. Run the inspection code from Step 0. Read the output carefully.
2. Based on the actual layout you see:
   - Determine the exact Data-sheet column that holds series codes (likely column A).
   - Determine the exact Data-sheet row that holds year headers.
   - Determine the exact Data-sheet data range (columns and rows).
   - Determine which of the three Task-sheet blocks (rows 12-17, 19-24, 26-31) corresponds to Loaded Containers Inbound, Loaded Containers Outbound, and Terminal Throughput Capacity. Check the labels in column A or B or C of those row groups.
   - Map the six ports in rows 35-40 to the correct rows in the three blocks.
3. Edit `/root/solve.py` to use the correct references discovered above. Key things to get right:
   - The INDEX/MATCH formulas must reference the correct Data-sheet ranges. Use absolute references with $ signs for the lookup arrays. The series code lookup should be in one column, the year lookup in one row.
   - The Net container flow formula must use the correct block rows for inbound, outbound, and capacity.
   - Use `PERCENTILE.INC` (not `PERCENTILE`) for rows 46 and 47 to avoid #NAME? errors.
4. Run `python3 /root/solve.py`.

## 3 – Validate the output
```bash
python3 - <<'PY'
import openpyxl
wb = openpyxl.load_workbook('/root/output/result.xlsx', data_only=False)
ws = wb['Task']
print('=== Spot-check formulas ===')
for cell in ['H12','L17','H19','L24','H26','L31','H35','L40','H42','H46','H47','H50','L50']:
    print(f'  {cell}: {ws[cell].value}')
# Verify no None in required ranges
for rng_label, rows, cols in [
    ('Lookup1', range(12,18), range(8,13)),
    ('Lookup2', range(19,25), range(8,13)),
    ('Lookup3', range(26,32), range(8,13)),
    ('NetFlow', range(35,41), range(8,13)),
    ('Stats',   range(42,48), range(8,13)),
    ('Weighted', [50],        range(8,13)),
]:
    for r in rows:
        for c in cols:
            v = ws.cell(row=r, column=c).value
            if v is None:
                print(f'  WARNING: {rng_label} cell row={r} col={c} is None!')
print('Validation complete.')
PY
```

If any cell is None or a formula looks wrong, fix and re-run before finishing.

## Critical reminders
- Do NOT use `PERCENTILE(...)`. Always use `PERCENTILE.INC(...)` for the 25th and 75th percentile rows.
- Do NOT add sheets, macros, VBA, external links, or helper tabs.
- Do NOT change existing formatting.
- The final file MUST be at `/root/output/result.xlsx`.
- Every yellow cell must contain a **formula**, not a hardcoded value.
- Read the actual Data sheet structure before writing any formulas — do not guess column ranges.

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