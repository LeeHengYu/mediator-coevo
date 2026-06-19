# Task Instruction

Execute the following Python script to inspect the workbook structure, build the correct formulas, and save the result.

```python
import shutil, os, openpyxl
from openpyxl.utils import get_column_letter

# --- 0. Copy workbook -------------------------------------------------------
os.makedirs('/root/output', exist_ok=True)
shutil.copy('/root/data/workbook.xlsx', '/root/output/result.xlsx')

wb = openpyxl.load_workbook('/root/output/result.xlsx')

# --- 0a. Inspect sheets and key cells so we understand the layout -----------
print('Sheet names:', wb.sheetnames)
ws_task = wb['Task']
ws_data = wb['Data']

print('\n--- Task sheet: column D, rows 10-50 ---')
for r in range(10, 51):
    d_val = ws_task.cell(row=r, column=4).value  # col D
    h_val = ws_task.cell(row=r, column=8).value  # col H
    print(f'  Row {r}: D={d_val!r}  H={h_val!r}')

print('\n--- Task sheet: row 10 (header), cols H-L ---')
for c in range(8, 13):
    print(f'  {get_column_letter(c)}10 = {ws_task.cell(row=10, column=c).value!r}')

print('\n--- Data sheet: rows 19-40, first 20 cols ---')
for r in range(19, 41):
    vals = [ws_data.cell(row=r, column=c).value for c in range(1, 21)]
    print(f'  Row {r}: {vals}')

print('\n--- Data sheet: row 20 (possible header) ---')
for c in range(1, 21):
    print(f'  {get_column_letter(c)}20 = {ws_data.cell(row=r, column=c).value!r}')

wb.close()
```

After inspecting the output, run the following comprehensive script. **Adjust column/row references if the inspection reveals different layout details.** The script below assumes the common layout where:
- Task!D12:D17, D19:D24, D26:D31 contain series codes for the three blocks.
- Task!H10:L10 contain year values.
- Data sheet rows 21:38 contain the source data, with series codes in one column and year values across columns.

```python
import shutil, os, openpyxl
from openpyxl.utils import get_column_letter

os.makedirs('/root/output', exist_ok=True)
shutil.copy('/root/data/workbook.xlsx', '/root/output/result.xlsx')

wb = openpyxl.load_workbook('/root/output/result.xlsx')
ws = wb['Task']

# ============================================================================
# STEP 1 – Lookup formulas in H12:L17, H19:L24, H26:L31
# Pattern: INDEX(Data!$B$21:$T$38, MATCH(D12, Data!$A$21:$A$38, 0), MATCH(H$10, Data!$B$20:$T$20, 0))
# We will discover the exact Data layout from inspection output above and
# adjust the references accordingly. The template below uses a common layout.
# ============================================================================

# After inspecting the Data sheet, determine:
#   - Which column holds the series/lookup key (assume col A)
#   - Which row holds the year header (assume row 20)
#   - Data values span from B21:?38
# Read Data sheet to find actual extents
ws_data = wb['Data']

# Find the last used column in Data row 20 (year header row)
data_last_col = 1
for c in range(1, 100):
    if ws_data.cell(row=20, column=c).value is not None:
        data_last_col = c
print(f'Data last col with header in row 20: {data_last_col} ({get_column_letter(data_last_col)})')

# Also check if the key column might be col A or col B
print('Data A21:', ws_data.cell(row=21, column=1).value)
print('Data B21:', ws_data.cell(row=21, column=2).value)

# We'll build references dynamically. Let's figure out the key column and
# value range from inspection. For now, use the most common pattern:
#   Key column = A, years header row = 20 starting col B,
#   data block = A21:?38 (keys) and B21:?38 (values)

# Determine key_col_letter, val_start_col_letter, val_end_col_letter, header_row
# by scanning Data row 20
header_row_data = 20
key_col_data = None
val_start_col_data = None
val_end_col_data = None

for c in range(1, 100):
    v = ws_data.cell(row=header_row_data, column=c).value
    if v is None and val_start_col_data is not None and val_end_col_data is not None:
        break
    if v is not None:
        # Check if it looks like a year (number >= 1900)
        try:
            if int(v) >= 1900:
                if val_start_col_data is None:
                    val_start_col_data = c
                val_end_col_data = c
                continue
        except (ValueError, TypeError):
            pass
        # Otherwise it might be a label column; the last label col before years is the key col
        key_col_data = c

if key_col_data is None:
    # Fallback: key is col A
    key_col_data = 1
if val_start_col_data is None:
    val_start_col_data = key_col_data + 1
    val_end_col_data = data_last_col

kcl = get_column_letter(key_col_data)
vscl = get_column_letter(val_start_col_data)
vecl = get_column_letter(val_end_col_data)

print(f'Key col: {kcl}, Value cols: {vscl}-{vecl}, Header row: {header_row_data}')

# Build the INDEX/MATCH formula template
# INDEX(Data!$<vscl>$21:$<vecl>$38, MATCH(<task_key_cell>, Data!$<kcl>$21:$<kcl>$38, 0), MATCH(<task_year_cell>, Data!$<vscl>$<header_row>:$<vecl>$<header_row>, 0))

def make_lookup_formula(task_key_cell, task_year_cell):
    """Return an INDEX/MATCH formula string."""
    return (
        f'=INDEX(Data!${vscl}$21:${vecl}$38,'
        f'MATCH({task_key_cell},Data!${kcl}$21:${kcl}$38,0),'
        f'MATCH({task_year_cell},Data!${vscl}${header_row_data}:${vecl}${header_row_data},0))'
    )

# Fill Step 1 blocks
for block_start in [12, 19, 26]:
    for r in range(block_start, block_start + 6):
        key_cell = f'$D{r}'  # series code in col D, absolute row
        for c in range(8, 13):  # H=8 .. L=12
            year_cell = f'{get_column_letter(c)}$10'  # year in row 10, absolute row
            formula = make_lookup_formula(key_cell, year_cell)
            ws.cell(row=r, column=c).value = formula
            # Debug first formula
            if r == block_start and c == 8:
                print(f'Sample formula at {get_column_letter(c)}{r}: {formula}')

# ============================================================================
# STEP 2 – Net patient flow in H35:L40
# (Patient Admissions - Patient Discharges) / Effective Bed Capacity * 100
# Admissions = rows 12:17, Discharges = rows 19:24, Capacity = rows 26:31
# ============================================================================
for i in range(6):  # 6 hospitals
    adm_row = 12 + i
    dis_row = 19 + i
    cap_row = 26 + i
    out_row = 35 + i
    for c in range(8, 13):
        col_letter = get_column_letter(c)
        formula = f'=({col_letter}{adm_row}-{col_letter}{dis_row})/{col_letter}{cap_row}*100'
        ws.cell(row=out_row, column=c).value = formula

# ============================================================================
# STEP 2 continued – Statistics in H42:L47
# Row 42: MIN, 43: MAX, 44: MEDIAN, 45: AVERAGE, 46: 25th pctl, 47: 75th pctl
# ============================================================================
# Read labels in column D/E/F for rows 42-47 to confirm order
for r in range(42, 48):
    label = ws.cell(row=r, column=4).value
    if label is None:
        label = ws.cell(row=r, column=5).value
    if label is None:
        label = ws.cell(row=r, column=6).value
    if label is None:
        label = ws.cell(row=r, column=7).value
    print(f'  Row {r} label: {label!r}')

# We'll assign based on typical order: min, max, median, mean, 25th, 75th
# But let's detect from labels
stat_formulas = {}
for r in range(42, 48):
    # Scan cols A-G for a label
    label = ''
    for cc in range(1, 8):
        v = ws.cell(row=r, column=cc).value
        if v is not None:
            label = str(v).lower()
            break
    stat_formulas[r] = label

print('Detected stat labels:', stat_formulas)

for r in range(42, 48):
    label = stat_formulas[r]
    for c in range(8, 13):
        cl = get_column_letter(c)
        rng = f'{cl}35:{cl}40'
        if 'min' in label:
            f = f'=MIN({rng})'
        elif 'max' in label:
            f = f'=MAX({rng})'
        elif 'median' in label:
            f = f'=MEDIAN({rng})'
        elif 'mean' in label or 'average' in label:
            f = f'=AVERAGE({rng})'
        elif '25' in label or 'q1' in label or 'first' in label:
            f = f'=PERCENTILE({rng},0.25)'
        elif '75' in label or 'q3' in label or 'third' in label:
            f = f'=PERCENTILE({rng},0.75)'
        else:
            # Fallback order: min, max, median, average, 25th, 75th
            idx = r - 42
            templates = [
                f'=MIN({rng})',
                f'=MAX({rng})',
                f'=MEDIAN({rng})',
                f'=AVERAGE({rng})',
                f'=PERCENTILE({rng},0.25)',
                f'=PERCENTILE({rng},0.75)',
            ]
            f = templates[idx]
        ws.cell(row=r, column=c).value = f
        if c == 8:
            print(f'  {cl}{r}: {f}')

# ============================================================================
# STEP 3 – Weighted mean in H50:L50
# SUMPRODUCT(net_flow_col, capacity_col) / SUM(capacity_col)
# ============================================================================
for c in range(8, 13):
    cl = get_column_letter(c)
    net_rng = f'{cl}35:{cl}40'
    cap_rng = f'{cl}26:{cl}31'
    formula = f'=SUMPRODUCT({net_rng},{cap_rng})/SUM({cap_rng})'
    ws.cell(row=50, column=c).value = formula
    print(f'  H50 weighted mean {cl}50: {formula}')

# ============================================================================
# Save
# ============================================================================
wb.save('/root/output/result.xlsx')
wb.close()
print('\nDone – saved to /root/output/result.xlsx')

# Quick verification: reload and check a few cells
wb2 = openpyxl.load_workbook('/root/output/result.xlsx')
ws2 = wb2['Task']
for cell_ref in ['H12', 'L17', 'H19', 'H26', 'H35', 'H42', 'H47', 'H50', 'L50']:
    print(f'  {cell_ref} = {ws2[cell_ref].value!r}')
wb2.close()
```

IMPORTANT execution notes:
1. Run the FIRST (inspection) script first. Read its output carefully.
2. Before running the second script, verify these assumptions against the inspection output and adjust if needed:
   - The Data sheet's key column (series codes) – is it column A or another column?
   - The Data sheet's year header row – is it row 20?
   - The Data sheet's value range – rows 21:38, starting from which column?
   - The Task sheet's row labels for statistics (rows 42-47) – what are they?
3. If the inspection reveals different layout, adjust the column letters, row numbers, and range references in the second script accordingly.
4. After saving, verify the output by reloading the workbook and confirming that formula strings (not None) appear in the target cells.
5. Do NOT use `data_only=True` when loading – we want to preserve formulas.
6. The PERCENTILE function (not PERCENTILE.INC or PERCENTILE.EXC) is safest for openpyxl compatibility. If labels say "percentile" use PERCENTILE. Avoid _xlfn. prefixed names as they may cause #NAME? errors in some contexts.

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
Task metadata: author_email=catpaw@meituan.com, author_name=CatPaw Task Engineer, category=spreadsheet-formula-reuse, difficulty=easy, tags=[excel, formulas, lookup, statistics, weighted-mean].
Verifier config: timeout_sec=600.0.