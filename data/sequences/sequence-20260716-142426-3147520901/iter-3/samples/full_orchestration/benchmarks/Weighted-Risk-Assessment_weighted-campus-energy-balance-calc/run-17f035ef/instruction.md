# Task Instruction

Execute the following steps exactly, in order.

## 0. Inspect the workbook
```bash
cd /root
pip install openpyxl > /dev/null 2>&1
python3 - <<'PYEOF'
import openpyxl, json
wb = openpyxl.load_workbook('data/workbook.xlsx', data_only=False)
for s in wb.sheetnames:
    print(f'=== Sheet: {s} ===')
    ws = wb[s]
    print(f'  dims: {ws.dimensions}')
    # Print first 55 rows, cols A-M
    for r in range(1, min(ws.max_row+1, 60)):
        vals = []
        for c in range(1, 14):
            cell = ws.cell(r, c)
            v = cell.value
            vals.append(f'{openpyxl.utils.get_column_letter(c)}{r}={v}')
        print('  ', ' | '.join(vals))
PYEOF
```
Read the output carefully. Identify:
- The series codes in column D for rows 12-17, 19-24, 26-31.
- The years in row 10 for columns H-L.
- The structure of the Data sheet rows 21-38 (which column holds the series code, which row holds years, where the numeric data starts).
- The campus names in rows 35-40 and what rows 42-47 labels are (min, max, median, mean, 25th, 75th percentile).
- Row 50 label.

## 1. Write the formulas with openpyxl

Create and run a Python script `/root/solve.py` that:

```python
import openpyxl
import shutil, os

os.makedirs('/root/output', exist_ok=True)
shutil.copy('/root/data/workbook.xlsx', '/root/output/result.xlsx')

wb = openpyxl.load_workbook('/root/output/result.xlsx')
ws = wb['Task']
data_ws = wb['Data']

# First, inspect the Data sheet to find:
#   - Which column contains the series code (likely column A or B in Data rows 21-38)
#   - Which row contains the year headers
#   - The data range dimensions
# Print this information for verification.

# Based on inspection, construct formulas.
# The key pattern for Step 1 cells (e.g., H12) using INDEX-MATCH:
#   =INDEX(Data!$B$21:$Z$38, MATCH($D12, Data!$A$21:$A$38, 0), MATCH(H$10, Data!$B$20:$Z$20, 0))
# Adjust column letters and row numbers based on actual Data sheet layout.

# IMPORTANT: Use the actual layout discovered in Step 0.
# The formula must reference:
#   - $D{row} for the series code (absolute column, relative row)
#   - {col}$10 for the year (relative column, absolute row)
#   - The Data sheet range for the lookup

# Step 1: Populate H12:L17, H19:L24, H26:L31 with INDEX/MATCH formulas
# Step 2: H35:L40 = (Renewable_Gen - Grid_Consumption) / Baseline_Energy_Demand * 100
#   Map which row blocks correspond to Renewable Generation, Grid Consumption, Baseline Energy Demand
# Step 2 stats: H42:L47 = MIN, MAX, MEDIAN, AVERAGE, PERCENTILE(.., 0.25), PERCENTILE(.., 0.75)
#   over H35:H40 etc.
# Step 3: H50:L50 = SUMPRODUCT weighted mean

wb.save('/root/output/result.xlsx')
```

However, you MUST first complete Step 0 inspection and adapt the script to the actual layout. Here are the critical rules:

### Formula construction rules
- For Step 1 (lookup cells): Use `INDEX` with `MATCH`. The formula pattern should be:
  `=INDEX(Data!<data_range>, MATCH($D{row}, Data!<series_code_column_range>, 0), MATCH({col}$10, Data!<year_header_range>, 0))`
  Adjust ranges based on actual Data sheet structure discovered in Step 0.

- For Step 2 (Net renewable balance H35:L40): Identify which of the three row blocks (12-17, 19-24, 26-31) corresponds to Renewable Generation, Grid Consumption, and Baseline Energy Demand by reading the labels. Then for each campus row i (0..5):
  `=(H{12+i} - H{19+i}) / H{26+i} * 100` (adjust block assignments based on actual labels)

- For Step 2 stats (H42:L47), use these Excel functions on the range H35:H40 (column-wise):
  - Row 42: `=MIN(H$35:H$40)`
  - Row 43: `=MAX(H$35:H$40)`
  - Row 44: `=MEDIAN(H$35:H$40)`
  - Row 45: `=AVERAGE(H$35:H$40)`
  - Row 46: `=PERCENTILE(H$35:H$40,0.25)`
  - Row 47: `=PERCENTILE(H$35:H$40,0.75)`
  Use `PERCENTILE` not `PERCENTILE.INC` or `PERCENTILE.EXC` to avoid #NAME? errors.

- For Step 3 (H50:L50): `=SUMPRODUCT(H$35:H$40, H$26:H$31) / SUM(H$26:H$31)`
  (weighted mean = sum of value*weight / sum of weights)
  Adjust the weight range (Baseline Energy Demand block) based on actual layout.

### Critical checks
- Use `PERCENTILE` (not `PERCENTILE.INC`) to avoid #NAME? errors in older Excel compatibility.
- Every cell must get a string formula (starting with '='), not None or a Python-computed value.
- After saving, re-open the file and verify that cells H12, H35, H42, H50 contain formula strings (not None).
- Do NOT modify any existing formatting, do NOT add sheets.
- Save to `/root/output/result.xlsx`.

## 2. Verify the output
```bash
python3 - <<'PYEOF'
import openpyxl
wb = openpyxl.load_workbook('/root/output/result.xlsx', data_only=False)
ws = wb['Task']
spot_checks = ['H12','L17','H19','L24','H26','L31','H35','L40','H42','L47','H50','L50']
for ref in spot_checks:
    cell = ws[ref]
    print(f'{ref} = {cell.value}')
    assert cell.value is not None, f'{ref} is None!'
    assert str(cell.value).startswith('='), f'{ref} is not a formula: {cell.value}'
print('All spot checks passed.')
PYEOF
```

If any check fails, diagnose and fix before finishing.

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