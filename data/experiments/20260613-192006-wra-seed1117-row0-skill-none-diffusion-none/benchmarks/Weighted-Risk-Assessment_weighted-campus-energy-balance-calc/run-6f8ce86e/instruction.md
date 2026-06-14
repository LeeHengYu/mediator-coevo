# Task Instruction

Execute the following Python script in a single step to produce `/root/output/result.xlsx`.

```python
import openpyxl, os, shutil

# --- Phase 0: Copy workbook and inspect Data sheet layout ---
os.makedirs('/root/output', exist_ok=True)
shutil.copy('/root/data/workbook.xlsx', '/root/output/result.xlsx')

wb = openpyxl.load_workbook('/root/output/result.xlsx')

# Inspect Data sheet to find the data range in rows 21:38
data_ws = wb['Data']
print('=== Data sheet inspection ===')
for r in range(20, 40):
    row_vals = []
    for c in range(1, 20):
        v = data_ws.cell(row=r, column=c).value
        row_vals.append(str(v)[:30] if v is not None else '')
    print(f'Row {r}: {row_vals}')

# Inspect Task sheet structure
task_ws = wb['Task']
print('\n=== Task sheet inspection ===')
for r in range(1, 55):
    row_vals = []
    for c in range(1, 15):
        v = task_ws.cell(row=r, column=c).value
        row_vals.append(str(v)[:40] if v is not None else '')
    print(f'Row {r}: {row_vals}')

wb.close()
```

After inspecting the output, execute the following script (adjust Data sheet references if the inspection reveals different coordinates):

```python
import openpyxl, os
from openpyxl.utils import get_column_letter

wb = openpyxl.load_workbook('/root/output/result.xlsx')
task = wb['Task']

# --- Determine key coordinates from inspection ---
# Years are in row 10, columns H(8) through L(12)
# Series codes are in column D for each block
# Data sheet rows 21:38 contain the lookup table

# We need to know the Data sheet layout:
# - Which column has the series code (for MATCH on series)
# - Which row has years (for MATCH on year)
# From typical layout: Data!A21:A38 has series codes, Data row 20 (or 21) has years
# We'll use INDEX/MATCH/MATCH pattern

# After inspection, set these (adjust if needed):
DATA_ROW_START = 21   # first data row on Data sheet
DATA_ROW_END = 38     # last data row on Data sheet
# The series code column on Data sheet (typically column A or B)
# The year header row on Data sheet (typically one row above data start, i.e., row 20)
# We'll determine from Phase 0 output.

# For now, build formulas using INDEX/MATCH/MATCH:
# =INDEX(Data!$B$21:$F$38, MATCH(D12, Data!$A$21:$A$38, 0), MATCH(H$10, Data!$B$20:$F$20, 0))
# The exact columns depend on inspection. Placeholder pattern:

# IMPORTANT: This script will be refined after Phase 0 inspection.
# The formula pattern is:
# =INDEX(<data_values_range>, MATCH($D<row>, <series_code_range>, 0), MATCH(<year_col>$10, <year_header_range>, 0))

print('Phase 0 inspection needed first - see output above')
wb.close()
```

**After Phase 0 inspection, execute the final formula-injection script:**

Based on the inspection output, construct and run a Python script that does the following:

1. **Open** `/root/output/result.xlsx` with openpyxl.

2. **Step 1 – Lookup formulas in H12:L17, H19:L24, H26:L31:**
   - For each cell at `(row, col)` in these three 6×5 blocks:
     - The series code is in column D of the same row on sheet `Task`.
     - The year is in row 10 of the same column on sheet `Task`.
     - Write an `INDEX`/`MATCH` formula referencing the Data sheet. Use the exact ranges found during inspection. Example pattern:
       ```
       =INDEX(Data!$<val_start>$<data_row_start>:$<val_end>$<data_row_end>, MATCH($D{row}, Data!$<series_col>$<data_row_start>:$<series_col>$<data_row_end>, 0), MATCH({col_letter}$10, Data!$<val_start>$<year_header_row>:$<val_end>$<year_header_row>, 0))
       ```
     - Use absolute references (`$`) for the Data ranges and the series code column; use `$D{row}` for the series code and `{col_letter}$10` for the year so formulas can be understood per-cell.

3. **Step 2 – Net renewable balance in H35:L40:**
   - The six campuses correspond to rows 35-40. The three input blocks are:
     - Renewable Generation: rows 12-17 (block 1)
     - Grid Consumption: rows 19-24 (block 2)  
     - Baseline Energy Demand: rows 26-31 (block 3)
   - For cell at `(row, col)` where row offset `i` = row - 35:
     - Renewable Gen row = 12 + i
     - Grid Consumption row = 19 + i
     - Baseline Energy Demand row = 26 + i
     - Formula: `=({col_letter}{ren_row} - {col_letter}{grid_row}) / {col_letter}{base_row} * 100`

4. **Step 2 – Summary statistics in H42:L47:**
   - Row 42: `=MIN({col_letter}35:{col_letter}40)`
   - Row 43: `=MAX({col_letter}35:{col_letter}40)`
   - Row 44: `=MEDIAN({col_letter}35:{col_letter}40)`
   - Row 45: `=AVERAGE({col_letter}35:{col_letter}40)`
   - Row 46: `=PERCENTILE({col_letter}35:{col_letter}40,0.25)`
   - Row 47: `=PERCENTILE({col_letter}35:{col_letter}40,0.75)`
   - **Use `PERCENTILE` not `PERCENTILE.INC` or `PERCENTILE.EXC`** to avoid #NAME? errors.

5. **Step 3 – Weighted mean in H50:L50:**
   - For each column (H through L):
     - `=SUMPRODUCT({col_letter}35:{col_letter}40, {col_letter}26:{col_letter}31) / SUM({col_letter}26:{col_letter}31)`

6. **Save** the workbook to `/root/output/result.xlsx`.

7. **Verify** by reopening the file and printing cells H12, H35, H42, H50 to confirm they contain formula strings (not None).

**Critical implementation notes:**
- Assign formula strings directly: `task.cell(row=r, column=c).value = formula_string`
- Ensure every formula string starts with `=`
- Call `wb.save('/root/output/result.xlsx')` before closing
- Do NOT use `data_only=True` when loading
- Match the exact row/column references from the Data sheet inspection in Phase 0
- The summary stats row assignments (42-47) must match the labels in the Task sheet; verify label order during inspection and adjust if needed (e.g., if min is row 42 vs row 43)

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