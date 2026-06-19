# Task Instruction

Execute the following Python script in a single step to inspect the Data sheet, build all formulas, and save the result.

```python
import openpyxl, shutil, os

# --- Phase 0: Copy workbook and inspect Data sheet layout ---
os.makedirs('/root/output', exist_ok=True)
shutil.copy('/root/data/workbook.xlsx', '/root/output/result.xlsx')

wb = openpyxl.load_workbook('/root/output/result.xlsx')
task = wb['Task']
data = wb['Data']

# Print Task sheet structure for verification
print('=== Task sheet row 10 (years) ===')
for c in range(8, 13):  # H-L
    print(f'  col {c} ({task.cell(10,c).column_letter}): {task.cell(10,c).value}')

print('=== Task sheet column D (series codes) ===')
for r in range(12, 32):
    val = task.cell(r, 4).value
    if val is not None:
        print(f'  row {r}: {val}')

print('=== Task sheet row 34-47 labels ===')
for r in range(34, 48):
    print(f'  row {r}: col_B={task.cell(r,2).value}, col_D={task.cell(r,4).value}, col_E={task.cell(r,5).value}')

print('=== Task sheet row 49-51 ===')
for r in range(49, 52):
    print(f'  row {r}: col_B={task.cell(r,2).value}, col_D={task.cell(r,4).value}')

print('=== Data sheet rows 21-38 ===')
for r in range(20, 39):
    row_vals = []
    for c in range(1, min(data.max_column+1, 20)):
        row_vals.append(str(data.cell(r, c).value))
    print(f'  row {r}: {", ".join(row_vals)}')

print('=== Data sheet row 20 (header row above 21) ===')
for c in range(1, min(data.max_column+1, 20)):
    print(f'  col {c} ({data.cell(20,c).column_letter}): {data.cell(20,c).value}')

# Also check a few more header rows
for hr in [1, 2, 3, 19]:
    print(f'=== Data sheet row {hr} ===')
    for c in range(1, min(data.max_column+1, 20)):
        v = data.cell(hr, c).value
        if v is not None:
            print(f'  col {c}: {v}')

wb.close()
```

After inspecting the output, run the following script to populate all formulas and save:

```python
import openpyxl

wb = openpyxl.load_workbook('/root/output/result.xlsx')
task = wb['Task']

# -----------------------------------------------------------------
# Based on inspection, determine the Data sheet layout.
# Data rows 21:38 contain the source records.
# We need to find:
#   - Which column holds the series code (lookup key)
#   - Which row holds the year headers
#   - The range boundaries for INDEX/MATCH
# Adjust the references below after Phase 0 output.
# -----------------------------------------------------------------
# IMPORTANT: After seeing Phase 0 output, fill in the correct
# references. The template below uses INDEX/MATCH/MATCH pattern.
#
# Typical layout assumption (adjust after inspection):
#   Data!A21:A38 = series codes
#   Data!B20:XX20 = year headers (or row 21 is header)
#   Data!B21:XX38 = values
# -----------------------------------------------------------------

# We'll read the inspection output and adjust. For now, use a
# pattern that works with the standard Weighted-Risk-Assessment layout.

# After Phase 0, replace these placeholders:
DATA_LOOKUP_COL = 'A'       # column with series codes on Data sheet
DATA_YEAR_ROW = 20          # row with year headers on Data sheet  
DATA_FIRST_VAL_COL = 'B'    # first value column
DATA_LAST_VAL_COL = 'F'     # last value column (adjust after inspection)
DATA_FIRST_ROW = 21
DATA_LAST_ROW = 38

# Build the data range string for INDEX/MATCH/MATCH
# INDEX(Data!$B$21:$F$38, MATCH(D12,Data!$A$21:$A$38,0), MATCH(H$10,Data!$B$20:$F$20,0))

def make_lookup_formula(series_cell, year_cell,
                        val_range, code_range, year_range):
    return (f'=INDEX(Data!{val_range},'
            f'MATCH({series_cell},Data!{code_range},0),'
            f'MATCH({year_cell},Data!{year_range},0))')

wb.close()
print('Phase 1 template ready - need Phase 0 output first')
```

**CRITICAL: This is a two-phase task. First run Phase 0 (inspection), then read the output carefully and construct the final script.**

After Phase 0 inspection, construct and run the final complete script following these rules:

### Step 1: Lookup formulas in H12:L17, H19:L24, H26:L31
For each cell in these three blocks (rows 12-17, 19-24, 26-31; columns H-L):
- Use `INDEX(Data!<value_range>, MATCH($D<row>, Data!<code_column_range>, 0), MATCH(<col>$10, Data!<year_row_range>, 0))`
- The series code reference is `$D<row>` (column D of current row)
- The year reference is `<col>$10` (year from row 10 of current column)
- Use absolute references (`$`) for the Data ranges so they don't shift
- Assign the formula string to `task.cell(row=r, column=c).value`

### Step 2: Net reliability gap in H35:L40
Look at the Task sheet to identify which rows in the three blocks correspond to:
- Successful API Requests (likely rows 12:17)
- Failed API Requests (likely rows 19:24)  
- Compute Capacity (likely rows 26:31)

The formula for each cell: `=(<Successful_cell> - <Failed_cell>) / <Capacity_cell> * 100`

For row 35 col H: `=(H12-H19)/H26*100`, row 36: `=(H13-H20)/H27*100`, etc.

### Step 2 continued: Statistics in H42:L47
For each column H through L:
- Row 42: `=MIN(H35:H40)`
- Row 43: `=MAX(H35:H40)`  
- Row 44: `=MEDIAN(H35:H40)`
- Row 45: `=AVERAGE(H35:H40)`
- Row 46: `=PERCENTILE(H35:H40,0.25)`
- Row 47: `=PERCENTILE(H35:H40,0.75)`

**Use `PERCENTILE` not `PERCENTILE.INC` or `PERCENTILE.EXC`** — those cause #NAME? errors.

Check the actual row labels in rows 42-47 to confirm the order (min/max/median/mean/25th/75th) matches.

### Step 3: Weighted mean in H50:L50
For each column: `=SUMPRODUCT(H35:H40,H26:H31)/SUM(H26:H31)`

### Final save
- `wb.save('/root/output/result.xlsx')`
- Verify by reopening and printing a few cells to confirm formulas are present (not None)

### Verification
After saving, reopen the file and print cells H12, H35, H42, H50 to confirm they contain formula strings (starting with '=').

```python
wb2 = openpyxl.load_workbook('/root/output/result.xlsx')
t = wb2['Task']
for (r,c) in [(12,8),(35,8),(42,8),(50,8)]:
    print(f'  ({r},{c}): {t.cell(r,c).value}')
wb2.close()
```

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