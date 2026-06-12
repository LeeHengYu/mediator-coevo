# Task Instruction

## Task: Populate formulas in /root/data/workbook.xlsx and save to /root/output/result.xlsx

This task requires writing Excel formulas into specific cells of an existing workbook. Follow these phases exactly.

### Phase 0: Setup
```bash
mkdir -p /root/output
```

### Phase 1: Inspect the workbook structure
Before writing ANY formulas, inspect the workbook thoroughly using Python + openpyxl.

```python
import openpyxl
wb = openpyxl.load_workbook('/root/data/workbook.xlsx')

# 1. List all sheet names
print('Sheets:', wb.sheetnames)

# 2. Inspect 'Task' sheet structure
ts = wb['Task']
print('\n--- Task sheet ---')
# Print rows 1-55, columns A-M to understand layout
for row in ts.iter_rows(min_row=1, max_row=55, min_col=1, max_col=13, values_only=False):
    vals = [(c.coordinate, c.value) for c in row if c.value is not None]
    if vals:
        print(vals)

# 3. Inspect 'Data' sheet structure
ds = wb['Data']
print('\n--- Data sheet rows 1-5 (headers) ---')
for row in ds.iter_rows(min_row=1, max_row=5, min_col=1, max_col=20, values_only=False):
    vals = [(c.coordinate, c.value) for c in row if c.value is not None]
    if vals:
        print(vals)

print('\n--- Data sheet rows 18-40 ---')
for row in ds.iter_rows(min_row=18, max_row=40, min_col=1, max_col=20, values_only=False):
    vals = [(c.coordinate, c.value) for c in row if c.value is not None]
    if vals:
        print(vals)

# Also check what's in row 10 of Task (years) and column D (series codes)
print('\n--- Task row 10 (years) ---')
for c in ts[10]:
    if c.value is not None:
        print(c.coordinate, c.value)

print('\n--- Task column D rows 12-31 ---')
for row in range(12, 32):
    c = ts.cell(row=row, column=4)
    if c.value is not None:
        print(c.coordinate, c.value)

print('\n--- Task rows 33-50 labels ---')
for row in range(33, 51):
    vals = [(ts.cell(row=row, column=col).coordinate, ts.cell(row=row, column=col).value) for col in range(1, 8) if ts.cell(row=row, column=col).value is not None]
    if vals:
        print(vals)

wb.close()
```

Run this and **read the output carefully**. You need to identify:
- The exact column in the Data sheet that contains series codes (likely column A or B)
- The exact row in the Data sheet that contains year headers
- The data matrix range on the Data sheet (rows 21:38)
- The year values in Task!H10:L10 and their format
- The series codes in Task!D12:D17, D19:D24, D26:D31
- The labels/structure of rows 35-50 on the Task sheet

### Phase 2: Write formulas
Based on your Phase 1 inspection, write a Python script using openpyxl to populate formulas. Use `data_only=False` when loading (which is the default).

Key formula patterns (adapt column letters and row numbers based on Phase 1 findings):

**Step 1 - Lookup formulas in H12:L17, H19:L24, H26:L31:**
Use INDEX/MATCH pattern. For each cell, e.g., H12:
```
=INDEX(Data!<data_range>, MATCH($D12, Data!<series_code_column>, 0), MATCH(H$10, Data!<year_header_row>, 0))
```
- Lock column D with `$D12` so it stays fixed when copying across columns
- Lock row 10 with `H$10` so it stays fixed when copying down rows
- The data range, series code column, and year header row must match what you found in Phase 1

**Step 2 - Net reliability gap in H35:L40:**
Formula: `(Successful API Requests - Failed API Requests) / Compute Capacity * 100`
- Identify which of the three blocks (H12:L17, H19:L24, H26:L31) corresponds to each metric
- For example, if H12:L17 = Successful API Requests, H19:L24 = Failed API Requests, H26:L31 = Compute Capacity, then H35 = `=(H12-H19)/H26*100`
- Adjust row offsets so each of the 6 regions maps correctly

**Step 2 - Summary statistics in H42:L47 (column-wise over H35:L40):**
- MIN: `=MIN(H35:H40)`
- MAX: `=MAX(H35:H40)`
- MEDIAN: `=MEDIAN(H35:H40)`
- AVERAGE: `=AVERAGE(H35:H40)`
- 25th percentile: `=PERCENTILE(H35:H40,0.25)`
- 75th percentile: `=PERCENTILE(H35:H40,0.75)`
Match the order to the labels you found in Phase 1 (rows 42-47).

**Step 3 - Weighted mean in H50:L50:**
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```
This uses the Net reliability gap values as the values and Compute Capacity as weights.

### Phase 3: Save and validate
1. Save to `/root/output/result.xlsx`
2. Reload the saved file and verify:
   - Cells H12:L17, H19:L24, H26:L31 contain formula strings (not None, not plain values)
   - Cells H35:L40 contain formula strings
   - Cells H42:L47 contain formula strings
   - Cells H50:L50 contain formula strings
   - Print all formula strings to confirm correctness
3. Verify no new sheets were added
4. Verify formatting is preserved (spot-check a few cells for fill colors)

### Critical Reminders
- Do NOT use `data_only=True` when loading - this would strip formulas
- Do NOT add sheets, macros, VBA, or helper tabs
- Do NOT modify existing formatting
- If any formula references look wrong after Phase 1 inspection, adjust before writing
- The Data sheet row range for lookup is rows 21:38 as stated in the instructions
- Double-check that year header row and series code column on Data sheet are correctly identified

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