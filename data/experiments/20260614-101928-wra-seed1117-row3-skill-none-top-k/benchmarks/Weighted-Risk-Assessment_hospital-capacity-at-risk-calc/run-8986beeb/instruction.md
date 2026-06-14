# Task Instruction

Execute the following steps to produce /root/output/result.xlsx from /root/data/workbook.xlsx.

## 1 – Inspect the workbook
```python
import openpyxl, pprint
wb = openpyxl.load_workbook('/root/data/workbook.xlsx', data_only=True)
for s in wb.sheetnames:
    print(f'--- {s} ---')
    ws = wb[s]
    print(f'  rows: {ws.min_row}-{ws.max_row}, cols: {ws.min_column}-{ws.max_column}')
# Task sheet: print rows 1-55, cols A-M to understand layout
ts = wb['Task']
for r in ts.iter_rows(min_row=1, max_row=55, max_col=13, values_only=False):
    print([(c.coordinate, c.value) for c in r])
# Data sheet: print rows 1-40, cols A-Z
ds = wb['Data']
for r in ds.iter_rows(min_row=1, max_row=40, max_col=26, values_only=False):
    print([(c.coordinate, c.value) for c in r])
wb.close()
```
Study the output carefully. Identify:
- The series codes in column D for rows 12-17, 19-24, 26-31.
- The years in H10:L10.
- The Data sheet layout in rows 21-38 (which column holds the series code, which columns hold the year values, and which row holds the year headers).
- The labels/structure in rows 35-50.

## 2 – Build and inject formulas with openpyxl (data_only=False)
```python
import openpyxl, os, shutil
os.makedirs('/root/output', exist_ok=True)
shutil.copy('/root/data/workbook.xlsx', '/root/output/result.xlsx')
wb = openpyxl.load_workbook('/root/output/result.xlsx')  # data_only=False to preserve formulas
ts = wb['Task']
```

### Step 1 – Lookup formulas in H12:L31
For each yellow cell in the three blocks (H12:L17, H19:L24, H26:L31), write an INDEX/MATCH formula. The pattern should be:
```
=INDEX(Data!<value_columns>, MATCH($D12, Data!<series_code_column>, 0), MATCH(H$10, Data!<year_header_row>, 0))
```
Adjust the exact ranges based on what you found in the inspection step. Use absolute references on the series-code column ($D12) and the year row (H$10) so formulas copy correctly across the block. The Data!<value_columns> should cover the rectangular block of numeric data in rows 21:38. The series code column and year header row must be identified from the inspection.

**IMPORTANT**: Use `MATCH` (not `XMATCH`), `INDEX` (not `XLOOKUP`), and standard Excel function names only. The failed campus-budget task used unrecognized function names that caused #NAME? errors – avoid this.

Write formulas for all 6 rows × 5 columns in each of the three blocks.

### Step 2 – Net capacity headroom (H35:L40)
For each cell in H35:L40, write:
```
=(H12 - H19) / H26 * 100
```
where H12 corresponds to 'Available Care Slots' (rows 12-17), H19 to 'Occupied Care Slots' (rows 19-24), and H26 to 'Staffed Bed Capacity' (rows 26-31). Adjust row references for each of the 6 hospital clusters.

For example, H35 = (H12-H19)/H26*100, H36 = (H13-H20)/H27*100, etc.

### Step 2 continued – Summary statistics (H42:L47)
For each column H through L:
- Row 42: `=MIN(H35:H40)`
- Row 43: `=MAX(H35:H40)`
- Row 44: `=MEDIAN(H35:H40)`
- Row 45: `=AVERAGE(H35:H40)`
- Row 46: `=PERCENTILE(H35:H40,0.25)`  (use PERCENTILE, not PERCENTILE.INC or PERCENTILE.EXC)
- Row 47: `=PERCENTILE(H35:H40,0.75)`

**CRITICAL**: Use `PERCENTILE` not `PERCENTILE.INC` or `PERCENTILE.EXC`. The campus-budget failure was caused by #NAME? errors from unrecognized function names. Stick to classic Excel function names.

Verify the row labels match the expected statistics. If the labels say something different (e.g., row 42 is MAX not MIN), adjust accordingly.

### Step 3 – Weighted mean (H50:L50)
For each column H through L:
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```

## 3 – Save
```python
wb.save('/root/output/result.xlsx')
wb.close()
```

## 4 – Validate
Reopen the saved file and print out the formulas in key cells to confirm they are correctly written:
```python
wb2 = openpyxl.load_workbook('/root/output/result.xlsx')
ts2 = wb2['Task']
for coord in ['H12','L17','H19','L24','H26','L31','H35','L40','H42','H43','H44','H45','H46','H47','H50','L50']:
    print(f'{coord}: {ts2[coord].value}')
wb2.close()
```
Confirm all formulas use valid function names and correct references. If any look wrong, fix and re-save.

## Key Constraints
- Do NOT add sheets, macros, VBA, external links, or helper tabs.
- Do NOT change existing formatting.
- Use only classic Excel function names: INDEX, MATCH, MIN, MAX, MEDIAN, AVERAGE, PERCENTILE, SUMPRODUCT, SUM.
- Save final result to /root/output/result.xlsx.

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