# Task Instruction

Execute the following steps exactly, in order.

## Step 0 – Inspect the source workbook

```python
import openpyxl, pprint
wb = openpyxl.load_workbook('/root/data/workbook.xlsx')

# ---- Task sheet layout ----
ts = wb['Task']
print('=== Task sheet: row 10 (years) ===')
for c in range(8, 13):                       # H-L = cols 8-12
    print(f'  {ts.cell(10, c).value}', end='')
print()

print('\n=== Task sheet: D column series codes ===')
for r in range(12, 32):
    print(f'  row {r}: D={ts.cell(r, 4).value}   (label col C = {ts.cell(r, 3).value})')

print('\n=== Task sheet: rows 35-50, col C-D ===')
for r in range(35, 51):
    print(f'  row {r}: C={ts.cell(r, 3).value}  D={ts.cell(r, 4).value}')

print('\n=== Task sheet: rows 42-47 col C ===')
for r in range(42, 48):
    print(f'  row {r}: C={ts.cell(r, 3).value}')

# ---- Data sheet layout ----
ds = wb['Data']
print('\n=== Data sheet: row 20 (header?) and rows 21-38 ===')
for r in range(20, 39):
    vals = []
    for c in range(1, ds.max_column+1):
        vals.append(ds.cell(r, c).value)
    print(f'  row {r}: {vals}')

print('\n=== Data sheet: row 1 header ===')
for c in range(1, ds.max_column+1):
    print(f'  col {c}: {ds.cell(1, c).value}')

print('\nData sheet dims:', ds.min_row, ds.max_row, ds.min_column, ds.max_column)
wb.close()
```

Record the exact column layout of the Data sheet (which column holds the series code, which columns hold years, which rows hold data). Record the exact series codes in column D of the Task sheet and the exact year values in row 10 (H10:L10). Record the stat labels in rows 42-47.

## Step 1 – Write the formulas using openpyxl

Based on the inspection, write a Python script that:

1. Opens `/root/data/workbook.xlsx` with openpyxl (data_only=False).
2. For each cell in H12:L17, H19:L24, H26:L31, writes an INDEX/MATCH formula that:
   - Uses the series code from column D of that row on the Task sheet.
   - Uses the year from row 10 of that column on the Task sheet.
   - Looks up data from the Data sheet rows and columns identified in Step 0.
   - The formula pattern should be:
     `=INDEX(Data!<data_range>, MATCH(D{row}, Data!<series_code_column>, 0), MATCH(<col>10, Data!<year_header_row>, 0))`
   - Make sure the data range, series code column, and year header row references are correct based on Step 0 inspection.
   - Use absolute references ($) for the Data ranges but relative references for D{row} and the year cell.

3. For H35:L40, writes a formula for Net Patient Flow:
   `=(H12-H19)/H26*100` pattern, adjusting row references for each hospital (row 35 uses rows 12, 19, 26; row 36 uses rows 13, 20, 27; etc.) and each column.

4. For rows 42-47 (stats), based on the labels found in Step 0, writes:
   - Minimum: `=MIN(H35:H40)` (or the equivalent column)
   - Maximum: `=MAX(H35:H40)`
   - Median: `=MEDIAN(H35:H40)`
   - Mean: `=AVERAGE(H35:H40)`
   - 25th percentile: `=PERCENTILE(H35:H40, 0.25)`
   - 75th percentile: `=PERCENTILE(H35:H40, 0.75)`
   Map each stat label to its function based on the actual label text in column C.

5. For H50:L50, writes a SUMPRODUCT weighted mean:
   `=SUMPRODUCT(H35:H40, H26:H31)/SUM(H26:H31)` for each column H through L.

6. Saves to `/root/output/result.xlsx`.

## Step 2 – Evaluate formulas so the verifier sees numeric values

The verifier reads cell values, not formulas. After saving with openpyxl, use one of these approaches to force evaluation:

**Approach A (preferred): Use LibreOffice in headless mode to open and re-save:**
```bash
mkdir -p /root/output
# First check if libreoffice is available
which libreoffice || which soffice
# Convert: open the xlsx and re-save it, which evaluates all formulas
libreoffice --headless --calc --convert-to xlsx --outdir /root/output /root/output/result.xlsx
```
Note: LibreOffice may rename the output. Check and rename if needed so the final file is `/root/output/result.xlsx`.

**Approach B (fallback): If LibreOffice is not available**, try `ssconvert` (Gnumeric):
```bash
ssconvert /root/output/result.xlsx /root/output/result_evaluated.xlsx
mv /root/output/result_evaluated.xlsx /root/output/result.xlsx
```

**Approach C (last resort): If neither tool is available**, compute the values in Python and write them directly as values (not formulas) to the cells. This means:
- Parse the Data sheet to build a lookup dictionary.
- Compute Net Patient Flow values.
- Compute stats.
- Compute weighted means.
- Write all as numeric values.
But ALSO write the formulas as the cell values if the verifier checks for formula presence. Since the previous failure showed 'None' values, the verifier likely wants numeric values, so write numeric values directly.

## Step 3 – Validate the output

After saving, re-open `/root/output/result.xlsx` with openpyxl (data_only=True) and print the values in:
- H12:L17 (should be numbers, not None)
- H35:L40
- H42:L47
- H50:L50

If any cell is None, the formula evaluation did not work. Fall back to Approach C.

## Critical Notes
- The previous failure was because openpyxl wrote formulas but they were not evaluated, returning None.
- The cross-task failure artifact shows #N/A errors from wrong lookup ranges. Double-check the Data sheet layout carefully.
- Do NOT add sheets, macros, VBA, external links, or helper tabs.
- Do NOT change existing formatting.
- Ensure `/root/output/` directory exists before saving.

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