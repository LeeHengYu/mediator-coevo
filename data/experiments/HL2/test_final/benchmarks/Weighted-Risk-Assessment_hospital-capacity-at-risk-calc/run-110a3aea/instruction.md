# Task Instruction

Execute the following steps carefully and in order.

## 0. Inspect the source workbook

```python
import openpyxl, pprint
wb = openpyxl.load_workbook('/root/data/workbook.xlsx', data_only=True)

# ---- Task sheet layout ----
ts = wb['Task']
print('=== Task sheet: row 10 (year headers) ===')
for c in range(8, 13):                       # H=8 .. L=12
    print(f'  col {c} ({ts.cell(10,c).coordinate}): {repr(ts.cell(10,c).value)}')

print('\n=== Task sheet: series codes col D, rows 12-31 ===')
for r in range(12, 32):
    print(f'  D{r}: {repr(ts.cell(r, 4).value)}')

print('\n=== Task sheet: row 35-40 labels (col D or nearby) ===')
for r in range(35, 41):
    for c in range(1, 8):
        v = ts.cell(r, c).value
        if v is not None:
            print(f'  {ts.cell(r,c).coordinate}: {repr(v)}')

print('\n=== Task sheet: row 42-47 labels ===')
for r in range(42, 48):
    for c in range(1, 8):
        v = ts.cell(r, c).value
        if v is not None:
            print(f'  {ts.cell(r,c).coordinate}: {repr(v)}')

print('\n=== Task sheet: row 50 labels ===')
for r in range(49, 52):
    for c in range(1, 8):
        v = ts.cell(r, c).value
        if v is not None:
            print(f'  {ts.cell(r,c).coordinate}: {repr(v)}')

# ---- Data sheet layout ----
ds = wb['Data']
print('\n=== Data sheet: rows 19-40, first 20 cols ===')
for r in range(19, 41):
    vals = []
    for c in range(1, 21):
        v = ds.cell(r, c).value
        if v is not None:
            vals.append(f'{ds.cell(r,c).coordinate}={repr(v)}')
    if vals:
        print(f'  row {r}: ' + ', '.join(vals))

print('\n=== Data sheet: row 21 full scan (up to col 30) ===')
for c in range(1, 31):
    v = ds.cell(21, c).value
    if v is not None:
        print(f'  {ds.cell(21,c).coordinate}: {repr(v)}')

wb.close()
```

Run this and **read every line of output carefully** before proceeding. The output tells you:
- The exact year values in Task!H10:L10 (could be int, float, or string).
- The exact series-code strings in Task!D12:D17, D19:D24, D26:D31.
- The exact layout of Data rows 21-38: which column holds the series code (the lookup key), which row holds the year headers, and where the numeric data lives.

## 1. Build the lookup formulas (H12:L17, H19:L24, H26:L31)

Based on what you see in step 0, construct INDEX/MATCH formulas. The critical things to get right:

- **Lookup-key column on Data sheet**: identify which column in Data!rows 21:38 contains the series codes. Call it DATA_KEY_COL (e.g. column A, B, C…).
- **Year-header row on Data sheet**: identify which row in Data contains the year values that match Task!H10:L10. Call it DATA_YEAR_ROW.
- **Data range**: the rectangular block of numbers in Data rows 21:38.

Use this pattern (adjust ranges to match what you found):

```
=INDEX(Data!$<first_data_col>$21:$<last_data_col>$38,
       MATCH($D12, Data!$<key_col>$21:$<key_col>$38, 0),
       MATCH(H$10, Data!$<first_data_col>$<year_row>:$<last_data_col>$<year_row>, 0))
```

Make sure:
- The MATCH for the series code searches the **exact same rows** (21:38) as the INDEX array.
- The MATCH for the year searches the **exact same columns** as the INDEX array.
- Dollar signs lock rows/columns appropriately so the formula can be dragged across H-L and down rows.
- If the year values are stored as numbers in one sheet and text in the other, wrap one side in VALUE() or TEXT() or use exact matching (0).

**Before writing formulas to the workbook, print the first formula string you plan to use for cell H12 and visually verify it against the Data layout.**

Write formulas for all 3 blocks: H12:L17, H19:L24, H26:L31.

## 2. Net capacity headroom (H35:L40)

For each of the 6 hospital clusters (rows 35-40) and each year column (H-L):

```
= (H12 - H19) / H26 * 100
```

where row 12 = Available Care Slots, row 19 = Occupied Care Slots, row 26 = Staffed Bed Capacity. Adjust row references so each cluster maps correctly (cluster 1 uses rows 12/19/26, cluster 2 uses rows 13/20/27, etc.).

## 3. Summary statistics (H42:L47)

For each year column, compute over the 6-row block H35:H40 (through L35:L40):
- Row 42: MIN
- Row 43: MAX
- Row 44: MEDIAN
- Row 45: AVERAGE
- Row 46: PERCENTILE(range, 0.25)  or  PERCENTILE.INC(range, 0.25)
- Row 47: PERCENTILE(range, 0.75)  or  PERCENTILE.INC(range, 0.75)

**Check the labels in rows 42-47 from your inspection to confirm the correct order of statistics.**

## 4. Weighted mean (H50:L50)

For each year column:
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```

## 5. Save

Save to `/root/output/result.xlsx`. Create the output directory if needed. Do NOT change sheet names, formatting, or add sheets/macros.

## 6. Validate

After saving, reopen the file with openpyxl (data_only=False) and print:
- The formula in H12 (should be a string starting with '=')
- The formula in H35
- The formula in H42, H45, H46, H47
- The formula in H50

Then reopen with data_only=True (note: openpyxl won't evaluate formulas, so values may be None — that's OK). The key validation is that the formula strings look correct.

Alternatively, if you can use xlcalc or another evaluator, verify H12 returns 252.0.

## CRITICAL REMINDERS
- The #N/A errors in the previous run were caused by misaligned ranges in the lookup formulas. **Inspect the Data sheet layout first** and match ranges exactly.
- Pay attention to data types: if years are integers on one sheet and strings on another, the MATCH will fail.
- Do not guess at the Data sheet layout — read it from the file.
- Use openpyxl for reading/writing. Write formula strings (starting with '=') into cells.

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