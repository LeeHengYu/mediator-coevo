# Task Instruction

Execute the following steps to produce `/root/output/result.xlsx`.

## 0 – Inspect the workbook
```python
import openpyxl, os, shutil
wb = openpyxl.load_workbook('/root/data/workbook.xlsx')
ts = wb['Task']
ds = wb['Data']

# Print Task sheet layout: headers, labels, yellow-cell regions
for r in range(1, 55):
    row_vals = []
    for c in range(1, 15):
        cell = ts.cell(r, c)
        row_vals.append(f"{cell.coordinate}={cell.value}")
    print(' | '.join(row_vals))

print('\n--- Data sheet rows 1-45 ---')
for r in range(1, 45):
    row_vals = []
    for c in range(1, 15):
        cell = ds.cell(r, c)
        row_vals.append(f"{cell.coordinate}={cell.value}")
    print(' | '.join(row_vals))
```
Read all output carefully before proceeding.

## 1 – Understand the structure
- Identify the series codes in column D for each of the three blocks (H12:L17, H19:L24, H26:L31).
- Identify the years in row 10 spanning columns H–L.
- Identify the Data sheet layout in rows 21–38: which row holds which series, which column holds which year.
- Identify what the three blocks represent (likely "Latency Budget Preserved", "Latency Budget Consumed", and "Covered Request Capacity").

## 2 – Write lookup formulas (Step 1)
Use the `INDEX/MATCH/MATCH` pattern with the Data sheet range `Data!$A$21:$<lastcol>$38`.

For each yellow cell in the three blocks, write a formula like:
```
=INDEX(Data!$B$21:$<lastcol>$38, MATCH($D12, Data!$A$21:$A$38, 0), MATCH(H$10, Data!$B$20:$<lastcol>$20, 0))
```
Adjust column letters and row numbers based on the actual inspection. Use `$D<row>` (absolute column, relative row) and `<col>$10` (relative column, absolute row) so the formula fills correctly across the 6×5 grid in each block.

Apply the formulas to all three blocks: H12:L17, H19:L24, H26:L31.

## 3 – Net SLA buffer (Step 2, rows 35–40)
For each cell in H35:L40, write:
```
=(H12-H19)/H26*100
```
adjusting row references so that:
- Row 35 uses rows 12, 19, 26
- Row 36 uses rows 13, 20, 27
- etc. through row 40 using rows 17, 24, 31

This computes `(Latency Budget Preserved - Latency Budget Consumed) / Covered Request Capacity * 100`.

**Verify the block labels** before assuming which block is which. The formula numerator/denominator must match the task description.

## 4 – Summary statistics (Step 2, rows 42–47)
For H42:L47, write column-wise formulas over H35:L40 (the Net SLA buffer block):
- Row 42 (MIN): `=MIN(H35:H40)`
- Row 43 (MAX): `=MAX(H35:H40)`
- Row 44 (MEDIAN): `=MEDIAN(H35:H40)`
- Row 45 (MEAN): `=AVERAGE(H35:H40)`
- Row 46 (25th percentile): `=PERCENTILE(H35:H40, 0.25)`
- Row 47 (75th percentile): `=PERCENTILE(H35:H40, 0.75)`

**IMPORTANT**: Check the labels in column D/E/F/G for rows 42–47 to confirm the order. Map each statistic to the correct row based on the label. Use `PERCENTILE` (not `PERCENTILE.INC` or `PERCENTILE.EXC`) unless the labels say otherwise — `PERCENTILE` is safest for compatibility and avoids `#NAME?` errors that were seen in a related task.

## 5 – Weighted mean (Step 3, row 50)
For H50:L50, write:
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```
This uses Net SLA buffer percentages as values and Covered Request Capacity as weights.

## 6 – Save
```python
os.makedirs('/root/output', exist_ok=True)
wb.save('/root/output/result.xlsx')
```

## 7 – Verify
Reload the saved file with openpyxl (data_only=False) and print all formula cells in the Task sheet to confirm:
- All cells in H12:L17, H19:L24, H26:L31 contain INDEX/MATCH formulas
- All cells in H35:L40 contain the net buffer formula
- All cells in H42:L47 contain the correct stat functions
- All cells in H50:L50 contain SUMPRODUCT formulas
- No cells are empty or contain literal values where formulas are expected

Also check that no new sheets were added and existing formatting is preserved.

## Key Cautions
- Use `PERCENTILE` not `PERCENTILE.INC` or `PERCENTILE.EXC` to avoid #NAME? errors.
- Confirm block-to-meaning mapping by reading labels before writing formulas.
- Use absolute/relative references correctly so formulas fill across the 6-row × 5-column grids.
- Do NOT add sheets, macros, VBA, or helper columns.

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