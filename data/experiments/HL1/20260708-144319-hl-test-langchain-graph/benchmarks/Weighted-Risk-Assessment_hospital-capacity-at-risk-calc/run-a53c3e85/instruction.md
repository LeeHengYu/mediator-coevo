# Task Instruction

Execute the following steps to produce /root/output/result.xlsx from /root/data/workbook.xlsx.

## Step 0 – Inspect the workbook structure

```python
import openpyxl, os
wb = openpyxl.load_workbook('/root/data/workbook.xlsx')
for s in wb.sheetnames:
    print(f'--- Sheet: {s} ---')
    ws = wb[s]
    print(f'  Dimensions: {ws.dimensions}')
    # Print first 50 rows with content
    for row in ws.iter_rows(min_row=1, max_row=50, values_only=False):
        vals = [(c.coordinate, c.value) for c in row if c.value is not None]
        if vals:
            print(f'  {vals}')
```

Run this and study the output carefully. Identify:
- The series codes in column D of the Task sheet (rows 12-17, 19-24, 26-31).
- The year headers in row 10 of the Task sheet (columns H-L).
- The layout of the Data sheet rows 21-38: which row holds series codes, which row holds years, and where the data values are.
- The exact cell references for Staffed Bed Capacity (H26:L31), Available Care Slots, and Occupied Care Slots blocks.
- The labels/structure in rows 35-50 of the Task sheet.

## Step 1 – Write lookup formulas in H12:L31

Using openpyxl, write INDEX/MATCH formulas into cells H12:L31 (covering the three blocks H12:L17, H19:L24, H26:L31).

The formula pattern for each cell should be:
```
=INDEX(Data!<data_range>, MATCH($D{row}, Data!<series_code_column>, 0), MATCH({col}$10, Data!<year_row>, 0))
```

Use the actual ranges discovered in Step 0. Key points:
- Lock the column reference for the series code with `$D{row}` so it doesn't shift across columns.
- Lock the row reference for the year with `{col}$10` so it doesn't shift down rows.
- The Data sheet range references must cover rows 21:38 appropriately.
- Skip any cells that are NOT in the yellow target ranges (rows 18, 25, and 32+ are not lookup rows).

## Step 2 – Net capacity headroom in H35:L40

For each cell in H35:L40, write a formula:
```
=({Available_Care_Slots_cell} - {Occupied_Care_Slots_cell}) / {Staffed_Bed_Capacity_cell} * 100
```

Map each of the six hospital cluster rows (35-40) to the corresponding rows in the three blocks above. For example, if cluster 1 is in row 12/19/26 of the three blocks, then row 35 references those. Verify the mapping from the Task sheet labels.

## Step 3 – Summary statistics in H42:L47

For each column (H through L), write these formulas:
- Row 42: `=MIN({col}35:{col}40)`
- Row 43: `=MAX({col}35:{col}40)`
- Row 44: `=MEDIAN({col}35:{col}40)`
- Row 45: `=AVERAGE({col}35:{col}40)`
- Row 46: `=PERCENTILE({col}35:{col}40, 0.25)`
- Row 47: `=PERCENTILE({col}35:{col}40, 0.75)`

Verify the row labels match (check which row is MIN, MAX, etc.).

## Step 4 – Weighted mean in H50:L50

For each column (H through L):
```
=SUMPRODUCT({col}35:{col}40, {col}26:{col}31) / SUM({col}26:{col}31)
```

This uses the Net capacity headroom percentages as values and Staffed Bed Capacity as weights.

## Step 5 – Save

```python
os.makedirs('/root/output', exist_ok=True)
wb.save('/root/output/result.xlsx')
```

## Verification

After saving, reload the workbook and print the formulas in a few sample cells (e.g., H12, L17, H19, L24, H26, L31, H35, H42, H47, H50) to confirm they are correctly written as formula strings (starting with '='). Confirm no sheets were added or removed. Confirm the file exists at /root/output/result.xlsx.

## Important Notes
- Do NOT use data_only=True when loading; formulas must be preserved.
- Do NOT add any new sheets, macros, or VBA.
- Do NOT change any existing formatting.
- If the row-label mapping for statistics (rows 42-47) differs from MIN/MAX/MEDIAN/AVERAGE/P25/P75, adjust to match the actual labels in the Task sheet.
- If the Data sheet layout differs from expectations, adapt the INDEX/MATCH ranges accordingly based on Step 0 findings.

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