# Task Instruction

Execute the following steps to complete the task.

## Step 0 – Inspect the workbook
1. Copy the source workbook:  
   ```
   cp /root/data/workbook.xlsx /root/output/result.xlsx
   ```
2. Open `/root/output/result.xlsx` with `openpyxl` (do NOT use `data_only=True`).
3. Print the following so you can verify every assumption before writing formulas:
   - **Task sheet**: Print rows 10-50, columns A-L (values and any existing formulas). Pay special attention to:
     - Row 10 (the year headers in columns H-L).
     - Column D rows 12-17, 19-24, 26-31 (the series codes).
     - Rows 35-40 labels and column D values.
     - Rows 42-47 labels (should be Min, Max, Median, Average/Mean, 25th percentile, 75th percentile or similar).
     - Row 50 label.
   - **Data sheet**: Print rows 20-39, columns A-Z (or at least through the last populated column). Identify:
     - Which row contains the year headers (likely row 21).
     - Which column contains the series codes (likely column B).
     - The exact data range (first data row, last data row, first data column, last data column).

## Step 1 – Write lookup formulas in the three blocks

Based on the inspection, construct `INDEX/MATCH/MATCH` formulas. The general pattern for cell `H12` would be something like:

```
=INDEX(Data!$C$22:$<lastcol>$38, MATCH($D12, Data!$B$22:$B$38, 0), MATCH(H$10, Data!$C$21:$<lastcol>$21, 0))
```

**Adjust all references** based on what you actually see in the inspection:
- The data array should span from the first numeric column to the last numeric column, and from the first data row to the last data row.
- The row lookup vector is the series-code column in Data (likely column B), spanning the same rows as the data array.
- The column lookup vector is the year header row in Data, spanning the same columns as the data array.
- `$D12` references the series code on the Task sheet (column D of the current row). Make column D absolute (`$D`) and row relative.
- `H$10` references the year in row 10. Make row 10 absolute (`$10`) and column relative.

Write these formulas into every cell in:
- `H12:L17` (block 1 – e.g., Committed Funding)
- `H19:L24` (block 2 – e.g., Operating Spend)
- `H26:L31` (block 3 – e.g., Approved Budget Base)

Loop over each block and each cell, constructing the formula string in Python and assigning it to the cell.

## Step 2 – Net Budget Buffer and summary statistics

In `H35:L40`, write formulas for Net Budget Buffer:
```
=(H12-H19)/H26*100
```
Adjust row references for each of the 6 department rows (rows 35-40 map to the department offsets in rows 12-17, 19-24, 26-31). Specifically:
- Row 35 uses rows 12, 19, 26
- Row 36 uses rows 13, 20, 27
- Row 37 uses rows 14, 21, 28
- Row 38 uses rows 15, 22, 29
- Row 39 uses rows 16, 23, 30
- Row 40 uses rows 17, 24, 31

In `H42:L47`, write summary statistics over `H35:H40` (adjust column for each of H-L). Check the labels in column A/B/C/D of rows 42-47 to confirm the order. The expected mapping is:
- Row 42: `=MIN(H35:H40)` (or `=MIN(H$35:H$40)`)
- Row 43: `=MAX(H35:H40)`
- Row 44: `=MEDIAN(H35:H40)`
- Row 45: `=AVERAGE(H35:H40)`
- Row 46: `=PERCENTILE(H35:H40,0.25)`
- Row 47: `=PERCENTILE(H35:H40,0.75)`

**Verify the label-to-function mapping** against the actual labels you printed. Adjust if the order differs.

## Step 3 – Weighted mean

In `H50:L50`, write:
```
=SUMPRODUCT(H35:H40,H26:H31)/SUM(H26:H31)
```
(Adjust column letter for each of H through L.)

## Step 4 – Save and validate

1. Save the workbook (`wb.save('/root/output/result.xlsx')`).
2. Re-open the saved file and print cells `H12`, `H19`, `H26`, `H35`, `H42`, `H50` to confirm formulas are present (not None).
3. Confirm no extra sheets were added.
4. Confirm the file exists at `/root/output/result.xlsx`.

## Critical reminders
- Do NOT use `data_only=True` when opening the workbook.
- Do NOT add sheets, macros, VBA, or external links.
- Do NOT alter existing formatting.
- Use `openpyxl` for all operations.
- Always inspect the actual workbook structure BEFORE writing any formulas. The exact row/column layout may differ from assumptions.

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