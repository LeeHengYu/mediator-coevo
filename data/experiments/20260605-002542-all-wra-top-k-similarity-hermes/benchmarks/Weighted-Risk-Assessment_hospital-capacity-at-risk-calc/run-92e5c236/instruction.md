# Task Instruction

Execute the following steps carefully to produce `/root/output/result.xlsx`.

## Preparation

1. `mkdir -p /root/output`
2. Read the workbook `/root/data/workbook.xlsx` using `openpyxl` (with `data_only=False` so formulas are preserved).
3. Inspect the `Task` sheet thoroughly:
   - Print rows 10-50 for columns D through L to understand the layout: series codes in column D, years in row 10, yellow target regions.
   - Print the exact content of cells D12:D17, D19:D24, D26:D31 to see the series codes for each block.
   - Print H10:L10 to see the year headers.
4. Inspect the `Data` sheet:
   - Print rows 21-38 completely (all columns) to understand the data layout: which row holds which series, which columns hold which years, and the exact structure.
   - Identify the header row for the data table (likely row 20 or 21) and the column layout.

## Step 1: Populate lookup formulas in H12:L17, H19:L24, H26:L31

For each yellow cell at row `r`, column `c` (H=8, I=9, J=10, K=11, L=12):
- The series code is in `Task!D{r}`.
- The year is in `Task!{col_letter}10` (e.g., H10, I10, etc.).
- The data is on `Data` sheet rows 21:38.

Based on the Data sheet inspection, construct an INDEX/MATCH formula. The typical pattern is:

```
=INDEX(Data!<data_range>, MATCH(D{r}, Data!<series_column>, 0), MATCH({col}10, Data!<year_header_row>, 0))
```

Adjust the ranges based on what you see in the Data sheet. The `<data_range>` should cover the full rectangular block of data rows 21:38 and all relevant columns. The `<series_column>` is the column in Data that holds series codes. The `<year_header_row>` is the row in Data that holds year values.

**Important:** Make sure the MATCH for the year header references the same row range that contains the year labels in the Data sheet, and the MATCH for the series code references the same column range that contains series codes.

Write all 54 formulas (18 rows × 3 blocks × 5 columns) into the cells.

## Step 2: Net capacity headroom in H35:L40

For each column c (H through L) and each cluster row offset i (0 through 5):
- Available Care Slots = row 12+i (H12:L17 block)
- Occupied Care Slots = row 19+i (H19:L24 block)  
- Staffed Bed Capacity = row 26+i (H26:L31 block)

Formula for cell at row 35+i, column c:
```
=({c}{12+i} - {c}{19+i}) / {c}{26+i} * 100
```
For example, H35 = `=(H12-H19)/H26*100`

Write all 30 formulas.

## Step 2 continued: Statistics in H42:L47

For each column c (H through L), rows 42-47 should contain:
- Row 42: `=MIN({c}35:{c}40)`
- Row 43: `=MAX({c}35:{c}40)`
- Row 44: `=MEDIAN({c}35:{c}40)`
- Row 45: `=AVERAGE({c}35:{c}40)`
- Row 46: `=PERCENTILE({c}35:{c}40, 0.25)`  
- Row 47: `=PERCENTILE({c}35:{c}40, 0.75)`

**CRITICAL:** Check what labels are in column D (or nearby) for rows 42-47 to confirm the exact order (min, max, median, mean, 25th, 75th). Adjust the row assignments to match the labels. The order above is a guess — use the actual labels.

**CRITICAL about PERCENTILE:** Use `PERCENTILE` (not `PERCENTILE.INC` or `PERCENTILE.EXC`) because `openpyxl` may not recognize the dotted variants and they will produce `#NAME?` errors. Similarly, use `AVERAGE` not `MEAN`. If you see the cross-task artifacts, the `#NAME?` errors in H46:L47 were caused by using function names that openpyxl/Excel didn't recognize — so stick to basic function names: `MIN`, `MAX`, `MEDIAN`, `AVERAGE`, `PERCENTILE`.

## Step 3: Weighted mean in H50:L50

For each column c (H through L):
```
=SUMPRODUCT({c}35:{c}40, {c}26:{c}31) / SUM({c}26:{c}31)
```

This computes the weighted mean of the net capacity headroom percentages using Staffed Bed Capacity as weights.

## Final steps

1. After writing all formulas, save the workbook to `/root/output/result.xlsx`.
2. Re-open the saved file and print the formula content of a sample of cells (e.g., H12, H35, H42, H46, H47, H50) to verify formulas were written correctly.
3. Do NOT add any new sheets, macros, VBA, or change formatting.
4. Do NOT use `data_only=True` when saving — formulas must be preserved as formulas.

## Important notes
- Before writing formulas, inspect the Data sheet carefully to get the exact row/column layout.
- Adapt all ranges to what you actually observe in the sheets.
- Use `openpyxl` to write formulas as strings starting with `=`.
- Double-check that PERCENTILE (not PERCENTILE.INC/PERCENTILE.EXC) is used for the percentile rows.

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