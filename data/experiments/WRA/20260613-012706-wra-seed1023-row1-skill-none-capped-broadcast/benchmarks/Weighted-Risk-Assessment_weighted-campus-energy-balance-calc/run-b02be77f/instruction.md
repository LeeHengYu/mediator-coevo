# Task Instruction

Execute the following steps precisely to complete the weighted campus energy balance workbook task.

## Step 0: Inspect the workbook
1. Copy `/root/data/workbook.xlsx` to `/root/output/result.xlsx`.
2. Open `/root/output/result.xlsx` using openpyxl and inspect:
   - Sheet `Task`: Read and print cells in column D rows 12-17, 19-24, 26-31 to see the series codes. Read row 10 columns H-L to see the years. Read cells H35:L40, H42:L47, H50:L50 to see what's there (likely empty). Read any labels in column A or B for rows 12-17, 19-24, 26-31, 35-40, 42-47, 50 to understand the structure. Also check what row 10 looks like across columns H-L (the year headers).
   - Sheet `Data`: Read rows 21-38 to understand the data layout. Print the first row (row 21 or whichever is the header) across many columns to see how data is organized. Determine whether data is arranged with series codes in a column and years across rows, or vice versa.
3. Print all findings before making any edits.

## Step 1: Populate lookup formulas in H12:L17, H19:L24, H26:L31

Using openpyxl, write Excel formulas (not computed values) into each yellow cell. For each cell at row `r` and column `c` (H=8, I=9, J=10, K=11, L=12):

- The series code is in `$D{r}` (column D of the same row).
- The year is in the corresponding column of row 10, e.g., `H$10`, `I$10`, etc.
- The data source is on sheet `Data` rows 21:38.

Based on the data layout discovered in Step 0, use one of the allowed lookup patterns. The most likely pattern is INDEX-MATCH:

```
=INDEX(Data!<data_range>, MATCH($D{r}, Data!<series_code_column>, 0), MATCH({col}$10, Data!<year_row>, 0))
```

Adjust the exact ranges based on what you find in Step 0. The key requirements:
- Every formula must use TWO inputs: the series code from column D of the current row, and the year from row 10.
- The lookup must reference Data sheet rows 21:38.
- Must use one of: VLOOKUP+MATCH, HLOOKUP+MATCH, XLOOKUP+MATCH, or INDEX+MATCH.

IMPORTANT: Use `Translator` from `openpyxl.formula.translate` or manually construct each formula string. Make sure dollar signs are correct: lock the row for the year reference (`$10`) and lock the column for the series code (`$D`).

## Step 2: Net renewable balance formulas in H35:L40

For each campus (6 campuses, rows 35-40), and each year column (H-L), write a formula:
```
=(<Renewable_Generation_cell> - <Grid_Consumption_cell>) / <Baseline_Energy_Demand_cell> * 100
```

Where:
- Renewable Generation is in rows 12-17 (the first block)
- Grid Consumption is in rows 19-24 (the second block)
- Baseline Energy Demand is in rows 26-31 (the third block)

So for row 35 col H: `=(H12-H19)/H26*100`, for row 36 col H: `=(H13-H20)/H27*100`, etc.

Verify the row mapping by checking the campus names/labels.

Then in H42:L47, write column-wise aggregate formulas over H35:L40:
- Row 42: `=MIN(H35:H40)` (and similarly for I-L)
- Row 43: `=MAX(H35:H40)`
- Row 44: `=MEDIAN(H35:H40)`
- Row 45: `=AVERAGE(H35:H40)`
- Row 46: `=PERCENTILE(H35:H40,0.25)`
- Row 47: `=PERCENTILE(H35:H40,0.75)`

Check the labels in column A/B for rows 42-47 to confirm which statistic goes in which row. Adjust the row assignments accordingly.

## Step 3: Weighted mean in H50:L50

For each column c in H-L:
```
=SUMPRODUCT({c}35:{c}40, {c}26:{c}31) / SUM({c}26:{c}31)
```

For example H50: `=SUMPRODUCT(H35:H40,H26:H31)/SUM(H26:H31)`

## Step 4: Save and validate
1. Save the workbook to `/root/output/result.xlsx`.
2. Re-open it and verify:
   - Cells H12:L31 contain formula strings (not None, not plain values).
   - Cells H35:L40 contain formulas.
   - Cells H42:L47 contain formulas.
   - Cells H50:L50 contain formulas.
   - No new sheets were added.
   - Print a sample of formulas to confirm correctness.
3. Confirm the file is saved at `/root/output/result.xlsx`.

## Critical Notes
- Use `openpyxl` with `data_only=False` (default) so formulas are preserved.
- Do NOT use `data_only=True` when reading.
- When writing formulas, ensure they start with `=`.
- Do NOT modify any existing formatting, values, or structure outside the specified cells.
- Do NOT add sheets, macros, VBA, external links, or helper tabs.
- If the Data sheet has a specific structure (e.g., series codes in column A and years in a header row), adapt the INDEX-MATCH formula accordingly. Print the structure before writing formulas.

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