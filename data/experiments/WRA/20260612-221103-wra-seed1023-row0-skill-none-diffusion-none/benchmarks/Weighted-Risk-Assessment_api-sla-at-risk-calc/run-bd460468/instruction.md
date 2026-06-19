# Task Instruction

Execute the following steps in a single Python script using openpyxl to produce `/root/output/result.xlsx`.

## Preliminary Inspection

1. Load `/root/data/workbook.xlsx` with `openpyxl.load_workbook('/root/data/workbook.xlsx')`. Do NOT use `data_only=True`.
2. Print the sheet names to confirm `Task` and `Data` exist.
3. On sheet `Task`, print:
   - Row 10 (columns A–L) to see the year headers.
   - Column D, rows 12–17, 19–24, 26–31 to see the series codes for each block.
   - Rows 35–40 column D to see the service names for Net SLA buffer.
   - Row 42–47 column D (or wherever labels are) to see the stat labels (min, max, median, mean, 25th pct, 75th pct).
   - Row 50 column D to see the weighted mean label.
4. On sheet `Data`, print:
   - Row 20 (or the header row just above row 21) columns A–Z to see column headers.
   - Rows 21–38, columns A–F (or more) to see the structure of source data (series codes, years, values).
5. Identify:
   - Which column in `Data` contains the series code (call it `code_col`).
   - Which row in `Data` contains year headers (if HLOOKUP-style) or which column contains years.
   - The data range layout.

Print all of this clearly before writing any formulas.

## Step 1: Lookup Formulas in H12:L17, H19:L24, H26:L31

For each block and each cell, write an `INDEX(MATCH,MATCH)` formula. The pattern:
```
=INDEX(Data!$<data_range>, MATCH($D<row>, Data!$<code_column>, 0), MATCH(H$10, Data!$<year_row>, 0))
```

Adjust the references based on what you discovered in the inspection:
- `$D<row>` is the series code in column D of the current row on `Task`.
- `H$10` (or I$10, J$10, etc.) is the year from row 10.
- The data range and match ranges come from `Data` rows 21:38.

Make the row reference in `$D<row>` use a dollar sign on the column (`$D`) and the column reference in the year use a dollar sign on the row (`$10`), so the formula is correct for each cell.

Write formulas for all cells in H12:L17, H19:L24, H26:L31 (that's 3 blocks × 6 rows × 5 columns = 90 cells).

## Step 2: Net SLA Buffer in H35:L40

The formula for each cell is:
```
=(H12 - H19) / H26 * 100
```
where H12 corresponds to "Latency Budget Preserved" (rows 12–17), H19 to "Latency Budget Consumed" (rows 19–24), and H26 to "Covered Request Capacity" (rows 26–31). Adjust row offsets so row 35 uses rows 12, 19, 26; row 36 uses rows 13, 20, 27; etc.

Write formulas for all 30 cells (6 rows × 5 columns).

## Step 3: Summary Statistics in H42:L47

Based on the labels you found in the inspection, write formulas for each column H through L. The stats block covers rows 42–47. Use these Excel functions:
- MIN: `=MIN(H35:H40)`
- MAX: `=MAX(H35:H40)`
- MEDIAN: `=MEDIAN(H35:H40)`
- AVERAGE: `=AVERAGE(H35:H40)`
- 25th percentile: `=PERCENTILE(H35:H40,0.25)`
- 75th percentile: `=PERCENTILE(H35:H40,0.75)`

**CRITICAL**: Use `PERCENTILE` (not `PERCENTILE.INC` or `PERCENTILE.EXC`). The function name must be exactly `PERCENTILE` with no period. Verify the label order from your inspection to assign the correct function to the correct row. Do NOT assume the order — read the labels.

## Step 4: Weighted Mean in H50:L50

For each column (H through L):
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```
This computes the weighted mean of Net SLA buffer percentages weighted by Covered Request Capacity.

## Saving

1. Create `/root/output/` directory if it doesn't exist.
2. Save the workbook to `/root/output/result.xlsx`.
3. Reload the saved file and print a sample of cells (e.g., H12, H19, H26, H35, H42, H46, H47, H50) to confirm formulas are present and not None.

## Important Constraints
- Do NOT add sheets, macros, VBA, external links, or helper tabs.
- Do NOT change existing formatting.
- Do NOT use `data_only=True` when loading.
- Use `PERCENTILE` not `PERCENTILE.INC` for the percentile functions.
- Verify every formula string is a valid Excel formula (starts with `=`).
- Match the exact row-to-statistic mapping from the labels in the workbook.

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