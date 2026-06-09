# Task Instruction

Execute the following steps to produce `/root/output/result.xlsx`.

## Phase 0 – Inspect the workbook
1. `mkdir -p /root/output`
2. Open `/root/data/workbook.xlsx` with openpyxl (do NOT use `data_only=True`).
3. Print the sheet names to confirm `Task` and `Data` exist.
4. On sheet `Task`, print:
   - Row 10 (columns A–L) to see the year headers.
   - Column D for rows 12–17, 19–24, 26–31 to see the series codes.
   - Row 35 label area and rows 35–40 column D to see department names.
   - Rows 42–47 column D/E to see stat labels.
   - Row 50 columns D–G to see the weighted-mean label.
5. On sheet `Data`, print:
   - Row 21 through row 38 for columns A–L (or however wide the data extends) to see the layout: where series codes live, where year headers are, and where values start.
   - Pay special attention to the exact column that holds the series codes and the exact row that holds the year headers.
6. Note the exact column letter for series codes on `Data` and the exact row number for year headers on `Data`.

## Phase 1 – Lookup formulas (H12:L31)
Using the inspection results, write INDEX/MATCH formulas into the yellow cells.

For each cell in H12:L17 (Committed Funding block), H19:L24 (Operating Spend block), and H26:L31 (Approved Budget Base block):

```
=INDEX(Data!<data_values_range>, MATCH($D<row>, Data!<series_code_column>, 0), MATCH(<col>$10, Data!<year_header_row>, 0))
```

Concrete construction rules:
- `$D<row>` – the series code in column D of the current row on `Task`, with the column locked (`$D`).
- `<col>$10` – the year in row 10 of the current column on `Task`, with the row locked (`$10`). Use H$10, I$10, etc.
- `Data!<data_values_range>` – the rectangular block on `Data` that contains all numeric values (rows 21–38 × the value columns). This must cover ALL rows that any series code could appear in.
- `Data!<series_code_column>` – the single column on `Data` holding series codes, spanning the same rows as the data range (rows 21–38).
- `Data!<year_header_row>` – the single row on `Data` holding year headers, spanning the same columns as the data range.

**Critical**: Verify from the Phase 0 output that:
- The series codes in column D of `Task` rows 12–17, 19–24, 26–31 match exactly (string content, no extra spaces) with values in the `Data` sheet series code column.
- The years in `Task` row 10 match exactly (type: number vs string) with the `Data` sheet year header row.
- If there's a mismatch risk, wrap the MATCH arguments with `TRIM()` or use explicit type coercion, but only if inspection reveals a problem.

Use openpyxl to write each formula as a string (e.g., `ws['H12'] = '=INDEX(...)'`).

## Phase 2 – Net Budget Buffer (H35:L40)
For each cell in H35:L40, write a formula:
```
=(<Committed_Funding_cell> - <Operating_Spend_cell>) / <Approved_Budget_Base_cell> * 100
```
Where:
- Committed Funding is in H12:L17
- Operating Spend is in H19:L24
- Approved Budget Base is in H26:L31

So H35 = `=(H12-H19)/H26*100`, H36 = `=(H13-H20)/H27*100`, etc., through L40 = `=(L17-L24)/L31*100`.

## Phase 3 – Summary statistics (H42:L47)
For each column H through L, in rows 42–47:
- Row 42 (MIN): `=MIN(H35:H40)` (adjust column)
- Row 43 (MAX): `=MAX(H35:H40)`
- Row 44 (MEDIAN): `=MEDIAN(H35:H40)`
- Row 45 (MEAN): `=AVERAGE(H35:H40)`
- Row 46 (25th pctl): `=PERCENTILE(H35:H40,0.25)`
- Row 47 (75th pctl): `=PERCENTILE(H35:H40,0.75)`

**Important**: Check the Phase 0 output to confirm which row is which statistic. The labels in column D/E for rows 42–47 will tell you the correct order. Adjust accordingly.

## Phase 4 – Weighted mean (H50:L50)
For each column H through L:
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```
This computes the weighted mean of the Net Budget Buffer percentages using Approved Budget Base as weights.

## Phase 5 – Save and validate
1. Save the workbook to `/root/output/result.xlsx`.
2. Reopen the saved file (without `data_only`) and print the formulas in a sample of cells (e.g., H12, L17, H35, L40, H42, H47, H50, L50) to confirm they are correctly written formula strings.
3. Confirm no new sheets were added.
4. Confirm the file exists at `/root/output/result.xlsx`.

## Important constraints
- Do NOT use `data_only=True` when opening the workbook.
- Do NOT add sheets, macros, VBA, external links, or helper tabs.
- Do NOT modify existing formatting, styles, or non-yellow cells.
- Write formulas as strings so Excel can evaluate them; do NOT compute values in Python and write numbers.

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