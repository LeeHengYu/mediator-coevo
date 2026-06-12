# Task Instruction

Execute the following steps exactly in order.

## 0 – Inspect the workbook structure

Open `/root/data/workbook.xlsx` with openpyxl (data_only=False). For both sheets `Task` and `Data`, print:

* Every cell value in rows 1–50 for columns A–L on sheet `Task` (print row number + list of cell values).
* Every cell value in rows 1–40 for columns A–Z on sheet `Data` (print row number + list of cell values).

Pay special attention to:
- Column D on `Task` (series codes for each service row)
- Row 10 on `Task` (year headers in H10:L10)
- Rows 21–38 on `Data` (the source data block): identify the layout — which row holds what series code, which column holds what year, where the header row is, etc.
- Rows 12–17, 19–24, 26–31, 35–40, 42–47, 50 on `Task` to understand the existing structure.

Do NOT proceed to formula writing until you have printed and understood all of this.

## 1 – Determine the lookup geometry

From the inspection:
1. Identify the Data sheet lookup range for rows 21:38. Determine the top-left cell, the column that holds series codes, and the row that holds year headers.
2. For each yellow-cell block on `Task` (H12:L17, H19:L24, H26:L31), confirm which series code is in column D and which year is in row 10.
3. Decide on the INDEX/MATCH pattern:
   - `=INDEX(Data!<data_range>, MATCH(<series_code>, Data!<series_col>, 0), MATCH(<year>, Data!<year_row>, 0))`
   - Adjust the data range, series column, and year row references based on the actual layout you observed.

Print the exact formula you plan to write into cell H12 as a sanity check before writing any formulas.

## 2 – Write lookup formulas (Step 1)

Using openpyxl, write INDEX/MATCH formulas into every cell in H12:L17, H19:L24, and H26:L31. Each formula should reference:
- The series code from column D of the current row (e.g., `Task!D12`)
- The year from row 10 of the current column (e.g., `Task!H10`)
- The Data sheet source range (rows 21:38)

Use absolute references for the Data range and the series-code column and year-header row within it. Use relative or mixed references for the current row's series code and the current column's year so the formula is correct per-cell.

IMPORTANT: Write the formula as a string starting with `=`. Do NOT use data_only mode.

## 3 – Write Net SLA Buffer formulas (Step 2, rows 35–40)

For each cell in H35:L40, write:
`= (<corresponding Latency Budget Preserved cell> - <corresponding Latency Budget Consumed cell>) / <corresponding Covered Request Capacity cell> * 100`

The three blocks are:
- Latency Budget Preserved: H12:L17
- Latency Budget Consumed: H19:L24
- Covered Request Capacity: H26:L31

So H35 = (H12 - H19) / H26 * 100, H36 = (H13 - H20) / H27 * 100, etc. Verify the row mapping by checking that the same service appears in the same relative position in each block.

## 4 – Write summary statistics (Step 2, rows 42–47)

For each column H through L:
- Row 42: `=MIN(Hxx:Hxx)` over the 6 Net SLA buffer cells (e.g., H35:H40)
- Row 43: `=MAX(H35:H40)`
- Row 44: `=MEDIAN(H35:H40)`
- Row 45: `=AVERAGE(H35:H40)`
- Row 46: `=PERCENTILE(H35:H40,0.25)` — use legacy `PERCENTILE`, NOT `PERCENTILE.INC`
- Row 47: `=PERCENTILE(H35:H40,0.75)` — use legacy `PERCENTILE`, NOT `PERCENTILE.INC`

CRITICAL: Previous execution failed because `PERCENTILE.INC` produced `#NAME?` errors. Use the legacy function name `PERCENTILE` instead.

Verify the row labels (column A or nearby) match MIN/MAX/MEDIAN/AVERAGE/25th/75th before writing.

## 5 – Write weighted mean (Step 3, row 50)

For each column H through L:
`=SUMPRODUCT(<Net SLA buffer column, e.g. H35:H40>, <Covered Request Capacity column, e.g. H26:H31>) / SUM(<Covered Request Capacity column, e.g. H26:H31>)`

## 6 – Save

Save the workbook to `/root/output/result.xlsx`. Create the `/root/output/` directory if it doesn't exist.

## 7 – Validate

Reopen `/root/output/result.xlsx` with openpyxl (data_only=False). Print the formula content of:
- H12, L17 (first and last lookup cells)
- H35, L40 (first and last net buffer cells)
- H42, H46, H47 (MIN and both PERCENTILE cells)
- H50, L50 (weighted mean cells)

Confirm none of them are None or empty. Confirm PERCENTILE cells use `PERCENTILE` (not `PERCENTILE.INC`).

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