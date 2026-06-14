# Task Instruction

Execute the following steps to produce `/root/output/result.xlsx`:

## 1. Inspect the workbook
```bash
mkdir -p /root/output
```
Open `/root/data/workbook.xlsx` with openpyxl (data_only=False) and inspect:
- Sheet `Task`: read the series codes in column D for rows 12-17, 19-24, 26-31. Read the year headers in H10:L10. Read any existing labels in rows 35-40, 42-47, 50. Note exact cell contents and formatting.
- Sheet `Data`: read rows 21-38 to understand the data layout (column headers, row keys, structure). Identify which column holds the series code and which row/column holds the year values.

Print all of these so you understand the exact structure before writing any formulas.

## 2. Populate lookup formulas in H12:L17, H19:L24, H26:L31

For each cell in these three blocks, write an Excel formula using `INDEX` with `MATCH`. The formula pattern should be:
```
=INDEX(Data!<data_range>, MATCH(<series_code_cell>, Data!<series_code_column>, 0), MATCH(<year_cell>, Data!<year_row>, 0))
```
where:
- `<series_code_cell>` is the cell in column D of the current row on sheet `Task` (e.g., `$D12` for row 12). Use `$D12` (absolute column, relative row) so it stays fixed when copied across columns.
- `<year_cell>` is the cell in row 10 of the current column (e.g., `H$10`). Use `H$10` (relative column, absolute row).
- `<data_range>`, `<series_code_column>`, and `<year_row>` must be determined from your inspection of the `Data` sheet rows 21-38.

IMPORTANT: Make sure the ranges in the INDEX/MATCH formula correctly reference the Data sheet. The data range should cover the full block of values (excluding headers), the series code column should be the leftmost column of that data block, and the year row should be the header row of that data block. Verify by checking that the MATCH for series codes searches down a column and the MATCH for years searches across a row.

Use `Translate=False` when writing formulas with openpyxl to prevent locale translation issues.

## 3. Calculate Net SLA buffer in H35:L40

The formula for each cell is:
```
=(Latency_Budget_Preserved - Latency_Budget_Consumed) / Covered_Request_Capacity * 100
```
Identify which of the three blocks (H12:L17, H19:L24, H26:L31) corresponds to each metric by checking the labels. Write Excel formulas referencing the appropriate cells. For example, if row 12 and row 35 correspond to the same service:
```
=(H12 - H19) / H26 * 100
```
Adjust row references based on the actual mapping between blocks and the formula components.

## 4. Calculate summary statistics in H42:L47

For each column H through L, compute these six statistics over the 6 Net SLA buffer values (rows 35-40):
- Row 42: `=MIN(H35:H40)`
- Row 43: `=MAX(H35:H40)`
- Row 44: `=MEDIAN(H35:H40)`
- Row 45: `=AVERAGE(H35:H40)`
- Row 46: `=PERCENTILE(H35:H40, 0.25)`
- Row 47: `=PERCENTILE(H35:H40, 0.75)`

Check the labels in column D/E/F/G for rows 42-47 to confirm the correct order (min, max, median, mean, 25th, 75th). Adjust the row assignments if the labels indicate a different order.

## 5. Calculate weighted mean in H50:L50

For each column, use SUMPRODUCT with the Net SLA buffer percentages (H35:H40) as values and the Covered Request Capacity block (H26:H31) as weights:
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```

## 6. Save

Save the workbook to `/root/output/result.xlsx`. Do NOT add any new sheets, macros, VBA, or external links. Preserve all existing formatting.

## 7. Verify

Reopen the saved file and spot-check:
- That cells in H12:L17, H19:L24, H26:L31 contain formula strings (not None, not raw values).
- That cells in H35:L40 contain formulas.
- That cells in H42:L47 contain formulas.
- That cells in H50:L50 contain formulas.
- Print a few formula strings to confirm correctness.

If any test file exists (e.g., `test_output.py` or `test_outputs.py`), run it with pytest to validate.

## Critical Notes
- The previous failed run on a similar task had cells returning None — this means formulas were never written. Double-check that every target cell gets a formula assigned.
- Use `ws['H12'] = '=INDEX(...)'` syntax (string starting with `=`) to ensure openpyxl treats it as a formula.
- Do NOT use `data_only=True` when loading — that strips formulas.
- Inspect before writing. Print the Data sheet structure so you get the ranges right.

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