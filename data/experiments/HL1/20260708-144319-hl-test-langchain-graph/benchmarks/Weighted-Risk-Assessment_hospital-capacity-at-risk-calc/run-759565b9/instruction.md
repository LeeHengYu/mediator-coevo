# Task Instruction

Execute the following steps to complete the task.

## 0. Inspect the workbook
1. Copy the workbook: `cp /root/data/workbook.xlsx /root/output/result.xlsx`
2. Open `/root/output/result.xlsx` with openpyxl and inspect:
   - Sheet names (confirm `Task` and `Data` exist).
   - On `Task`: read row 10 to find the year headers in columns H–L. Read column D rows 12–17, 19–24, 26–31 to find the series codes. Read row 35–40 labels, row 42–47 labels (min/max/median/mean/25th/75th), and row 50 label.
   - On `Data`: read rows 21–38 to understand the data layout (which row holds which series code, which columns hold which years). Determine whether data is arranged so that series codes are in a column and years are in a row header, or vice versa. Identify the exact column that holds the series code and the exact row that holds the year headers.
   - Print all of this so we know the exact cell references before writing any formulas.

## 1. Write lookup formulas in H12:L17, H19:L24, H26:L31
Based on the inspection, write `INDEX(MATCH,MATCH)` formulas into every yellow cell in those three blocks. Each formula should:
- Use the series code from column D of the current row (e.g., `$D12` with the column locked).
- Use the year from row 10 of the current column (e.g., `H$10` with the row locked).
- Reference the `Data` sheet rows 21:38 for the data array, the series-code column for the row lookup, and the year header row for the column lookup.
- Use exact match (0) in both MATCH functions.
- Example pattern: `=INDEX(Data!<data_range>, MATCH($D12, Data!<series_code_column>, 0), MATCH(H$10, Data!<year_header_row>, 0))`

Adjust the exact ranges based on what you found in step 0. Make sure references lock rows/columns appropriately so the formula can be dragged across H–L and down each block.

Use openpyxl to write these as string formulas (e.g., `ws['H12'] = '=INDEX(...)'`). Do NOT use `data_only` mode.

## 2. Write Net Capacity Headroom formulas in H35:L40
For each of the six hospital clusters (rows 35–40), the formula is:
`=(H12 - H19) / H26 * 100`
(adjusted for the actual row: row 35 uses rows 12, 19, 26; row 36 uses rows 13, 20, 27; etc.)

So for cell H35: `=(H12-H19)/H26*100`, for H36: `=(H13-H20)/H27*100`, etc. Columns H–L, rows 35–40.

Lock nothing extra here since each cell is specific.

## 3. Write summary statistics in H42:L47
For each column (H through L), write:
- Row 42 (MIN): `=MIN(H35:H40)`
- Row 43 (MAX): `=MAX(H35:H40)`
- Row 44 (MEDIAN): `=MEDIAN(H35:H40)`
- Row 45 (AVERAGE): `=AVERAGE(H35:H40)`
- Row 46 (25th percentile): `=PERCENTILE(H35:H40,0.25)`
- Row 47 (75th percentile): `=PERCENTILE(H35:H40,0.75)`

Verify the exact row-to-statistic mapping by reading the labels in column D/G of rows 42–47 during inspection.

## 4. Write weighted mean in H50:L50
For each column (H through L):
`=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)`

This computes the weighted mean of the Net Capacity Headroom percentages weighted by Staffed Bed Capacity.

## 5. Save and validate
- Save the workbook with `wb.save('/root/output/result.xlsx')`.
- Reopen the file and confirm:
  - No new sheets were added.
  - Cells H12, H19, H26, H35, H42, H50 all contain formula strings (not None, not bare values).
  - Print a sample of formulas to verify correctness.

## Important constraints
- Do NOT use `data_only=True` when opening for writing.
- Do NOT add sheets, macros, VBA, external links, or helper tabs.
- Do NOT change any existing formatting (fonts, fills, borders, number formats).
- Write formulas as strings starting with `=`.
- If openpyxl strips or mishandles any formula, try writing with the `Translator` or direct string assignment.
- The final saved file must be at `/root/output/result.xlsx`.

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