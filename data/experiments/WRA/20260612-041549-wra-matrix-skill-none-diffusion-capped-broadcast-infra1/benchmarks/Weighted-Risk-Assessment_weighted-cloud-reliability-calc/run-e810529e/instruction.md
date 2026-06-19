# Task Instruction

Execute the following steps in order.

## Phase 0 – Environment setup
```bash
pip install openpyxl
mkdir -p /root/output
```

## Phase 1 – Inspect the workbook structure
Open `/root/data/workbook.xlsx` with openpyxl (data_only=False) and print:
1. Sheet names.
2. Sheet `Task`: values in column D rows 12-17, 19-24, 26-31 (series codes), row 10 columns H-L (years), rows 35-40 column D or E (region labels for Net reliability gap), rows 42-47 column D-G (stat labels), row 50 columns D-G (weighted mean label).
3. Sheet `Data`: the header row and a sample of rows 21-38 to understand the layout — print row 21 through 38 for columns A-Z (or however far data extends). Identify which column holds the series code, which row holds years, and where the numeric data lives.
4. Print the current content of cells H12, H19, H26, H35, H42, H50 to confirm they are empty/yellow placeholders.

Record all findings before proceeding.

## Phase 2 – Determine lookup geometry
From the Phase 1 output, determine:
- On `Data` sheet: which column contains the series codes (let's call it `code_col`), which row contains the year headers (let's call it `year_row`), and the data body rows 21-38.
- The col_num offset needed for INDEX/MATCH: the number of columns from the code column to the first data column.
- Confirm the years in `Data` year_row match the years in `Task` row 10.

## Phase 3 – Write lookup formulas (Step 1)
Using openpyxl, write INDEX/MATCH formulas into the yellow cells.

For each cell in `H12:L17`, `H19:L24`, `H26:L31`:
- The formula pattern should be:
  `=INDEX(Data!<data_range>, MATCH($D{row}, Data!$<code_col>$21:$<code_col>$38, 0), MATCH(<Task>!<year_cell>, Data!<year_range>, 0))`
- Use absolute references for the Data lookup ranges and mixed references so the formula can be written per-cell.
- Make sure `$D{row}` uses the actual row number on the Task sheet for each cell.
- Make sure the year reference points to the correct cell in row 10 of Task sheet (H10, I10, J10, K10, L10).

After writing, re-read a sample cell (e.g., H12) to confirm the formula string is stored correctly.

## Phase 4 – Write Net reliability gap formulas (Step 2, rows 35-40)
The six regions in rows 35-40 correspond to the same six regions in the three blocks above. Determine which rows in blocks 12-17, 19-24, 26-31 correspond to each region in rows 35-40 by matching region/label text.

For each cell in `H35:L40`, write:
`=(<Successful_API_cell> - <Failed_API_cell>) / <Compute_Capacity_cell> * 100`

where:
- Successful API Requests block = rows 12-17 (or 19-24 — confirm from series code names)
- Failed API Requests block = the other block
- Compute Capacity block = rows 26-31

Carefully identify which block is which by reading the series codes or block headers from Phase 1.

## Phase 5 – Write summary statistics (Step 2, rows 42-47)
For each column H through L, in rows 42-47, write formulas for:
- Row 42 (minimum): `=MIN(<col>35:<col>40)`
- Row 43 (maximum): `=MAX(<col>35:<col>40)`
- Row 44 (median): `=MEDIAN(<col>35:<col>40)`
- Row 45 (mean): `=AVERAGE(<col>35:<col>40)`
- Row 46 (25th percentile): `=PERCENTILE(<col>35:<col>40, 0.25)`
- Row 47 (75th percentile): `=PERCENTILE(<col>35:<col>40, 0.75)`

Match the stat labels from Phase 1 to the correct rows — do NOT assume the order above; read the actual labels in column D/E/F/G rows 42-47 and assign accordingly.

## Phase 6 – Write weighted mean (Step 3, row 50)
For each column H through L, write in row 50:
`=SUMPRODUCT(<col>35:<col>40, <col>26:<col>31) / SUM(<col>26:<col>31)`

This computes the weighted mean of the Net reliability gap percentages using Compute Capacity as weights.

## Phase 7 – Save and validate
1. Save the workbook to `/root/output/result.xlsx`.
2. Re-open the saved file with openpyxl (data_only=False) and verify:
   - Cells H12, L17, H19, L24, H26, L31 contain formula strings (start with '=').
   - Cells H35, L40 contain formula strings.
   - Cells H42, L47 contain formula strings.
   - Cell H50 and L50 contain formula strings.
3. Print a summary of all verified cells.
4. If any cell is empty or None, diagnose and fix before finishing.

## Important constraints
- Do NOT add new sheets, macros, VBA, external links, or helper tabs.
- Do NOT alter existing formatting (fonts, fills, borders, etc.).
- Do NOT use data_only=True when writing; use it only if you need to check computed values separately.
- Use `keep_vba=False` (default) since no VBA is present.
- When opening the workbook for writing, do NOT discard existing defined names or print settings.

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
Task metadata: author_email=catpaw@meituan.com, author_name=CatPaw Task Engineer, category=spreadsheet-formula-reuse, difficulty=easy, tags=[excel, formulas, lookup, statistics, weighted-mean].
Verifier config: timeout_sec=600.0.