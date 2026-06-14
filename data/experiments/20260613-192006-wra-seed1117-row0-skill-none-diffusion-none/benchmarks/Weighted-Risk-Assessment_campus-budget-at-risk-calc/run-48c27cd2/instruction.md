# Task Instruction

Execute the following steps precisely to complete the task.

## 0. Setup
```bash
mkdir -p /root/output
pip install openpyxl
```

## 1. Inspect the workbook structure
Open `/root/data/workbook.xlsx` with openpyxl (with `data_only=False` to preserve formulas). Inspect:
- Sheet `Task`: Print the contents of rows 10-50, columns D through L, paying special attention to:
  - Row 10 (years in H10:L10)
  - Column D rows 12-17, 19-24, 26-31 (series codes)
  - The yellow cells H12:L17, H19:L24, H26:L31 (should be empty or placeholder)
  - Rows 35-40 (department names/labels), row 34 label
  - Rows 42-47 (stat labels: min, max, median, mean, 25th pctl, 75th pctl)
  - Row 50 (Campus Budget Council weighted mean)
- Sheet `Data`: Print rows 21-38 completely to understand the data layout (which row has headers, where series codes are, where years are, orientation of data).

Print all cell values clearly so we understand the exact layout before writing any formulas.

## 2. Understand the Data sheet layout
From the Data sheet rows 21-38, determine:
- Whether data is arranged with series codes in a column and years across a row (or vice versa)
- Which column contains the series codes
- Which row contains the year headers
- The exact range of data values

This determines which lookup pattern to use.

## 3. Populate H12:L17, H19:L24, H26:L31 with lookup formulas
For each yellow cell in these ranges, write a formula that:
- Takes the series code from column D of that row (e.g., `$D12` for row 12)
- Takes the year from row 10 of that column (e.g., `H$10` for column H)
- Looks up the value from `Data!` rows 21:38

Use `INDEX/MATCH` pattern. The exact formula depends on the Data layout discovered in step 2. For example, if Data has series codes in column A and years in a header row:
```
=INDEX(Data!<data_range>, MATCH($D12, Data!<series_code_column>, 0), MATCH(H$10, Data!<year_header_row>, 0))
```

Adapt the ranges based on actual inspection. Use absolute references for the data range and lookup arrays, and mixed references ($D12 for row-lock on column, H$10 for column-lock on row) so formulas can fill across the block.

Write these formulas using openpyxl by setting each cell's `.value` to the formula string. Do NOT use `data_only=True`. Iterate over all 6 rows × 5 columns in each of the three blocks (90 cells total).

## 4. Populate H35:L40 with Net Budget Buffer formulas
The formula is: `(Committed Funding - Operating Spend) / Approved Budget Base * 100`

Based on the Task sheet layout:
- H12:L17 likely corresponds to one metric (e.g., Committed Funding)
- H19:L24 likely corresponds to another metric (e.g., Operating Spend)
- H26:L31 likely corresponds to Approved Budget Base

Verify which block is which by checking labels near rows 11, 18, 25 on the Task sheet. Then for each cell in H35:L40, write:
```
=(H12-H19)/H26*100
```
(adjusted for the correct row offsets matching the same department and year)

The six departments in rows 35-40 should correspond to the six departments in rows 12-17 (and 19-24, 26-31). Verify the department order matches.

## 5. Populate H42:L47 with summary statistics
For each column H through L, calculate over the 6 values in rows 35-40:
- Row 42: `=MIN(H35:H40)` (or whichever row is minimum - check the labels)
- Row 43: `=MAX(H35:H40)`
- Row 44: `=MEDIAN(H35:H40)`
- Row 45: `=AVERAGE(H35:H40)`
- Row 46: `=PERCENTILE(H35:H40,0.25)`
- Row 47: `=PERCENTILE(H35:H40,0.75)`

Match the exact stat to the exact row based on the labels in column D (or wherever labels are) for rows 42-47. Read these labels first.

## 6. Populate H50:L50 with SUMPRODUCT weighted mean
For each column (H through L):
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```
This computes the weighted mean of the Net Budget Buffer percentages (H35:H40) weighted by Approved Budget Base (H26:H31).

## 7. Save and validate
- Save the workbook to `/root/output/result.xlsx` using openpyxl's `save()` method.
- Re-open the saved file and verify:
  - Formulas exist in all target cells (spot-check a few from each block)
  - No extra sheets were added
  - The file is valid and openable

## Critical constraints
- Do NOT use `data_only=True` when loading (this strips formulas)
- Do NOT add new sheets, macros, VBA, or external links
- Do NOT modify any existing formatting, values, or structure outside the target cells
- All formulas must be Excel-compatible spreadsheet formulas (not Python calculations)
- Use openpyxl throughout
- Read actual cell contents before writing formulas to ensure correct references
- After writing, re-read a sample of cells to confirm formulas were written correctly

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