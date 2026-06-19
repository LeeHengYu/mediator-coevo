# Task Instruction

Execute the following steps to complete the task.

## 0. Setup
```bash
mkdir -p /root/output
pip install openpyxl
```

## 1. Inspect the workbook structure
Open `/root/data/workbook.xlsx` with openpyxl and inspect:
- Sheet names (confirm `Task` and `Data` exist)
- On `Task` sheet: read rows 10-50, columns D-L to understand the layout:
  - Row 10: header row with years in H10:L10
  - Column D rows 12-17, 19-24, 26-31: series codes for three blocks
  - H12:L17, H19:L24, H26:L31: yellow cells to fill with lookup formulas
  - H35:L40: Net reliability gap calculation area; check what rows 35-40 contain (region labels, etc.)
  - H42:L47: summary statistics area (min, max, median, mean, 25th pctl, 75th pctl)
  - H50:L50: weighted mean for GCM
- On `Data` sheet: inspect rows 21-38 to understand the data layout (which row has headers, how series codes and years are arranged). Specifically determine:
  - Is the data arranged with series codes in a column and years across columns (suitable for VLOOKUP/HLOOKUP)?
  - Which column contains the series codes?
  - Which row contains the year headers?
  - The exact range needed for lookups

Print all of this information clearly before proceeding.

## 2. Write lookup formulas in H12:L17, H19:L24, H26:L31

Using openpyxl, write Excel formulas (as strings starting with `=`) into each cell. The formulas must use one of the allowed patterns: INDEX/MATCH, VLOOKUP/MATCH, HLOOKUP/MATCH, or XLOOKUP/MATCH.

For each cell at row `r`, column `c` (where c maps to H=8, I=9, J=10, K=11, L=12):
- The series code is in cell `$D{r}` (column D of the same row)
- The year is in the header row 10 at column `c`, i.e., cell like `H$10`, `I$10`, etc.
- The lookup range is on `Data` sheet rows 21:38

Based on the data layout discovered in step 1, construct the appropriate formula. For example, if using INDEX/MATCH/MATCH:
```
=INDEX(Data!<data_range>, MATCH($D{r}, Data!<series_code_column>, 0), MATCH({col}$10, Data!<year_header_row>, 0))
```

Make sure to:
- Use absolute references where appropriate ($D for column lock, $10 for row lock) so formulas are consistent
- Reference the exact ranges from the Data sheet
- Verify the formula pattern works by checking that the ranges are correct

IMPORTANT: When writing formulas with openpyxl, set `cell.value = '=FORMULA...'` as a string. Do NOT set `cell.data_type` manually; openpyxl handles formula cells automatically.

## 3. Write Net reliability gap formulas in H35:L40

The formula for each cell is:
`= (Successful API Requests - Failed API Requests) / Compute Capacity * 100`

Based on the three blocks:
- H12:L17 = first indicator block (check which one is Successful API Requests)
- H19:L24 = second indicator block (check which one is Failed API Requests)
- H26:L31 = third indicator block (check which one is Compute Capacity)

Determine which block corresponds to which metric by reading the labels. The six regions in rows 35-40 should correspond to the six rows in each block (rows 12-17, 19-24, 26-31).

For each cell at row `r_out` in 35-40 and column `c`:
```
= (H12_block_cell - H19_block_cell) / H26_block_cell * 100
```
where the row offsets map: row 35→rows 12,19,26; row 36→rows 13,20,27; etc.

## 4. Write summary statistics in H42:L47

For each column c (H through L), write these formulas referencing the Net reliability gap range (e.g., H35:H40 for column H):
- Row 42 (minimum): `=MIN(H35:H40)`
- Row 43 (maximum): `=MAX(H35:H40)`
- Row 44 (median): `=MEDIAN(H35:H40)`
- Row 45 (mean): `=AVERAGE(H35:H40)`
- Row 46 (25th percentile): `=PERCENTILE(H35:H40, 0.25)`
- Row 47 (75th percentile): `=PERCENTILE(H35:H40, 0.75)`

Check the row labels on the Task sheet to confirm the exact order of these statistics.

## 5. Write weighted mean in H50:L50

For each column c (H through L):
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```
(using the appropriate column letter for each)

This computes the weighted mean of Net reliability gap values weighted by Compute Capacity.

## 6. Save and verify

Save the workbook to `/root/output/result.xlsx`.

After saving, re-open the file with openpyxl and verify:
- All formula cells in the target ranges contain formula strings (start with `=`)
- No cells in those ranges are None or empty
- Sheet names are still only `Task` and `Data`
- Print a sample of formulas from each section for confirmation

IMPORTANT NOTES:
- Do NOT create any new sheets
- Do NOT delete any existing content outside the target ranges
- Preserve all existing formatting (do not modify fonts, fills, borders, etc.)
- When loading the workbook, do NOT use `data_only=True` (we need to preserve formulas)
- Use `keep_vba=False` (default) and do not add macros
- Read the actual cell labels and data layout BEFORE writing any formulas — do not assume the order of metrics or statistics

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