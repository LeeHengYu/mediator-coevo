# Task Instruction

Execute the following steps precisely to complete the task.

## 0. Setup
```bash
mkdir -p /root/output
pip install openpyxl
```

## 1. Inspect the workbook structure
- Open `/root/data/workbook.xlsx` with openpyxl (keep formulas via `data_only=False`).
- Print the sheet names to confirm `Task` and `Data` exist.
- Print the contents of sheet `Task` rows 10-50, columns D through L, to understand the layout: what's in D12:D17 (series codes for block 1), D19:D24 (block 2), D26:D31 (block 3), row 10 (years in H10:L10), and the labels in rows 35-50.
- Print sheet `Data` rows 21-38 to understand the lookup source: identify which column has the series codes and which row has the years, and the data layout (is it organized with series codes in a column and years across columns, or vice versa?).
- Print all of this BEFORE writing any formulas. Understanding the exact layout is critical.

## 2. Determine the lookup approach
Based on the inspection:
- Identify the exact column in `Data` that contains the series codes (let's call it the key column).
- Identify the exact row in `Data` that contains the year headers.
- Determine the data range boundaries.
- Choose INDEX/MATCH as the lookup pattern since it's the most flexible.

The formula pattern for each cell in H12:L17, H19:L24, H26:L31 should be something like:
```
=INDEX(Data!<data_range>, MATCH($D12, Data!<series_code_column>, 0), MATCH(H$10, Data!<year_header_row>, 0))
```
Adjust the references based on what you find in the inspection. The `$D12` uses a mixed reference (column absolute, row relative) and `H$10` uses (column relative, row absolute) so the formula can be filled across the grid.

## 3. Write the formulas using openpyxl
Use openpyxl to write formula strings into cells. Important notes:
- openpyxl stores formulas as strings starting with `=`.
- Use the `Translator` class or manually construct each formula with correct references.
- Do NOT use Python-side computation; write actual Excel formulas into the cells.
- Make sure references to the Data sheet use the syntax `Data!` prefix.
- Preserve existing formatting: when writing to a cell, only set `.value`, do not touch `.font`, `.fill`, `.border`, `.alignment`, `.number_format` etc.

### Step 1 formulas (H12:L17, H19:L24, H26:L31)
Write INDEX/MATCH formulas into each of the 90 cells (6 rows × 5 cols × 3 blocks).

### Step 2a: Net patient flow (H35:L40)
Based on the layout, the three blocks likely correspond to:
- H12:L17 = Patient Admissions (or similar)
- H19:L24 = Patient Discharges (or similar)
- H26:L31 = Effective Bed Capacity

Verify this from the labels in column D or nearby. The formula for each cell in H35:L40 is:
```
=(H12 - H19) / H26 * 100
```
(Adjust row references for each of the 6 hospitals, maintaining the correspondence: row 35 uses rows 12, 19, 26; row 36 uses rows 13, 20, 27; etc.)

### Step 2b: Summary statistics (H42:L47)
For each column (H through L), in the 6 rows 42-47:
- Row 42 (MIN): `=MIN(H35:H40)`
- Row 43 (MAX): `=MAX(H35:H40)`
- Row 44 (MEDIAN): `=MEDIAN(H35:H40)`
- Row 45 (MEAN): `=AVERAGE(H35:H40)`
- Row 46 (25th percentile): `=PERCENTILE(H35:H40, 0.25)` or `=PERCENTILE.INC(H35:H40, 0.25)`
- Row 47 (75th percentile): `=PERCENTILE(H35:H40, 0.75)` or `=PERCENTILE.INC(H35:H40, 0.75)`

Verify the exact row-to-statistic mapping from the labels in column D or nearby before writing.

### Step 3: Weighted mean (H50:L50)
For each column:
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```
This computes the weighted mean of Net patient flow percentages using Effective Bed Capacity as weights.

## 4. Verify the order of statistics rows
Before writing Step 2b formulas, print the labels in column D (or whatever label column) for rows 42-47 to confirm which row is min, max, median, mean, 25th, 75th percentile. Map formulas accordingly.

## 5. Save and validate
- Save the workbook to `/root/output/result.xlsx`.
- Reopen the saved file and print the formula strings in a sample of cells (e.g., H12, L17, H35, H40, H42, H47, H50) to confirm they are correct Excel formulas.
- Verify no new sheets were added.
- Verify the file is a valid xlsx by reopening it without errors.

## Critical constraints
- Only modify cell `.value` properties. Do not alter formatting, styles, or structure.
- Do not add sheets, macros, VBA, external links, or helper tabs.
- All formulas must be real Excel formulas (strings starting with `=`), not computed Python values.
- The lookup formulas must use INDEX with MATCH (or one of the other allowed patterns: VLOOKUP+MATCH, HLOOKUP+MATCH, XLOOKUP+MATCH).
- The weighted mean must use SUMPRODUCT.
- Inspect everything before writing. If the layout differs from assumptions (e.g., the blocks map to different metrics, or the statistics rows are in a different order), adapt accordingly.

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