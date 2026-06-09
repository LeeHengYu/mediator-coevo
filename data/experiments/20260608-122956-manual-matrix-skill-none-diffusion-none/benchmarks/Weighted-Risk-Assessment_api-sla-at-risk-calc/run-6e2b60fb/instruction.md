# Task Instruction

Execute the following steps precisely to complete the task.

## 0. Setup
```bash
mkdir -p /root/output
cp /root/data/workbook.xlsx /root/output/result.xlsx
```

## 1. Inspect the workbook structure
Open `/root/output/result.xlsx` with openpyxl and inspect:
- Sheet names (should be `Task` and `Data`)
- On sheet `Task`: read row 10 (H10:L10) to see the years; read column D rows 12-17, 19-24, 26-31 to see the series codes; read rows 35-40 column D for the service names; read H42:L47 labels; read H50:L50 area and any labels.
- On sheet `Data`: read rows 21-38 to understand the data layout — specifically which row is the header row, what columns contain, and how series codes and years are arranged.

Print all of this information so we understand the exact layout before writing any formulas.

## 2. Populate lookup formulas in H12:L17, H19:L24, H26:L31

Using openpyxl, write spreadsheet formulas (not computed values) into each yellow cell. Each formula must use one of the allowed lookup patterns: INDEX/MATCH, VLOOKUP/MATCH, HLOOKUP/MATCH, or XLOOKUP/MATCH.

The two inputs for every lookup are:
- The series code in column D of the current row (e.g., `$D12` for row 12, with the column anchored so it doesn't shift when copied across)
- The year in row 10 of the current column (e.g., `H$10` for column H, with the row anchored)

The source data is on sheet `Data` rows 21:38. You need to determine from inspection:
- Which column in Data contains the series codes (the lookup key)
- Which row in Data contains the year headers
- The data range for the values

Use INDEX/MATCH as the primary pattern since it's most flexible. The formula pattern should be something like:
`=INDEX(Data!<value_range>, MATCH($D12, Data!<series_code_column>, 0), MATCH(H$10, Data!<year_header_row>, 0))`

Adapt the exact ranges based on what you find in the inspection step. Make sure:
- Column D reference is anchored with `$D` so it doesn't shift horizontally
- Row 10 reference is anchored with `$10` so it doesn't shift vertically
- Data sheet ranges are absolute where needed
- Fill all cells in H12:L17 (6 rows × 5 cols = 30 cells), H19:L24 (30 cells), H26:L31 (30 cells)

## 3. Calculate Net SLA buffer in H35:L40

The formula is: `(Latency Budget Preserved - Latency Budget Consumed) / Covered Request Capacity * 100`

From the inspection, determine which of the three blocks (H12:L17, H19:L24, H26:L31) corresponds to:
- Latency Budget Preserved
- Latency Budget Consumed  
- Covered Request Capacity

Then for each cell in H35:L40, write a formula like:
`=(H12-H19)/H26*100` (adjusted for the correct block mapping and row offsets)

Make sure the six rows in 35:40 correspond to the six services in the same order as they appear in the three blocks above.

## 4. Summary statistics in H42:L47

For each column H through L, write these formulas referencing the Net SLA buffer block (H35:L40 for column H, etc.):
- Row 42: `=MIN(H35:H40)` (or whichever row is labeled minimum)
- Row 43: `=MAX(H35:H40)`
- Row 44: `=MEDIAN(H35:H40)`
- Row 45: `=AVERAGE(H35:H40)`
- Row 46: `=PERCENTILE(H35:H40, 0.25)`
- Row 47: `=PERCENTILE(H35:H40, 0.75)`

**IMPORTANT**: Check the actual labels in column D/E/F/G for rows 42-47 to determine the correct order of min, max, median, mean, 25th percentile, 75th percentile. Assign formulas to match the labels.

## 5. Weighted mean in H50:L50

For each column H through L:
`=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)`

This computes the weighted mean of Net SLA buffer percentages weighted by Covered Request Capacity.

## 6. Save and verify

Save the workbook. Then reopen it and verify:
- No new sheets were added
- Formulas are present (not just values) in all the required cells
- The formula patterns use MATCH as required
- No macros, VBA, or external links exist

Print a summary of all formulas written to confirm correctness.

## Critical Notes
- Use `openpyxl` for all Excel operations
- Write STRING formulas (starting with `=`) into cells, NOT computed Python values
- Do NOT change any existing formatting — do not touch fonts, fills, borders, number formats, column widths, row heights, etc.
- Do NOT add sheets, delete sheets, or rename sheets
- When writing formulas, reference the `Data` sheet as `Data!` (or with quotes if the name has spaces)
- Inspect before writing — the exact row/column layout of the Data sheet is critical to getting the formulas right

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