# Task Instruction

Execute the following steps precisely to complete the weighted campus energy balance workbook task.

## Pre-work: Inspect the workbook

1. Copy the workbook: `cp /root/data/workbook.xlsx /root/output/result.xlsx`
2. Use `openpyxl` to open `/root/output/result.xlsx` and inspect:
   - Sheet names (confirm `Task` and `Data` exist).
   - On sheet `Task`: read row 10 (the year headers in columns H–L), column D rows 12–17, 19–24, 26–31 to see the series codes, rows 35–40 labels, rows 42–47 labels, row 50 label.
   - On sheet `Data`: read rows 21–38 to understand the layout — identify which column holds series codes, which row holds years, and where the data values are. Print the first few columns and rows to understand orientation (is data organized with series codes in a column and years across rows, or vice versa?).
   - Print all findings clearly before proceeding.

## Step 1: Populate H12:L17, H19:L24, H26:L31 with lookup formulas

Based on the inspection, write formulas into the yellow cells. Each formula must use two inputs:
- The series code from column D of the same row (e.g., `$D12` for row 12)
- The year from row 10 of the same column (e.g., `H$10` for column H)

The lookup must reference sheet `Data` rows 21:38. Use an INDEX/MATCH pattern like:
```
=INDEX(Data!$B$21:$XX$38, MATCH($D12, Data!$A$21:$A$38, 0), MATCH(H$10, Data!$B$20:$XX$20, 0))
```
Adjust the exact column/row references based on what you find during inspection. The key requirements:
- The series code column on Data sheet must be identified (likely column A or B).
- The year header row on Data sheet must be identified (likely row 20 or 21).
- The data range must span rows 21:38 and the appropriate columns.
- Use absolute references for the Data ranges and mixed references ($D12 for series code column, H$10 for year row) so formulas can be filled across the grid.

Write the formula into every cell in H12:L17, H19:L24, and H26:L31 (that's 3 blocks × 6 rows × 5 columns = 90 cells).

## Step 2: Net renewable balance in H35:L40 and statistics in H42:L47

For H35:L40, the formula is:
```
=(H12 - H19) / H26 * 100
```
where row 12 corresponds to Renewable Generation, row 19 to Grid Consumption, and row 26 to Baseline Energy Demand for the same campus. Match the campus ordering: row 35 uses rows 12, 19, 26; row 36 uses rows 13, 20, 27; etc.

For H42:L47, calculate column-wise statistics over H35:L40:
- Row 42 (minimum): `=MIN(H35:H40)` (or whichever label matches MIN)
- Row 43 (maximum): `=MAX(H35:H40)`
- Row 44 (median): `=MEDIAN(H35:H40)`
- Row 45 (simple mean): `=AVERAGE(H35:H40)`
- Row 46 (25th percentile): `=PERCENTILE(H35:H40, 0.25)`
- Row 47 (75th percentile): `=PERCENTILE(H35:H40, 0.75)`

**Important**: Read the actual labels in column A/B/C for rows 42–47 to determine which statistic goes in which row. Map them correctly — do not assume the order above.

## Step 3: Weighted mean in H50:L50

For each column H through L:
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```
This computes the weighted mean of the Net renewable balance percentages using Baseline Energy Demand as weights.

## Implementation details

- Use `openpyxl` with `keep_vba=False`. Open the workbook with `data_only=False` so existing formulas are preserved.
- When writing formulas, set `cell.value = '=FORMULA...'` as a string starting with `=`.
- Do NOT modify any formatting, styles, fills, fonts, borders, or number formats.
- Do NOT add or remove sheets.
- Do NOT add macros, VBA, external links, or helper tabs.
- Save to `/root/output/result.xlsx`.

## Validation

After saving, re-open the file and:
1. Confirm cells H12, L17, H19, L24, H26, L31 contain formula strings (start with `=`).
2. Confirm cells H35, L40 contain formula strings.
3. Confirm cells H42, L47 contain formula strings.
4. Confirm cells H50, L50 contain formula strings.
5. Print a sample of formulas from each block to verify correctness.

If any cell is missing a formula or contains a plain value where a formula is expected, fix it before finishing.

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