# Task Instruction

Execute the following steps to produce /root/output/result.xlsx from /root/data/workbook.xlsx.

## Step 0 – Inspect the workbook
1. `mkdir -p /root/output`
2. Open `/root/data/workbook.xlsx` with openpyxl (data_only=False) and inspect:
   - Sheet names (confirm `Task` and `Data` exist).
   - On `Task`: read row 10 to find the year headers in columns H–L. Read column D rows 12–17, 19–24, 26–31 to find the series codes. Read any labels in rows 35–40 (campus names), rows 42–47 (stat labels), and row 50 (MCEC label). Note the exact text/values.
   - On `Data`: read row 20 (or the header row just above row 21) to find column headers (years or series codes). Read rows 21–38 to understand the data layout: which column holds the series code, which columns hold year values, and how they map.
   - Print all of this so you understand the exact layout before writing any formulas.

## Step 1 – Populate lookup formulas in H12:L17, H19:L24, H26:L31
For each yellow cell in those three blocks, write an `INDEX`/`MATCH`/`MATCH` formula. The formula pattern is:

```
=INDEX(Data!$B$21:$XX$38, MATCH($D{row}, Data!$A$21:$A$38, 0), MATCH(H$10, Data!$B$20:$XX$20, 0))
```

Adjust the exact column/row references based on what you discovered in Step 0:
- The first MATCH finds the series code from column D of the current row in the Data sheet's series-code column.
- The second MATCH finds the year from row 10 of the Task sheet in the Data sheet's header row.
- The INDEX range covers the data body (excluding the series-code column and header row).

Use absolute references (`$`) appropriately so that:
- The series-code reference uses `$D{row}` (column locked, row varies).
- The year reference uses `{col}$10` (row locked, column varies).

Write these as string formulas (e.g., `cell.value = '=INDEX(...)'`). Do NOT set them as Python-computed values.

## Step 2 – Net renewable balance in H35:L40
For each cell in H35:L40, write a formula:
```
=({renewable_cell} - {grid_cell}) / {baseline_cell} * 100
```
where:
- `{renewable_cell}` is the corresponding cell in the Renewable Generation block (H12:L17),
- `{grid_cell}` is the corresponding cell in the Grid Consumption block (H19:L24),
- `{baseline_cell}` is the corresponding cell in the Baseline Energy Demand block (H26:L31).

The row offset between blocks should be consistent (e.g., row 35 maps to row 12, row 19, row 26; row 36 maps to row 13, row 20, row 27; etc.). Confirm the campus ordering matches before writing.

## Step 3 – Summary statistics in H42:L47
For each column (H through L), write these formulas in rows 42–47:
- Row 42: `=MIN(H35:H40)` (minimum)
- Row 43: `=MAX(H35:H40)` (maximum)
- Row 44: `=MEDIAN(H35:H40)` (median)
- Row 45: `=AVERAGE(H35:H40)` (simple mean)
- Row 46: `=PERCENTILE(H35:H40, 0.25)` (25th percentile)
- Row 47: `=PERCENTILE(H35:H40, 0.75)` (75th percentile)

Check the labels in column D/E/F/G of rows 42–47 to confirm which row is which statistic. Adjust the row assignments if the labels differ from the order above.

## Step 4 – Weighted mean in H50:L50
For each column (H through L), write:
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```
This computes the weighted mean of the Net renewable balance percentages using Baseline Energy Demand as weights.

## Step 5 – Save
Save the workbook to `/root/output/result.xlsx`. Do NOT change any existing formatting, do not add sheets, macros, VBA, external links, or helper tabs.

## Step 6 – Validate
Reopen `/root/output/result.xlsx` with openpyxl (data_only=False) and verify:
1. Every cell in H12:L17, H19:L24, H26:L31 contains a string starting with `=` (a formula, not None or a number).
2. Every cell in H35:L40 contains a formula string starting with `=`.
3. Every cell in H42:L47 contains a formula string starting with `=`.
4. Every cell in H50:L50 contains a formula string starting with `=`.
5. Print a sample of formulas from each block for visual confirmation.

If any cell is None or a bare number, fix it before finishing.

## Key Warnings
- The previous failed run on a sibling task produced None values because formulas were not written as strings. Always assign formula strings (e.g., `ws['H12'] = '=INDEX(...)'`).
- Do NOT use data_only=True when writing formulas.
- Do NOT evaluate formulas in Python; write them as Excel formula strings.
- Carefully inspect the Data sheet layout before constructing INDEX/MATCH references. Off-by-one errors in row/column references are the most common failure mode.

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