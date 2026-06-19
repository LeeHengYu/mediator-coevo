# Task Instruction

Execute the following steps precisely to complete the task.

## 0. Setup
```bash
mkdir -p /root/output
pip install openpyxl
```

## 1. Inspect the workbook structure
Open `/root/data/workbook.xlsx` with openpyxl and inspect:
- Sheet names
- Sheet `Task`: print rows 10-50 for columns D through L (values and current formulas). Pay special attention to:
  - Row 10 (years)
  - Column D rows 12-17, 19-24, 26-31 (series codes)
  - Column G or labels for rows 12-17, 19-24, 26-31 (to understand which block is which metric)
  - Rows 35-40 (port names/labels for Net container flow)
  - Rows 42-47 (labels: min, max, median, mean, 25th, 75th percentile)
  - Row 50 (CPA weighted mean)
- Sheet `Data`: print rows 21-38 to see the layout (which row has which series code, which columns have which years, header row structure).

Print all of this before making any edits. This is critical to understand the exact cell references.

## 2. Understand the Data sheet layout
From the Data sheet rows 21-38, determine:
- Which row contains headers (series codes in which column, years in which row)
- The exact range structure so we know how to build VLOOKUP/HLOOKUP/INDEX-MATCH formulas
- Whether data is arranged with series codes in rows and years in columns, or vice versa

## 3. Populate H12:L17, H19:L24, H26:L31 with lookup formulas
For each cell in these ranges, write a formula that:
- Uses the series code from column D of that row
- Uses the year from row 10 of that column
- Looks up the value from Data!$21:$38
- Uses one of the allowed patterns: INDEX/MATCH/MATCH is typically most flexible

The exact formula pattern depends on the Data sheet layout discovered in step 2. For example, if Data has series codes in column A and years across row 21 (or some header row), use:
`=INDEX(Data!<data_range>, MATCH($D12, Data!<series_code_column>, 0), MATCH(H$10, Data!<year_row>, 0))`

Adjust the ranges based on what you find. Use absolute references for the lookup arrays ($D12 for series code with column locked, H$10 for year with row locked) so formulas can be filled across the range.

Write these formulas using openpyxl by setting each cell's value to the formula string (e.g., cell.value = '=INDEX(...)').

## 4. Populate H35:L40 with Net Container Flow formulas
Based on the block structure:
- Rows 12-17 appear to be one metric (e.g., Loaded Containers Inbound)
- Rows 19-24 appear to be another metric (e.g., Loaded Containers Outbound)  
- Rows 26-31 appear to be Terminal Throughput Capacity

Verify which block is which by reading the labels. Then for each cell in H35:L40:
`=(H12-H19)/H26*100` (adjusting row references for the corresponding port)

The six ports in rows 35-40 should correspond to the six ports in rows 12-17, 19-24, 26-31. Verify the ordering matches.

## 5. Populate H42:L47 with summary statistics
For each column H through L:
- Row 42: `=MIN(H35:H40)`
- Row 43: `=MAX(H35:H40)`
- Row 44: `=MEDIAN(H35:H40)`
- Row 45: `=AVERAGE(H35:H40)` (simple mean)
- Row 46: `=PERCENTILE(H35:H40,0.25)` (25th percentile)
- Row 47: `=PERCENTILE(H35:H40,0.75)` (75th percentile)

Verify the order of min/max/median/mean/25th/75th by reading the labels in the rows first. Adjust row assignments to match the actual labels.

## 6. Populate H50:L50 with weighted mean using SUMPRODUCT
For each column H through L:
`=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)`

This computes the weighted mean of Net Container Flow percentages weighted by Terminal Throughput Capacity.

## 7. Save
Save the workbook to `/root/output/result.xlsx`. Do NOT change formatting, do NOT add sheets.

## 8. Verify
Reopen `/root/output/result.xlsx` and print the formula content of all modified cells to confirm:
- H12:L17, H19:L24, H26:L31 contain INDEX/MATCH (or similar) lookup formulas
- H35:L40 contain the net container flow calculation
- H42:L47 contain MIN, MAX, MEDIAN, AVERAGE, PERCENTILE formulas
- H50:L50 contain SUMPRODUCT formulas
- No extra sheets were added
- The workbook opens without errors

## Important Notes
- Read the actual cell contents and labels BEFORE writing any formulas. The exact row/column references depend on what you find.
- Use openpyxl with data_only=False to preserve and write formulas.
- Do not overwrite any cells outside the specified ranges.
- Do not change formatting, styles, or cell colors.
- If the labels for rows 42-47 differ from my assumed order (min/max/median/mean/25th/75th), match the formulas to the actual labels.

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