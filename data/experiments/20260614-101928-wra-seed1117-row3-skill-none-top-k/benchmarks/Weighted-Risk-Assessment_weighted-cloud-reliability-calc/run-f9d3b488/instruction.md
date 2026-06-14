# Task Instruction

Execute the following steps to produce /root/output/result.xlsx from /root/data/workbook.xlsx.

## Step 0 – Inspect the workbook
Open /root/data/workbook.xlsx with openpyxl (data_only=False). Print:
- Sheet names.
- Task sheet: contents of column D rows 12-31 (series codes), row 10 columns H-L (years), rows 35-40 column D (region names for Net reliability gap), rows 42-47 column D (stat labels), row 50 column D.
- Data sheet: row 1 (or header row) to understand column layout, then rows 21-38 to see the lookup source. Print enough to identify the column that holds the series code and the row that holds years.

Use this inspection to confirm exact coordinates before writing any formula.

## Step 1 – Lookup formulas in H12:L31
For every cell in the three blocks H12:L17, H19:L24, H26:L31, write an INDEX/MATCH formula that:
- Looks up the series code from column D of the current row against the series-code column in Data!$21:$38.
- Looks up the year from Task row 10 against the year header row in the Data sheet.
- Pattern: =INDEX(Data!<data_range>, MATCH(<series_code_cell>, Data!<series_code_column>, 0), MATCH(<year_cell>, Data!<year_header_row>, 0))
- Use absolute references for the Data ranges and the year row; use a relative reference for the series code cell (column D of the current row) and mixed reference for the year cell (absolute row, relative column).

## Step 2 – Net reliability gap (H35:L40)
For each of the six regions (rows 35-40), compute:
  = (Successful API Requests value − Failed API Requests value) / Compute Capacity value * 100
where each value comes from the corresponding column in the lookup blocks above (H12:L17 = Successful API Requests, H19:L24 = Failed API Requests, H26:L31 = Compute Capacity). Match regions by their row offset within each block.

## Step 3 – Summary statistics (H42:L47)
For each column H through L:
- Row 42 (MIN): =MIN(H35:H40)  (adjust column letter)
- Row 43 (MAX): =MAX(H35:H40)
- Row 44 (MEDIAN): =MEDIAN(H35:H40)
- Row 45 (MEAN): =AVERAGE(H35:H40)
- Row 46 (25th percentile): =PERCENTILE(H35:H40, 0.25)
- Row 47 (75th percentile): =PERCENTILE(H35:H40, 0.75)

**Critical**: Use `PERCENTILE` (legacy name), NOT `PERCENTILE.INC` or `_xlfn.PERCENTILE.INC`. The legacy `PERCENTILE` avoids #NAME? errors in the evaluation environment.

## Step 4 – Weighted mean (H50:L50)
For each column H through L:
  =SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
This uses the Net reliability gap percentages as values and the Compute Capacity block as weights.

## Step 5 – Save
- Do NOT change any existing formatting, sheet names, or structure.
- Create /root/output/ if it doesn't exist.
- Save the workbook to /root/output/result.xlsx.

## Step 6 – Validate
Reopen /root/output/result.xlsx with openpyxl (data_only=False). Print the formula (not value) stored in cells H12, L17, H19, L24, H26, L31, H35, H40, H42, H47, H50, L50. Confirm none are None and all look correct. Also spot-check that the formulas reference the correct ranges.

## Important constraints
- Use openpyxl only. No xlsxwriter, no pandas ExcelWriter.
- Do not add sheets, macros, VBA, external links, or helper tabs.
- Preserve all existing content and formatting.
- Use legacy function names: PERCENTILE, MEDIAN, MIN, MAX, AVERAGE (not .INC/.EXC variants).
- Adapt all row/column references based on what you discover in Step 0. Do not hardcode Data sheet layout without verifying it first.

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