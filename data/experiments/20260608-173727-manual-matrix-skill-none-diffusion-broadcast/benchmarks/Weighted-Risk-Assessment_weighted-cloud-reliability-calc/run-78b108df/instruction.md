# Task Instruction

You must update `/root/data/workbook.xlsx` and save the result to `/root/output/result.xlsx`. Follow these steps precisely.

## Preliminary: Inspect the workbook
1. `mkdir -p /root/output`
2. Use Python with openpyxl to open `/root/data/workbook.xlsx` and inspect:
   - Sheet names (confirm `Task` and `Data` exist).
   - On sheet `Task`: read row 10 (the year headers in columns H–L), column D rows 12–17, 19–24, 26–31 to see the series codes, rows 35–40 to see region labels, row 41 label, rows 42–47 labels (min/max/median/mean/25th/75th), row 50 label.
   - On sheet `Data`: read rows 21–38 to understand the data layout (which row is the header row, which column has series codes, where the year data starts, etc.).
   - Print all of this so you understand the exact structure before writing any formulas.

## Step 1: Populate H12:L17, H19:L24, H26:L31 with lookup formulas

For each cell in these three blocks, write a spreadsheet formula (not a Python-computed value) that looks up data from `Data!$21:$38`. The formula must use:
- The series code from column D of the same row on sheet `Task`
- The year from row 10 of the same column on sheet `Task`

Use one of these patterns: `VLOOKUP`+`MATCH`, `HLOOKUP`+`MATCH`, `XLOOKUP`+`MATCH`, or `INDEX`+`MATCH`.

IMPORTANT: Before writing formulas, determine the exact layout of Data rows 21:38. Identify:
- Which column contains the series codes (likely column A or B or similar)
- Which row contains the year headers
- The data range for values

Then construct the appropriate formula. For example, if using INDEX+MATCH:
`=INDEX(Data!<data_range>, MATCH($D12, Data!<series_code_column>, 0), MATCH(H$10, Data!<year_header_row>, 0))`

Adjust references based on actual layout. Use $ signs appropriately so column D and row 10 references are anchored correctly for copying across the block.

## Step 2: Net reliability gap in H35:L40 and summary stats in H42:L47

For H35:L40 (6 regions × 5 years), write spreadsheet formulas:
`= (H12 - H19) / H26 * 100`
where H12 corresponds to "Successful API Requests" row, H19 to "Failed API Requests" row, and H26 to "Compute Capacity" row for the same region. Adjust row references for each of the 6 regions (rows 12–17 map to rows 35–40, rows 19–24 map to rows 35–40, rows 26–31 map to rows 35–40). Specifically:
- Row 35: `=(H12-H19)/H26*100`
- Row 36: `=(H13-H20)/H27*100`
- Row 37: `=(H14-H21)/H28*100`
- Row 38: `=(H15-H22)/H29*100`
- Row 39: `=(H16-H23)/H30*100`
- Row 40: `=(H17-H24)/H31*100`

For H42:L47, write spreadsheet formulas for column-wise statistics over H35:H40 (through L35:L40):
- Row 42 (MIN): `=MIN(H35:H40)`
- Row 43 (MAX): `=MAX(H35:H40)`
- Row 44 (MEDIAN): `=MEDIAN(H35:H40)`
- Row 45 (MEAN): `=AVERAGE(H35:H40)`
- Row 46 (25th percentile): `=PERCENTILE(H35:H40, 0.25)` or `=PERCENTILE.INC(H35:H40, 0.25)`
- Row 47 (75th percentile): `=PERCENTILE(H35:H40, 0.75)` or `=PERCENTILE.INC(H35:H40, 0.75)`

Check the actual row labels (42–47) to confirm which row is which statistic; adjust accordingly.

## Step 3: Weighted mean in H50:L50

For each column H–L in row 50, write:
`=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)`

This computes the weighted mean of the Net reliability gap percentages using Compute Capacity as weights.

## Implementation approach

Use openpyxl in Python. When writing formulas:
- Use `ws['H12'] = '=INDEX(...)'` syntax (string starting with `=`).
- Loop over columns H(8) through L(12) and appropriate rows.
- Do NOT change any existing formatting, styles, or cell values outside the specified ranges.
- Do NOT add sheets, macros, VBA, or external links.

After writing all formulas, save to `/root/output/result.xlsx`.

## Validation

After saving, re-open the file and:
1. Confirm all target cells contain formula strings (not None or plain values).
2. Print a sample of the formulas to verify correctness.
3. Confirm no extra sheets were added.
4. Confirm the file is saved at `/root/output/result.xlsx`.

CRITICAL: Read the actual workbook structure carefully before writing any formulas. The exact row/column mapping on the Data sheet and the exact labels/positions on the Task sheet must be verified by inspection, not assumed. If the row labels for statistics (rows 42-47) are in a different order than min/max/median/mean/25th/75th, match the formulas to the actual labels.

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