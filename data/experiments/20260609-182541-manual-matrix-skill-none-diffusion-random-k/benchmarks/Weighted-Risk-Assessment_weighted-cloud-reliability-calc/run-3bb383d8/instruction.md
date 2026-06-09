# Task Instruction

Implement the weighted cloud reliability calculation workbook as follows:

## Setup
1. Read the existing workbook at `/root/data/workbook.xlsx` using openpyxl. Inspect the `Task` and `Data` sheets to understand the layout:
   - On `Task` sheet: check column D for series codes, row 10 for years, and the yellow cell ranges.
   - On `Data` sheet: check rows 21-38 for the source data layout (columns and headers).
2. Note the exact column/row structure of the Data sheet so formulas reference correctly.

## Step 1: Populate lookup formulas in H12:L17, H19:L24, H26:L31
For each cell in these three blocks, write an `INDEX(MATCH,MATCH)` formula:
- The lookup uses two keys: the series code from column D of the current row, and the year from row 10.
- The data source is on sheet `Data` in rows 21:38.
- Use this pattern: `=INDEX(Data!$B$21:$<lastcol>$38, MATCH($D<row>,Data!$A$21:$A$38,0), MATCH(H$10,Data!$B$20:$<lastcol>$20,0))`
- Adjust the exact ranges after inspecting the Data sheet layout. The row anchor for series codes should be column A (or whichever column holds them), and the column anchor for years should be the header row above the data.
- Make sure the column reference for the series code uses `$D<row>` (absolute column, relative row) and the year reference uses `<col>$10` (relative column, absolute row) so the formula copies correctly across the 5 columns and down the rows.

## Step 2: Net reliability gap in H35:L40 and statistics in H42:L47
For H35:L40, write formulas for each of the six regions:
`=(H12-H19)/H26*100` (adjusting row references per region — row 12 maps to row 35, row 13 to row 36, etc., and similarly for the Failed API Requests block rows 19-24 and Compute Capacity block rows 26-31).

Specifically:
- H35 = (H12 - H19) / H26 * 100
- H36 = (H13 - H20) / H27 * 100
- H37 = (H14 - H21) / H28 * 100
- H38 = (H15 - H22) / H29 * 100
- H39 = (H16 - H23) / H30 * 100
- H40 = (H17 - H24) / H31 * 100

And similarly for columns I through L.

For H42:L47, write column-wise statistics over H35:H40 (through L35:L40):
- Row 42: `=MIN(H35:H40)`
- Row 43: `=MAX(H35:H40)`
- Row 44: `=MEDIAN(H35:H40)`
- Row 45: `=AVERAGE(H35:H40)`
- Row 46: `=PERCENTILE.INC(H35:H40,0.25)`
- Row 47: `=PERCENTILE.INC(H35:H40,0.75)`

**CRITICAL**: Use `PERCENTILE.INC` (not `PERCENTILE` alone, which may cause #NAME? errors in some engines). This was the cause of failures in similar tasks.

## Step 3: Weighted mean in H50:L50
For each column H through L:
`=SUMPRODUCT(H35:H40,H26:H31)/SUM(H26:H31)`

This computes the weighted mean of the Net reliability gap values using Compute Capacity as weights.

## Saving
1. Save the workbook to `/root/output/result.xlsx`. Create the `/root/output/` directory if it doesn't exist.
2. Do NOT add any new sheets, macros, VBA, external links, or helper tabs.
3. Preserve all existing formatting.

## Validation
After saving, re-open the file with openpyxl (data_only=False) and verify:
- Cells in H12:L17, H19:L24, H26:L31 contain string formulas (starting with '=').
- Cells in H35:L40 contain formulas.
- Cells in H42:L47 contain formulas, and specifically rows 46-47 use `PERCENTILE.INC`.
- Cells in H50:L50 contain `SUMPRODUCT` formulas.
- No extra sheets were added.

## Important Notes
- Before writing any formulas, INSPECT the Data sheet thoroughly to get exact cell references (which column has series codes, which row has year headers, what the data range boundaries are).
- Adjust all formula references based on what you actually find in the workbook — do not assume the layout matches my description exactly.
- The row-to-row mapping between the three blocks (rows 12-17, 19-24, 26-31) and the derived block (rows 35-40) must correspond to the same six regions.

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