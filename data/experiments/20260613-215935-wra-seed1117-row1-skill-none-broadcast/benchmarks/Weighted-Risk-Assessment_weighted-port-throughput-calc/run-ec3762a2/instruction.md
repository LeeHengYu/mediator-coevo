# Task Instruction

You must update the Excel workbook at `/root/data/workbook.xlsx` and save the result to `/root/output/result.xlsx`. Follow these steps precisely:

## Preliminary
1. `mkdir -p /root/output`
2. Inspect the workbook structure using openpyxl to understand:
   - Sheet `Task`: layout of columns, what is in D12:D17, D19:D24, D26:D31 (series codes), row 10 (years in H10:L10), the yellow target ranges, and the existing content/formatting.
   - Sheet `Data`: layout of rows 21:38 — identify what column holds the series codes and which row holds the year headers. Note the exact column letters.
3. Print out these key reference values so you can write correct formulas.

## Step 1: Populate lookup formulas in H12:L17, H19:L24, H26:L31

For each cell in these three blocks, write an `INDEX/MATCH` formula that:
- Looks up the series code from column D of the current row against the series-code column in `Data` rows 21:38.
- Looks up the year from row 10 of the current column against the year header row in `Data`.
- Returns the intersecting value.

Use the pattern: `=INDEX(Data!<data_range>, MATCH(<series_code_cell>, Data!<series_code_column>, 0), MATCH(<year_cell>, Data!<year_header_row>, 0))`

Make sure:
- The `Data!<data_range>` covers the full rectangular block of numeric data in rows 21:38.
- The `<series_code_column>` is the column in Data that contains the series codes, spanning rows 21:38.
- The `<year_header_row>` is the row in Data that contains the year values, spanning the same columns as the data range.
- Use appropriate `$` signs for anchoring: anchor the series code column reference and year header row reference, but let the current-row series code cell and current-column year cell float appropriately so formulas can be filled across the block.

## Step 2: Net container flow in H35:L40 and statistics in H42:L47

For H35:L40, each cell should compute:
`=(H12 - H19) / H26 * 100` (adjusted for the correct row offsets)

Specifically, for row 35 (first port): `=(H12 - H19) / H26 * 100`
For row 36: `=(H13 - H20) / H27 * 100`
...and so on for all 6 ports, across columns H through L.

For H42:L47 (column-wise statistics over H35:L40):
- Row 42: `=MIN(H35:H40)` (minimum)
- Row 43: `=MAX(H35:H40)` (maximum)
- Row 44: `=MEDIAN(H35:H40)` (median)
- Row 45: `=AVERAGE(H35:H40)` (simple mean)
- Row 46: `=PERCENTILE(H35:H40, 0.25)` (25th percentile)
- Row 47: `=PERCENTILE(H35:H40, 0.75)` (75th percentile)

Before writing these, verify the row labels in the Task sheet to confirm which statistic goes in which row. Adjust the row assignments if the labels differ from the order above.

## Step 3: Weighted mean in H50:L50

For each column H through L:
`=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)`

This uses the net container flow percentages as values and Terminal Throughput Capacity as weights.

## Implementation approach

Use openpyxl in Python. Open the workbook with `data_only=False` to preserve existing formulas. When writing formulas, assign them as strings starting with `=`. Do NOT use `data_only=True`. Save to `/root/output/result.xlsx`.

After saving, re-open the file and verify:
1. The formula cells are not None — they contain formula strings.
2. Spot-check a few cells to confirm formula text looks correct.
3. Confirm no new sheets were added.
4. Confirm the file saves without error.

## Critical warnings
- Do NOT leave any target cells empty. The failed hospital-bedflow task failed precisely because target ranges were left unpopulated (returned None). Every cell in H12:L17, H19:L24, H26:L31, H35:L40, H42:L47, and H50:L50 must contain a formula.
- Do NOT add sheets, macros, VBA, or external links.
- Preserve all existing formatting.
- Read the actual sheet structure before writing any formulas — do not assume column/row positions.

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