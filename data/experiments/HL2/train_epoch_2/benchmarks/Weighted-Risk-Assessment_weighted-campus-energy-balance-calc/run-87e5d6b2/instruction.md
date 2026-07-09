# Task Instruction

Execute the following steps to produce `/root/output/result.xlsx`:

## 0 – Preparation
- `mkdir -p /root/output`
- Open `/root/data/workbook.xlsx` with openpyxl (`data_only=False`) and inspect:
  - Sheet `Task`: read the series codes in column D for rows 12-17, 19-24, 26-31. Read the years in H10:L10. Read any existing content/formatting in the yellow target ranges. Read row labels in rows 35-40, 42-47, 50.
  - Sheet `Data`: read rows 21-38 to understand the layout (which column holds the series code, which row holds what, how years are arranged). Print enough to understand the exact column/row positions.
- Print all findings before writing any formulas.

## 1 – Lookup formulas in H12:L17, H19:L24, H26:L31

For every cell in these three blocks, write an `INDEX`/`MATCH` formula that:
- Looks up the series code from column D of the current row against the series-code column in `Data!$21:$38`.
- Looks up the year from row 10 of the current column against the year row in `Data`.
- Uses the pattern: `=INDEX(Data!<data_range>, MATCH(<series_code_cell>, Data!<series_code_column>, 0), MATCH(<year_cell>, Data!<year_row>, 0))`
- Adjust the exact ranges based on what you discovered in step 0. Lock rows/columns with `$` appropriately so the formula can be dragged across the block if needed, but since you're writing each cell individually that's less critical—just make sure each cell references the correct series code and year.

## 2 – Net renewable balance in H35:L40

For each campus (rows 35-40) and each year column (H-L), write:
`=(H12 - H19) / H26 * 100`
adjusting row references so that:
- Row 12-17 = Renewable Generation (first block)
- Row 19-24 = Grid Consumption (second block)
- Row 26-31 = Baseline Energy Demand (third block)
Match each campus row correctly (campus in row 35 corresponds to row 12, 19, 26; campus in row 36 to rows 13, 20, 27; etc.).

## 3 – Summary statistics in H42:L47

For each year column (H through L), write these formulas:
- Row 42 (Min): `=MIN(H35:H40)`
- Row 43 (Max): `=MAX(H35:H40)`
- Row 44 (Median): `=MEDIAN(H35:H40)`
- Row 45 (Mean): `=AVERAGE(H35:H40)`
- Row 46 (25th percentile): `=PERCENTILE(H35:H40,0.25)`
- Row 47 (75th percentile): `=PERCENTILE(H35:H40,0.75)`

**CRITICAL**: Use `PERCENTILE` (legacy name), NOT `PERCENTILE.INC`. The previous run failed with `#NAME?` errors because the evaluator engine does not recognize `PERCENTILE.INC`. Double-check after writing that the formula strings stored are exactly `PERCENTILE(...)` with no `.INC` suffix.

## 4 – Weighted mean in H50:L50

For each year column (H through L):
`=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)`

This computes the weighted mean of the net-renewable-balance percentages using Baseline Energy Demand as weights.

## 5 – Save and Validate
- Save the workbook to `/root/output/result.xlsx`.
- Re-open the saved file and print the formula strings (not computed values) for a sample of cells:
  - One cell from each lookup block (e.g., H12, H19, H26)
  - H35 (net balance)
  - H42 through H47 (all stats)
  - H50 (weighted mean)
- Specifically confirm that H46 and H47 contain `PERCENTILE` and NOT `PERCENTILE.INC`.
- Confirm no `#NAME?` or other error tokens appear in formula strings.
- Do NOT add any new sheets, macros, VBA, external links, or helper tabs.
- Do NOT alter existing formatting.

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