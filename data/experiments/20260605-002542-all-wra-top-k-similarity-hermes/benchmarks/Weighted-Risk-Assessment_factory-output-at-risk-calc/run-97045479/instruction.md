# Task Instruction

You need to update `/root/data/workbook.xlsx` and save the result to `/root/output/result.xlsx`.

## Preliminary Investigation

1. First, examine the workbook structure:
   - Open `/root/data/workbook.xlsx` with openpyxl (with `data_only=False` so you see formulas).
   - Print the sheet names.
   - On sheet `Task`: print rows 10-50 for columns D through L to understand the layout — especially the series codes in column D, the years in row 10, the yellow cell ranges, and what labels exist in rows 35-50.
   - On sheet `Data`: print rows 21-38 to understand the data layout — column headers, row labels, and how series codes and years map to data.
   - Also check if there are any existing formulas or values already in the target cells.
   - Check the test file at `/tests/test_outputs.py` to understand exactly what the verifier expects.

2. Determine the exact lookup structure:
   - What column on `Data` contains the series codes (that match column D on `Task`)?
   - What row on `Data` contains the years (that match row 10 on `Task`)?
   - This will determine the correct MATCH/INDEX or VLOOKUP/MATCH pattern.

## Step 1: Lookup Formulas in H12:L17, H19:L24, H26:L31

For each cell in these ranges, write a formula that:
- Takes the series code from column D of the same row on `Task`
- Takes the year from row 10 of the same column on `Task`
- Looks up the value from `Data!` rows 21:38

Use `INDEX(MATCH, MATCH)` pattern — this is the most reliable. For example:
`=INDEX(Data!$B$21:$Z$38, MATCH($D12, Data!$A$21:$A$38, 0), MATCH(H$10, Data!$B$20:$Z$20, 0))`

Adjust the exact ranges based on your inspection of the Data sheet layout. Make sure:
- The row lookup range for series codes is the correct column on Data.
- The column lookup range for years is the correct row on Data.
- References are anchored properly (mixed references: $ on the lookup arrays, $ on row for year, $ on column for series code).

## Step 2: Net Production Slack in H35:L40 and Statistics in H42:L47

For H35:L40, calculate:
`= (Finished Output - Scrap And Rework) / Rated Production Capacity * 100`

You need to identify which of the three blocks (H12:L17, H19:L24, H26:L31) corresponds to "Finished Output", "Scrap And Rework", and "Rated Production Capacity". Check the labels in the Task sheet (likely in column B or C near rows 11, 18, 25) to determine this mapping. Then for each cell, e.g. H35:
`= (H12 - H19) / H26 * 100`  (adjust based on actual block mapping)

For H42:L47, calculate column-wise statistics. Based on cross-task feedback, **do NOT use PERCENTILE.INC or PERCENTILE.EXC** — these cause #NAME? errors in some engines. Instead:
- **Minimum**: `=MIN(H35:H40)`
- **Maximum**: `=MAX(H35:H40)`
- **Median**: `=MEDIAN(H35:H40)`
- **Simple Mean**: `=AVERAGE(H35:H40)`
- **25th percentile**: `=PERCENTILE(H35:H40, 0.25)` — use `PERCENTILE`, NOT `PERCENTILE.INC` or `PERCENTILE.EXC`
- **75th percentile**: `=PERCENTILE(H35:H40, 0.75)` — use `PERCENTILE`, NOT `PERCENTILE.INC` or `PERCENTILE.EXC`

Check the row labels (likely in column B or C, rows 42-47) to confirm which row gets which statistic.

## Step 3: Weighted Mean in H50:L50

For each column (H through L):
`=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)`

This computes the weighted mean of the Net Production Slack percentages weighted by Rated Production Capacity.

## Important Notes

- Use `openpyxl` to write formulas (store formula strings in cells).
- Do NOT use `data_only=True` when loading — you need to preserve existing formulas.
- Preserve all existing formatting — do not clear or overwrite cells outside the target ranges.
- Create `/root/output/` directory if it doesn't exist.
- Save to `/root/output/result.xlsx`.
- After saving, re-open the file and print the formula cells to verify they were written correctly.
- Run any test files found (e.g., `cd /root && python -m pytest tests/`) to check your work.

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
Task metadata: author_email=catpaw@meituan.com, author_name=CatPaw Task Engineer, category=spreadsheet-formula-reuse, difficulty=hard, tags=[excel, formulas, lookup, statistics, weighted-mean].
Verifier config: timeout_sec=600.0.