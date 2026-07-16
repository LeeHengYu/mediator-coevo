# Task Instruction

You must update the workbook `/root/data/workbook.xlsx` and save the result to `/root/output/result.xlsx`. Follow these steps precisely.

## Phase 0 – Inspect the workbook

1. `mkdir -p /root/output`
2. Use `openpyxl` (read-only, data_only=False) to inspect:
   - Sheet names (confirm `Task` and `Data` exist).
   - On sheet `Task`: read cells D12:D17 (series codes for block 1), D19:D24 (block 2), D26:D31 (block 3). Read H10:L10 (years row). Read row 35-40 column D for service names. Read rows 42-47 column D-G for the stat labels (min, max, median, mean, 25th, 75th). Read row 50 for the weighted-mean label. Read H26:L31 current values (or note they'll be formulas after Step 1).
   - On sheet `Data`: read rows 21-38 to understand the layout — which row holds which field, which column holds which year, where series codes appear. Print the first column (or row headers) and the top row of that block so you understand whether data is arranged with series codes in rows or columns.
   - Print all findings before writing any formulas.

## Phase 1 – Populate lookup formulas in H12:L17, H19:L24, H26:L31

Based on the inspection, write `INDEX/MATCH` formulas into every yellow cell in those three blocks. Each formula must look up:
- **Row key**: the series code from column D of the same row (use `$D12` style with absolute column).
- **Column key**: the year from row 10 (use `H$10` style with absolute row).
- **Data source**: sheet `Data`, rows 21:38.

Use the pattern:
```
=INDEX(Data!<data_range>, MATCH($D12, Data!<series_code_column>, 0), MATCH(H$10, Data!<year_row>, 0))
```
Adjust `<data_range>`, `<series_code_column>`, and `<year_row>` based on what you found in Phase 0. The references must use mixed addressing so they copy correctly across the 5 columns (H-L) and 6 rows within each block.

Write formulas using `openpyxl` by assigning formula strings (starting with `=`) to each cell. Do NOT set values — set formula strings.

## Phase 2 – Net SLA buffer (H35:L40)

For each of the 6 services (rows 35-40) and 5 year-columns (H-L), enter a formula:
```
=(H12 - H19) / H26 * 100
```
where row 12/19/26 correspond to the same service's row in the three blocks (Latency Budget Preserved = block 1 rows 12-17, Latency Budget Consumed = block 2 rows 19-24, Covered Request Capacity = block 3 rows 26-31). Adjust row references for each service row. Use relative references so each cell points to the correct block-1, block-2, block-3 row for that service.

Specifically:
- H35 = (H12 - H19) / H26 * 100
- H36 = (H13 - H20) / H27 * 100
- H37 = (H14 - H21) / H28 * 100
- H38 = (H15 - H22) / H29 * 100
- H39 = (H16 - H23) / H30 * 100
- H40 = (H17 - H24) / H31 * 100
- Same pattern for columns I through L.

## Phase 3 – Summary statistics (H42:L47)

For each column H-L, enter these formulas in the 6 stat rows:
- Row 42 (MIN): `=MIN(H35:H40)`
- Row 43 (MAX): `=MAX(H35:H40)`
- Row 44 (MEDIAN): `=MEDIAN(H35:H40)`
- Row 45 (AVERAGE): `=AVERAGE(H35:H40)`
- Row 46 (25th percentile): `=PERCENTILE(H35:H40, 0.25)`
- Row 47 (75th percentile): `=PERCENTILE(H35:H40, 0.75)`

**Important**: Verify the stat labels in column D/E/F/G of rows 42-47 during Phase 0 to confirm the correct order (min/max/median/mean/p25/p75). Adjust row assignments if the order differs.

## Phase 4 – Weighted mean (H50:L50)

For each column H-L:
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```
This computes the weighted mean of Net SLA buffer percentages weighted by Covered Request Capacity.

## Phase 5 – Save and validate

1. Save the workbook to `/root/output/result.xlsx`.
2. Re-open the saved file and print sample cells from each block to confirm formulas were written (they should show as formula strings, not values, when opened with data_only=False).
3. Verify no extra sheets were added, no macros, no external links.
4. Check that the formulas reference the correct Data sheet ranges by printing a few representative cells.

## Critical constraints
- Do NOT add new sheets, macros, VBA, external links, or helper tabs.
- Do NOT change existing formatting.
- Only write formula strings to cells; do not compute values in Python and write numbers.
- Use `openpyxl` for all Excel operations.
- If any cell in the yellow ranges already has content, overwrite it with the correct formula.

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