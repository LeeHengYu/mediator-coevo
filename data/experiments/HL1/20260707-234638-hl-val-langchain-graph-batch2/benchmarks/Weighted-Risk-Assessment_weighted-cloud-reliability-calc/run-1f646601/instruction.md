# Task Instruction

You must update `/root/data/workbook.xlsx` by writing spreadsheet formulas into specific cells on the `Task` sheet, then save the result to `/root/output/result.xlsx`. Follow these phases exactly.

## Phase 0 – Inspect the workbook
1. `mkdir -p /root/output`
2. Use `openpyxl` (with `data_only=False`) to open `/root/data/workbook.xlsx`.
3. Print the sheet names.
4. On sheet `Task`, print:
   - Row 10 (headers / years) — especially columns H through L.
   - Column D for rows 12–17, 19–24, 26–31 (series codes).
   - Row 35 label area and rows 35–40 column D (region names).
   - Rows 42–47 column D–G (stat labels: min, max, median, mean, 25th, 75th percentile).
   - Row 50 columns D–G (GCM label / weights info).
5. On sheet `Data`, print:
   - Row 1 (or header row) to understand column layout.
   - Rows 21–38 to see the source data block (print all columns that have content).
6. Identify:
   - Which column on `Data` holds the series codes (the lookup key).
   - Which row on `Data` holds the year headers.
   - The exact data range for the lookup (e.g., `Data!A21:Z38` or similar).

## Phase 1 – Populate lookup formulas in H12:L17, H19:L24, H26:L31

For each cell in these three blocks, write an `INDEX/MATCH` formula that:
- Matches the series code in column D of the current row against the series-code column in `Data` rows 21–38.
- Matches the year in row 10 of the current column against the year header row in `Data`.
- Returns the intersecting value.

Use the inspection results to build the correct absolute references. The formula pattern will be something like:
```
=INDEX(Data!$B$21:$Z$38, MATCH($D12, Data!$A$21:$A$38, 0), MATCH(H$10, Data!$B$20:$Z$20, 0))
```
Adjust column/row references based on the actual layout you discovered in Phase 0. Use `$` anchoring so the series-code column reference is row-absolute and the year reference is column-absolute.

Write these formulas using `openpyxl` by assigning the formula string (starting with `=`) to each cell's `.value`.

## Phase 2 – Net reliability gap (H35:L40)

For each of the 6 regions (rows 35–40) and 5 year columns (H–L), write a formula:
```
=(H12 - H19) / H26 * 100
```
where H12 corresponds to the Successful API Requests block, H19 to the Failed API Requests block, and H26 to the Compute Capacity block. Adjust row references to match the correct region row within each block (row 35 maps to the first region in rows 12, 19, 26; row 36 maps to the second; etc.).

## Phase 3 – Summary statistics (H42:L47)

For each year column (H–L), write formulas in rows 42–47. Map each row to the correct stat based on the labels you found in Phase 0. Use these patterns:
- MIN: `=MIN(H35:H40)`
- MAX: `=MAX(H35:H40)`
- MEDIAN: `=MEDIAN(H35:H40)`
- MEAN (simple average): `=AVERAGE(H35:H40)`
- 25th percentile: `=PERCENTILE(H35:H40,0.25)`
- 75th percentile: `=PERCENTILE(H35:H40,0.75)`

Match each formula to the correct row based on the label in column D.

## Phase 4 – Weighted mean for GCM (H50:L50)

For each year column (H–L), write a `SUMPRODUCT` formula:
```
=SUMPRODUCT(H42:H47, H26:H31) / SUM(H26:H31)
```
Wait — re-read the instruction: "using the Step 2 percentages as values and the Compute Capacity block in H26:L31 as weights". So the values are H35:H40 (the net reliability gap percentages), and the weights are H26:H31 (compute capacity). The formula should be:
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```
Adjust column letter for each column H through L.

## Phase 5 – Save and validate
1. Save the workbook to `/root/output/result.xlsx`.
2. Re-open the saved file and print the formulas in a sampling of cells (e.g., H12, L17, H19, L24, H26, L31, H35, L40, H42, L47, H50, L50) to confirm they are correctly written.
3. Confirm no new sheets were added and no macros are present.

## Important constraints
- Do NOT use `data_only=True` when loading — you need to preserve and write formulas.
- Do NOT add sheets, macros, VBA, external links, or helper tabs.
- Do NOT alter existing formatting.
- Use `openpyxl` for all operations.
- If any cell reference or layout differs from what's assumed above, adapt based on your Phase 0 inspection. The Phase 0 inspection is critical — do it thoroughly before writing any formulas.

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