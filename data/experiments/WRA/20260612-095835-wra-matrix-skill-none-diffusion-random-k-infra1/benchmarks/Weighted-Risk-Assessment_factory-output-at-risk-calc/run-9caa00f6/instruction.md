# Task Instruction

Execute the following steps in order to produce `/root/output/result.xlsx`.

## Phase 0 – Inspection
1. Open `/root/data/workbook.xlsx` with openpyxl (keep_vba=False, data_only=False).
2. Print the sheet names to confirm `Task` and `Data` exist.
3. On sheet `Task`:
   - Print rows 10-11 (to see the year headers in H10:L10 and any labels).
   - Print rows 12-17 with columns A-L (first lookup block; note series codes in column D).
   - Print rows 19-24 with columns A-L (second lookup block).
   - Print rows 26-31 with columns A-L (third lookup block – Rated Production Capacity).
   - Print rows 33-50 with columns A-L (derived rows: Net production slack, stats, weighted mean).
4. On sheet `Data`:
   - Print rows 1-5 to understand headers/layout.
   - Print rows 19-40 with all used columns to see the source data block (rows 21:38 per the task, but inspect a margin around it).
5. From the inspection, identify:
   - The exact column in `Data` that holds series codes.
   - The row in `Data` that holds years.
   - The data range for VLOOKUP/INDEX-MATCH.
   - The series codes referenced in Task column D for each of the three blocks.
   - The years in Task row 10 (H10:L10).
   - Which rows hold "Finished Output", "Scrap And Rework", and "Rated Production Capacity" labels (to map blocks correctly).

## Phase 1 – Populate lookup blocks H12:L17, H19:L24, H26:L31
For each cell in these three 6×5 blocks, write an Excel formula using INDEX/MATCH:
```
=INDEX(Data!<data_range>, MATCH($D<row>, Data!<series_code_column>, 0), MATCH(H$10, Data!<year_row>, 0))
```
- Use mixed references: `$D<row>` (absolute column, relative row) for the series code, `H$10` (relative column, absolute row) for the year.
- Identify the correct `<data_range>`, `<series_code_column>`, and `<year_row>` from Phase 0 inspection.
- The `<data_range>` should cover the numeric values in Data rows 21:38 (or whatever the inspection reveals), and the `<series_code_column>` and `<year_row>` should be the corresponding label column/row.

## Phase 2 – Net production slack (H35:L40)
For each of the 6 plants × 5 years, write a formula:
```
=(H12-H19)/H26*100
```
where H12 is the Finished Output cell, H19 is the Scrap And Rework cell, and H26 is the Rated Production Capacity cell for the same plant and year. Adjust row references for each plant row. Use relative column so it shifts across H-L.

Verify from the inspection which block corresponds to which metric:
- Block H12:L17 → check if it's Finished Output
- Block H19:L24 → check if it's Scrap And Rework
- Block H26:L31 → check if it's Rated Production Capacity
If the mapping is different, adjust accordingly.

## Phase 3 – Summary statistics (H42:L47)
For each column H through L, write formulas in rows 42-47:
- Row 42 (Minimum): `=MIN(H35:H40)`
- Row 43 (Maximum): `=MAX(H35:H40)`
- Row 44 (Median): `=MEDIAN(H35:H40)`
- Row 45 (Mean): `=AVERAGE(H35:H40)`
- Row 46 (25th percentile): `=PERCENTILE(H35:H40,0.25)`
- Row 47 (75th percentile): `=PERCENTILE(H35:H40,0.75)`

Check the labels in column A/B/C of rows 42-47 during inspection to confirm the exact order (min, max, median, mean, 25th, 75th). Adjust row assignments if the label order differs.

## Phase 4 – Weighted mean (H50:L50)
For each column H through L:
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```
This computes the weighted mean of Net production slack using Rated Production Capacity as weights.

## Phase 5 – Save and validate
1. Save the workbook to `/root/output/result.xlsx` (create `/root/output/` if needed).
2. Re-open the saved file with openpyxl (data_only=False) and spot-check:
   - Cell H12 contains a formula string (not None, not a bare value).
   - Cell H35 contains a formula string.
   - Cell H42 contains a formula string.
   - Cell H50 contains a formula string.
3. Also open with data_only=True and check that the cached values are not obviously wrong (they may be None if no calc engine ran, which is acceptable – the formulas are what matter).
4. Print confirmation of successful completion.

## Critical warnings
- Do NOT strip formulas or write plain values. Every yellow cell must contain an Excel formula string starting with '='.
- Do NOT add new sheets, macros, VBA, or external links.
- Do NOT alter existing formatting.
- If any cell in the lookup blocks returns None when read back (data_only=False), that means the formula was not written – debug immediately.
- The avoid/recheck artifact warns that a prior run produced None values in lookup and derived blocks. This happened because formulas were not actually written to cells. Double-check that `ws['H12'].value` etc. return formula strings after writing.

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