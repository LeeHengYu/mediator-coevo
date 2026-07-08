# Task Instruction

Execute the following five-phase plan to update `/root/data/workbook.xlsx` and save the result to `/root/output/result.xlsx`.

## Phase 0: Inspect
1. `mkdir -p /root/output`
2. Open `/root/data/workbook.xlsx` with openpyxl (data_only=False).
3. Print the sheet names to confirm `Task` and `Data` exist.
4. On sheet `Task`:
   - Print rows 10-11 (to see year headers in H10:L10).
   - Print rows 12-17 column D (series codes for block 1: Successful API Requests).
   - Print rows 19-24 column D (series codes for block 2: Failed API Requests).
   - Print rows 26-31 column D (series codes for block 3: Compute Capacity).
   - Print rows 35-40 column D (region names for Net reliability gap).
   - Print rows 42-47 column A-D (stat labels: min, max, median, mean, 25th, 75th).
   - Print row 50 columns A-G (to see the GCM weighted-mean label).
5. On sheet `Data`:
   - Print row 21 (header row — should contain years or column labels).
   - Print column A (or B) rows 21-38 to see series codes.
   - Print row 21 columns A through ~Q to see the year positions.
   - Identify: which column holds the series code, and which row holds the year headers.

Record every coordinate precisely before proceeding.

## Phase 1: INDEX/MATCH Lookup Formulas (H12:L31)
For each cell in the three blocks `H12:L17`, `H19:L24`, `H26:L31`:
- The lookup value pair is: (series code from column D of the current row, year from row 10 of the current column).
- Use an `INDEX/MATCH` formula against the Data sheet range rows 21:38.
- The formula pattern (adjust exact ranges based on Phase 0 inspection):
  ```
  =INDEX(Data!$B$22:$<lastcol>$38, MATCH($D<row>, Data!$A$22:$A$38, 0), MATCH(H$10, Data!$B$21:$<lastcol>$21, 0))
  ```
  Replace `$A$22:$A$38` with the actual series-code column range, `$B$21:$<lastcol>$21` with the actual year header range, and `$B$22:$<lastcol>$38` with the data body range — all based on what Phase 0 reveals.
- Use absolute references for the Data ranges and mixed references (`$D<row>` and `H$10`) so the formula is correct per cell.
- Write the formula as a string into each cell using openpyxl.

## Phase 2: Net Reliability Gap (H35:L40)
For each of the 6 regions (rows 35-40) and each of the 5 year columns (H-L):
- Write a formula: `=(<SuccessCell> - <FailedCell>) / <CapacityCell> * 100`
  where SuccessCell is from block H12:L17 (same relative row offset), FailedCell from H19:L24, CapacityCell from H26:L31.
- Example for H35: `=(H12-H19)/H26*100`

## Phase 3: Summary Statistics (H42:L47)
For each year column (H-L), write these formulas referencing the Net reliability gap block (rows 35:40 of that column):
- Row 42 (Min): `=MIN(H35:H40)`
- Row 43 (Max): `=MAX(H35:H40)`
- Row 44 (Median): `=MEDIAN(H35:H40)`
- Row 45 (Mean): `=AVERAGE(H35:H40)`
- Row 46 (25th percentile): `=PERCENTILE(H35:H40,0.25)`
- Row 47 (75th percentile): `=PERCENTILE(H35:H40,0.75)`

Verify the stat-label order from Phase 0 and adjust row assignments if the labels differ from min/max/median/mean/25th/75th.

## Phase 4: Weighted Mean (H50:L50)
For each year column (H-L):
- `=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)`
  This computes the weighted mean of Net reliability gap using Compute Capacity as weights.

## Phase 5: Save and Validate
1. Save the workbook to `/root/output/result.xlsx`.
2. Reopen the saved file and spot-check:
   - Cell H12 contains a formula string (not a bare value).
   - Cell H35 contains a formula string.
   - Cell H42 contains a formula string.
   - Cell H50 contains a formula string.
3. Print a few formula strings to confirm correctness.

IMPORTANT:
- Do NOT change any existing formatting, sheet structure, or cell values outside the specified ranges.
- Do NOT add sheets, macros, VBA, external links, or helper tabs.
- Use openpyxl throughout. Do not use xlsxwriter or pandas ExcelWriter.
- Adapt all ranges based on the actual inspection in Phase 0. Do not hardcode ranges without verifying them first.

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