# Task Instruction

## Task: Update `/root/data/workbook.xlsx` with formulas and save to `/root/output/result.xlsx`

### Phase 0: Inspect the workbook
1. Create `/root/output/` directory if it doesn't exist.
2. Use `openpyxl` (or similar) to open `/root/data/workbook.xlsx` and inspect:
   - Sheet `Task`: Read cells in column D rows 12-17, 19-24, 26-31 to find the series codes. Read row 10 columns H-L to find the years. Read rows 35-40 column D for service names. Read H42:L47 labels. Read H50:L50 area and any labels near row 50.
   - Sheet `Data`: Read rows 21-38 to understand the data layout — specifically which row/column holds what, what the header row is, how series codes and years are arranged.
   - Print all of this information so we understand the exact structure before writing any formulas.

### Phase 1: Populate lookup formulas in H12:L17, H19:L24, H26:L31

For each cell in these three blocks, write a spreadsheet **formula** (not a computed value). The formula must use one of these patterns: `VLOOKUP+MATCH`, `HLOOKUP+MATCH`, `XLOOKUP+MATCH`, or `INDEX+MATCH`.

The two lookup keys are:
- **Series code**: from column D of the same row (e.g., `$D12` for row 12)
- **Year**: from row 10 of the same column (e.g., `H$10` for column H)

The lookup range is sheet `Data` rows 21:38. Determine from inspection whether the data is arranged with series codes in a column (use VLOOKUP or INDEX/MATCH) or in a row (use HLOOKUP or INDEX/MATCH). Build the formula accordingly.

IMPORTANT: Use absolute references for the data range on the `Data` sheet and mixed references for the series code (lock column) and year (lock row) so formulas can be filled across the block. Reference the `Data` sheet correctly (e.g., `Data!A21:Z38` or similar based on actual extent).

Example pattern (adapt based on actual layout):
- If data has series codes in column A and years in a header row: `=INDEX(Data!$B$21:$Z$38, MATCH($D12, Data!$A$21:$A$38, 0), MATCH(H$10, Data!$B$20:$Z$20, 0))`
- Adjust ranges based on actual inspection.

### Phase 2: Net SLA Buffer in H35:L40

Write a formula in each cell of H35:L40 that computes:
```
(Latency Budget Preserved - Latency Budget Consumed) / Covered Request Capacity * 100
```

From inspection, determine which of the three blocks (H12:L17, H19:L24, H26:L31) corresponds to:
- Latency Budget Preserved
- Latency Budget Consumed  
- Covered Request Capacity

The rows should align (row 35 uses data from rows 12, 19, 26; row 36 from 13, 20, 27; etc.). Write cell formulas like:
```
=(H12 - H19) / H26 * 100
```
(Adjust row references based on which block is which — verify from the labels in the workbook.)

### Phase 3: Summary statistics in H42:L47

For each column H through L, compute column-wise statistics over the 6 values in rows 35:40:
- Row 42: `=MIN(H35:H40)` (or whichever row is minimum — check labels)
- Row 43: `=MAX(H35:H40)`
- Row 44: `=MEDIAN(H35:H40)`
- Row 45: `=AVERAGE(H35:H40)`
- Row 46: `=PERCENTILE(H35:H40, 0.25)` (25th percentile)
- Row 47: `=PERCENTILE(H35:H40, 0.75)` (75th percentile)

IMPORTANT: Match the statistic to the correct row by reading the labels in column D (or nearby) for rows 42-47. The order might differ from what I listed above.

### Phase 4: Weighted mean in H50:L50

For each column H through L, write a `SUMPRODUCT` formula:
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```
This computes the weighted mean of the Net SLA Buffer percentages (H35:H40) weighted by Covered Request Capacity (H26:L31).

### Phase 5: Save and Validate
1. Save the workbook to `/root/output/result.xlsx` preserving all existing formatting.
2. Re-open the saved file and verify:
   - Cells H12:L17, H19:L24, H26:L31 contain formula strings (not plain values)
   - Cells H35:L40 contain formulas
   - Cells H42:L47 contain formulas
   - Cells H50:L50 contain formulas
   - Print a sample of formulas from each block to confirm correctness
   - Confirm no extra sheets were added
   - Confirm the `Data` sheet is unchanged

### Critical Notes
- All entries must be **Excel formulas** (strings starting with `=`), not Python-computed values.
- Do NOT use `data_only=True` when reading — you need to preserve existing formulas.
- Do NOT modify any cells outside the specified ranges.
- Do NOT add sheets, macros, VBA, or external links.
- Use `openpyxl` to read and write. When writing formulas, just assign the formula string to the cell's value (e.g., `ws['H12'] = '=INDEX(...)'`).
- Before writing formulas, carefully inspect the Data sheet layout to get ranges exactly right.

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