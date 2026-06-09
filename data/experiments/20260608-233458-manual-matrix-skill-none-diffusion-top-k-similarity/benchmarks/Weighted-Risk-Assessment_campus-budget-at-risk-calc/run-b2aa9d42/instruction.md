# Task Instruction

Execute the following steps to produce /root/output/result.xlsx.

## Phase 0 – Inspect the workbook
1. `mkdir -p /root/output`
2. Open `/root/data/workbook.xlsx` with openpyxl (data_only=False). List sheet names.
3. On sheet `Task`: print rows 10-11 (to see year headers in H-L), print column D for rows 12-31 (series codes), print rows 35-50 so you understand the layout of Net budget buffer, summary stats, and weighted mean rows.
4. On sheet `Data`: print rows 21-38 with their column headers (row 20 or whichever row holds the header) so you know the lookup table structure (orientation, key column/row).
5. Based on what you see, decide whether the Data table is arranged with series codes in a column (VLOOKUP-friendly) or in a row (HLOOKUP-friendly). Note the exact range.

## Phase 1 – Lookup formulas (H12:L31)
For every cell in H12:L17, H19:L24, H26:L31, write an INDEX-MATCH formula that:
- Uses the series code from column D of the same row as the row-match key.
- Uses the year from row 10 of the same column as the column-match key.
- References the Data sheet rows 21:38 as the source.
- Use absolute references on the Data range and mixed references on D (row) and row 10 (column) so the formula can be filled across the block.

Concrete pattern (adjust column letters/row numbers after inspection):
```
=INDEX(Data!$B$21:$XX$38, MATCH($D12, Data!$A$21:$A$38, 0), MATCH(H$10, Data!$B$20:$XX$20, 0))
```
Adjust `$XX` and header row based on actual data extent found in Phase 0.

## Phase 2 – Net budget buffer & summary statistics (H35:L47)

### H35:L40 – Net budget buffer
Formula per cell: `=(CommittedFunding - OperatingSpend) / ApprovedBudgetBase * 100`
where CommittedFunding, OperatingSpend, and ApprovedBudgetBase are the corresponding cells from the three lookup blocks (rows 12-17, 19-24, 26-31). Identify which block is which from the row labels in column B or C.

### H42:L47 – Column-wise summary statistics
For each column (H through L):
- Row 42 (Min):    `=MIN(H35:H40)`
- Row 43 (Max):    `=MAX(H35:H40)`
- Row 44 (Median): `=MEDIAN(H35:H40)`
- Row 45 (Mean):   `=AVERAGE(H35:H40)`
- Row 46 (25th %): `=PERCENTILE.INC(H35:H40, 0.25)`
- Row 47 (75th %): `=PERCENTILE.INC(H35:H40, 0.75)`

**Important:** Use `PERCENTILE.INC` (not bare `PERCENTILE`) to avoid #NAME? errors. This is confirmed by cross-task feedback where `PERCENTILE` alone caused failures.

## Phase 3 – Weighted mean (H50:L50)
For each column (H through L):
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```
This computes the weighted mean of Net budget buffer using Approved Budget Base as weights.

## Phase 4 – Save and validate
1. Save the workbook to `/root/output/result.xlsx`.
2. Re-open the saved file and spot-check:
   - H12 contains a formula (not a plain value).
   - H35 contains a formula referencing the three blocks.
   - H46 contains `PERCENTILE.INC`.
   - H50 contains `SUMPRODUCT`.
3. Print a few formula strings to confirm correctness.

## Constraints
- Do not add new sheets, macros, VBA, external links, or helper tabs.
- Do not alter existing formatting, only write formulas into the specified cells.
- Use openpyxl for all operations.

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