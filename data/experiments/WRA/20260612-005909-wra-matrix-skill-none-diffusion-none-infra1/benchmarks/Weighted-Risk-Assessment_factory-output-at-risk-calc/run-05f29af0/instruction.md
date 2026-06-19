# Task Instruction

Execute the following steps carefully and in order.

## Phase 0 – Copy the workbook
```bash
cp /root/data/workbook.xlsx /root/output/result.xlsx
```

## Phase 1 – Inspect the workbook structure
Open `/root/output/result.xlsx` with openpyxl and print:
1. **Task sheet:**
   - Row 10 (H10:L10) – the year headers.
   - Column D rows 12–17, 19–24, 26–31 – the series codes.
   - Row 35–40 column D or any labels – to understand what goes in the Net production slack block.
   - Rows 42–47 column D or E or F – the stat labels (min, max, median, mean, 25th, 75th).
   - Row 50 – the weighted-mean row label.
   - H26:L31 current content (should be empty yellow cells for Rated Production Capacity lookup).
2. **Data sheet:**
   - Rows 21–38 fully: print every cell from column A (or 1) through column Q (or however wide the data goes). We need to see where the series codes live (which column), where the year headers live (which row), and the shape of the data matrix.
   - Also print row 20 or any header row above row 21 to see column headers.

Print everything clearly with row and column indices. Do NOT write any formulas yet.

## Phase 2 – Plan the formulas
Based on the inspection, determine:
- Which column in Data holds the series codes (call it `code_col`, e.g., column B).
- Which row in Data holds the year headers (call it `year_row`, e.g., row 21).
- The exact data matrix range for the values.
- Confirm the three blocks of 6 series codes each in Task column D (rows 12-17, 19-24, 26-31) and verify they exist in Data.

Then construct INDEX/MATCH formulas of this pattern:
```
=INDEX(Data!<value_range>, MATCH($D12, Data!<code_column_range>, 0), MATCH(H$10, Data!<year_row_range>, 0))
```
Adjust ranges based on actual inspection. Use `$D12` (absolute column) and `H$10` (absolute row) style references so the formula can be filled across the 5-column × 6-row blocks.

## Phase 3 – Write lookup formulas (Step 1)
Using openpyxl, write the INDEX/MATCH formula into every cell in:
- H12:L17 (block 1 – e.g., Finished Output)
- H19:L24 (block 2 – e.g., Scrap And Rework)
- H26:L31 (block 3 – e.g., Rated Production Capacity)

For each block, iterate row by row, column by column, adjusting the cell references in the formula string accordingly. Use absolute references as described.

## Phase 4 – Write Net production slack formulas (Step 2)
In H35:L40, for each of the 6 plants (rows) and 5 years (columns), write:
```
=(H12-H19)/H26*100
```
(adjusting row/col references for each cell, matching the same plant row offset and year column).

For H42:L47, write column-wise statistics over H35:L40:
- Row 42: MIN (e.g., `=MIN(H35:H40)`)
- Row 43: MAX (e.g., `=MAX(H35:H40)`)
- Row 44: MEDIAN (e.g., `=MEDIAN(H35:H40)`)
- Row 45: AVERAGE (e.g., `=AVERAGE(H35:H40)`)
- Row 46: 25th percentile – use `=_xlfn.PERCENTILE.INC(H35:H40,0.25)`
- Row 47: 75th percentile – use `=_xlfn.PERCENTILE.INC(H35:H40,0.75)`

**CRITICAL:** Use the `_xlfn.` prefix for PERCENTILE.INC to avoid #NAME? errors.

Check the actual stat labels in column D/E/F/G of rows 42-47 to confirm the correct order (min/max/median/mean/25th/75th). Adjust row assignments if the order differs.

## Phase 5 – Write weighted mean (Step 3)
In H50:L50, write a SUMPRODUCT formula:
```
=SUMPRODUCT(H35:H40,H26:H31)/SUM(H26:H31)
```
(adjusting column letter for each of the 5 columns H through L).

## Phase 6 – Save and verify
Save the workbook. Then reopen it and print the values/formulas in a sample of cells (e.g., H12, L17, H35, L40, H42, H47, H50, L50) to confirm formulas were written correctly. Check that no cells are None or have obvious errors.

Do NOT add any new sheets, macros, VBA, external links, or helper tabs. Do NOT change any existing formatting.

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