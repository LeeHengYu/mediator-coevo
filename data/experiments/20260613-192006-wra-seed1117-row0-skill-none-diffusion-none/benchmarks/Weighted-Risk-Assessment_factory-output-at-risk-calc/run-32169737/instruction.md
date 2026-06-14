# Task Instruction

Complete the following task to update an Excel workbook with formulas.

## Phase 0: Inspect the workbook

1. Copy `/root/data/workbook.xlsx` to `/root/output/result.xlsx` first.
2. Using openpyxl, open `/root/output/result.xlsx` and inspect both sheets thoroughly:
   - On sheet `Task`: print rows 1-55 showing all columns A through M, with cell values AND formulas (use `data_only=False`). Pay special attention to:
     - Column D rows 12-31 (series codes)
     - Row 10 columns H-L (years)
     - The labels/headers around rows 12-17, 19-24, 26-31 to understand which block is which (Finished Output, Scrap And Rework, Rated Production Capacity)
     - Rows 35-50 to understand the output layout
     - Any existing formulas that reveal the expected pattern
   - On sheet `Data`: print rows 1-40 showing all columns to understand the data layout, especially rows 21-38. Note the structure: where series codes are, where years are, how data is organized (row-wise vs column-wise).
3. Print the fill colors of cells in the yellow ranges (H12:L17, H19:L24, H26:L31) to confirm they are the target cells.
4. Note any existing formulas elsewhere in the workbook that could serve as a pattern.

## Phase 1: Write lookup formulas (Step 1)

Based on your inspection, write formulas in cells H12:L17, H19:L24, and H26:L31 on sheet `Task`. Each formula must:
- Use one of these patterns: VLOOKUP+MATCH, HLOOKUP+MATCH, XLOOKUP+MATCH, or INDEX+MATCH
- Reference the series code in column D of the same row
- Reference the year in row 10 of the same column
- Look up data from sheet `Data` rows 21:38
- Use appropriate absolute/mixed references so the formula works correctly across the range

IMPORTANT: When writing formulas with openpyxl, use the `Translator` class or manually adjust references. Write each cell's formula individually if needed. Make sure sheet references use the correct syntax (e.g., `Data!` prefix). Use `$` signs appropriately for anchoring.

Choose INDEX+MATCH as it's the most flexible. The exact formula structure depends on how Data is laid out (determine this from inspection).

## Phase 2: Calculate Net Production Slack (Step 2)

In cells H35:L40, write formulas for each of the 6 plants:
```
(Finished Output - Scrap And Rework) / Rated Production Capacity * 100
```
- Finished Output values are in H12:L17
- Scrap And Rework values are in H19:L24  
- Rated Production Capacity values are in H26:L31
- Verify which block is which from your inspection!

In H42:L47, write formulas for column-wise statistics over H35:L40:
- Row 42: MIN
- Row 43: MAX
- Row 44: MEDIAN
- Row 45: AVERAGE (simple mean)
- Row 46: PERCENTILE (or PERCENTILE.INC) with 0.25
- Row 47: PERCENTILE (or PERCENTILE.INC) with 0.75
- Check the labels in column D/E/F/G for rows 42-47 to confirm the correct order!

## Phase 3: Weighted Mean (Step 3)

In H50:L50, write a SUMPRODUCT formula for each column:
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```
This computes the weighted mean of the Step 2 percentages weighted by Rated Production Capacity.

## Phase 4: Validate

1. Save the workbook.
2. Reopen it and verify:
   - All target cells contain formulas (not hardcoded values)
   - Formula syntax is correct (no #NAME? or #REF! when you can check)
   - The workbook has exactly 2 sheets: `Task` and `Data`
   - Print sample formulas from each block to confirm correctness
3. Ensure the file is saved at `/root/output/result.xlsx`.

## Critical Notes
- Do NOT add any new sheets, macros, VBA, or external links.
- Do NOT change existing formatting (fonts, colors, borders, etc.).
- Use `openpyxl` to read and write. When writing formulas, assign formula strings to cells (e.g., `cell.value = '=INDEX(...)'`).
- Adjust the exact row/column references based on what you find during inspection. The instruction's references to blocks may need verification.
- If the Data sheet has data organized with series codes in a column and years in a header row, INDEX+MATCH would be: `=INDEX(Data!$B$21:$F$38, MATCH($D12, Data!$A$21:$A$38, 0), MATCH(H$10, Data!$B$20:$F$20, 0))` — but adjust ranges based on actual layout.

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