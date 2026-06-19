# Task Instruction

Complete the following task to update an Excel workbook with formulas.

## Phase 0: Inspect the workbook structure

1. Copy the workbook: `cp /root/data/workbook.xlsx /root/output/result.xlsx`
2. Use Python with openpyxl to inspect `/root/output/result.xlsx`:
   - Print the sheet names.
   - For the `Task` sheet: print all cell values and any cell formatting/fill colors for rows 1-55, columns A-M. Pay special attention to:
     - Row 10 (years row)
     - Column D rows 12-31 (series codes)
     - Cells H12:L17, H19:L24, H26:L31 (the yellow cells to fill with lookup formulas)
     - Cells H35:L40 (Net production slack)
     - Cells H42:L47 (summary statistics)
     - Cell range H50:L50 (weighted mean)
     - Any labels in column A-G that identify what each row block represents (e.g., which block is Finished Output, Scrap And Rework, Rated Production Capacity)
   - For the `Data` sheet: print all cell values for rows 1-40, focusing on rows 21-38. Identify the structure: which row/column holds series codes, which holds years, and how data is organized.
3. Print all findings clearly before proceeding.

## Phase 1: Populate lookup formulas in H12:L17, H19:L24, H26:L31

Based on the inspection, write spreadsheet formulas (not Python-computed values) into the yellow cells. Each formula must:
- Use the series code from column D of the SAME row on the `Task` sheet
- Use the year from row 10 of the SAME column on the `Task` sheet
- Look up the value from the `Data` sheet rows 21:38
- Use one of these patterns: INDEX+MATCH (recommended for reliability), VLOOKUP+MATCH, HLOOKUP+MATCH, or XLOOKUP+MATCH

IMPORTANT: When writing formulas with openpyxl, set the cell value to a string starting with '=' (e.g., cell.value = '=INDEX(Data!$B$21:$Z$38,MATCH(D12,Data!$A$21:$A$38,0),MATCH(H$10,Data!$B$20:$Z$20,0))'). Adjust the exact ranges based on what you find in the inspection. Make sure:
- Row references for the Data sheet lookup range cover rows 21:38
- Column references are appropriate for the data layout
- The series code reference uses an absolute row-relative column (like $D12) so column D is always used
- The year reference uses a relative column-absolute row (like H$10) so row 10 is always used
- Apply the same formula pattern across all cells in H12:L17, H19:L24, and H26:L31, adjusting only the row reference naturally

## Phase 2: Calculate Net production slack in H35:L40

Identify which row blocks correspond to:
- Finished Output (likely H12:L17)
- Scrap And Rework (likely H19:L24)
- Rated Production Capacity (likely H26:L31)

Confirm this from the labels found during inspection. The formula for each cell in H35:L40 is:
`=(FinishedOutput - ScrapAndRework) / RatedProductionCapacity * 100`

For example, H35 might be: `=(H12-H19)/H26*100` (mapping the first plant's row from each block). Adjust row references for each of the 6 plants.

Then in H42:L47, enter column-wise summary formulas:
- Row 42: MIN of H35:H40 (and similarly for columns I-L)
- Row 43: MAX of H35:H40
- Row 44: MEDIAN of H35:H40
- Row 45: AVERAGE of H35:H40
- Row 46: PERCENTILE(H35:H40, 0.25) or PERCENTILE.INC(H35:H40, 0.25)
- Row 47: PERCENTILE(H35:H40, 0.75) or PERCENTILE.INC(H35:H40, 0.75)

Check the labels in column A-G for rows 42-47 to confirm which row is which statistic. Match the formula to the label.

## Phase 3: Weighted mean in H50:L50

For each column H through L, enter a SUMPRODUCT formula:
`=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)`

This computes the weighted mean of the Net production slack percentages (H35:H40) weighted by Rated Production Capacity (H26:H31).

## Phase 4: Save and validate

1. Save the workbook to `/root/output/result.xlsx` using `workbook.save()`.
2. Re-open the file and verify:
   - All cells in H12:L17, H19:L24, H26:L31 contain formula strings (start with '=')
   - All cells in H35:L40 contain formula strings
   - All cells in H42:L47 contain formula strings
   - All cells in H50:L50 contain formula strings
   - No new sheets were added
   - Print the formulas for a sample of cells to confirm correctness
3. Verify the file exists: `ls -la /root/output/result.xlsx`

## Critical constraints
- Do NOT add any new sheets, macros, VBA, external links, or helper tabs.
- Do NOT change existing formatting (do not modify fonts, fills, borders, etc.).
- All values in the target cells must be Excel formulas, not hardcoded numbers.
- Use openpyxl for all Excel manipulation.
- The final file must be saved at `/root/output/result.xlsx`.

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