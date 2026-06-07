# Task Instruction

Complete the following task to update an Excel workbook with formulas.

## Setup
1. Copy `/root/data/workbook.xlsx` to `/root/output/result.xlsx` first, then work on the copy.
2. Install `openpyxl` if needed: `pip install openpyxl`.
3. Before writing ANY formulas, thoroughly inspect the workbook structure:
   - Read sheet `Task`: print all non-empty cells in rows 1-55, columns A-L. Pay special attention to:
     - Row 10 (contains years in columns H-L)
     - Column D rows 12-17, 19-24, 26-31 (contains series codes)
     - Row labels in column A or B for rows 12-17, 19-24, 26-31 (to understand which block is Latency Budget Preserved, Latency Budget Consumed, Covered Request Capacity)
     - Rows 35-50 to understand the output layout
   - Read sheet `Data`: print all non-empty cells in rows 18-40, columns A-Z (at minimum). Identify the structure of rows 21-38 — what's in each column, how series codes and years are arranged.
   - Print the exact content of row 10 on Task sheet to see the year headers.
   - Print column D for rows 12-31 on Task sheet to see the series codes.

## Step 1: Lookup Formulas in H12:L17, H19:L24, H26:L31

After inspecting the data layout on the `Data` sheet (rows 21:38), write spreadsheet formulas (not computed values) in the yellow cells. Each formula must:
- Use the series code from column D of the same row on `Task`
- Use the year from row 10 of the same column on `Task`
- Look up the value from `Data!$21:$38` (or the appropriate absolute range)
- Use one of these patterns: INDEX+MATCH, VLOOKUP+MATCH, HLOOKUP+MATCH, or XLOOKUP+MATCH

IMPORTANT: You must understand the Data sheet layout to choose the right lookup approach:
- If Data rows 21:38 have series codes in a column and years across columns, use INDEX(MATCH for row, MATCH for column) or VLOOKUP with MATCH for column index.
- If Data has a different layout, adapt accordingly.
- Use absolute references for the lookup range (e.g., `Data!$A$21:$Z$38`) so formulas can be filled across rows and columns.
- The MATCH for the year should reference the year in row 10 with the column fixed appropriately, and the MATCH for the series code should reference column D with the row fixed appropriately.

Write these as Excel formula strings in openpyxl (e.g., `cell.value = '=INDEX(Data!$B$21:$Z$38,MATCH($D12,Data!$A$21:$A$38,0),MATCH(H$10,Data!$B$20:$Z$20,0))'`). Adjust column/row references based on what you actually find in the data.

## Step 2: Net SLA Buffer in H35:L40 and Statistics in H42:L47

Based on your inspection, identify which of the three blocks (H12:L17, H19:L24, H26:L31) corresponds to:
- **Latency Budget Preserved**
- **Latency Budget Consumed**  
- **Covered Request Capacity**

The row labels on the Task sheet should tell you this.

In H35:L40, write formulas for each of the 6 services (rows) and 5 years (columns):
`= (Latency_Budget_Preserved - Latency_Budget_Consumed) / Covered_Request_Capacity * 100`

Reference the corresponding cells from the blocks above (e.g., if Preserved is rows 12-17, Consumed is rows 19-24, Capacity is rows 26-31, then H35 = `=(H12-H19)/H26*100`).

In H42:L47, write column-wise aggregate formulas over H35:L40:
- Row 42: MIN (e.g., `=MIN(H35:H40)`)
- Row 43: MAX (e.g., `=MAX(H35:H40)`)
- Row 44: MEDIAN (e.g., `=MEDIAN(H35:H40)`)
- Row 45: AVERAGE (e.g., `=AVERAGE(H35:H40)`)
- Row 46: PERCENTILE or PERCENTILE.INC with 0.25 (e.g., `=PERCENTILE(H35:H40,0.25)`)
- Row 47: PERCENTILE or PERCENTILE.INC with 0.75 (e.g., `=PERCENTILE(H35:H40,0.75)`)

Check the row labels on the Task sheet to confirm which row is which statistic.

## Step 3: Weighted Mean in H50:L50

In H50:L50, write SUMPRODUCT formulas for the weighted mean:
`=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)`

This uses the Net SLA buffer percentages (H35:H40) as values and the Covered Request Capacity block (H26:H31) as weights. Adjust the Covered Request Capacity range references based on your actual findings.

## Important Rules
- Write Excel formula strings, NOT computed Python values.
- Do NOT add new sheets, macros, VBA, external links, or helper tabs.
- Do NOT alter existing formatting. When writing with openpyxl, make sure to preserve existing styles. Open with `load_workbook` and only set `.value` on the target cells.
- After writing all formulas, save to `/root/output/result.xlsx`.
- After saving, re-open the file and print the formula content of a sample of cells (e.g., H12, H19, H26, H35, H42, H50) to verify they contain formula strings starting with `=`.
- Verify the row/column mapping is correct by cross-checking a few cells against the Data sheet structure.

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