# Task Instruction

Execute the following steps precisely to complete the workbook task.

## Setup
1. `cp /root/data/workbook.xlsx /root/output/result.xlsx`
2. Install openpyxl if needed: `pip install openpyxl`
3. Open `/root/output/result.xlsx` with openpyxl, keeping formulas (data_only=False).

## Inspect the workbook first
4. Read and print the `Task` sheet structure:
   - Print all content in rows 1-55, columns A-M, to understand the layout.
   - Identify the series codes in column D for rows 12-17, 19-24, 26-31, and 35-40.
   - Identify the years in row 10 for columns H through L.
   - Note any existing formulas or values already present.
5. Read and print the `Data` sheet structure:
   - Print rows 1-40, focusing on rows 21-38 to understand the data layout.
   - Identify how series codes and years are arranged (which row/column has series codes, which has years).
   - Determine the exact cell references for the data range.

## Step 1: Populate H12:L17, H19:L24, H26:L31 with lookup formulas
6. Based on inspection, for each cell in the three blocks (H12:L17, H19:L24, H26:L31):
   - The formula should look up the value from the Data sheet rows 21:38.
   - Use INDEX/MATCH or VLOOKUP/MATCH pattern. The exact formula depends on the Data sheet layout.
   - The two inputs are: (a) the series code from column D of the current row, (b) the year from row 10 of the current column.
   
   Likely formula pattern (adjust after inspection):
   - If Data sheet has series codes in a column and years in a row header, use something like:
     `=INDEX(Data!$B$21:$Z$38, MATCH($D12, Data!$A$21:$A$38, 0), MATCH(H$10, Data!$B$20:$Z$20, 0))`
   - Adjust the exact ranges based on what you observe in the Data sheet.
   - Make sure the series code reference locks the column ($D12) and the year reference locks the row (H$10) so formulas copy correctly across the block.
   - Write formulas for ALL cells in the three blocks (6 rows × 5 columns × 3 blocks = 90 cells).

## Step 2: Net renewable balance in H35:L40 and statistics in H42:L47
7. For H35:L40, the formula for Net renewable balance is:
   `= (Renewable_Generation - Grid_Consumption) / Baseline_Energy_Demand * 100`
   
   Based on the three blocks:
   - H12:L17 likely corresponds to one metric (e.g., Renewable Generation)
   - H19:L24 likely corresponds to another metric (e.g., Grid Consumption)  
   - H26:L31 likely corresponds to another metric (e.g., Baseline Energy Demand)
   
   Verify which block is which by checking the labels in the Task sheet. Then for each cell, e.g., H35:
   `= (H12 - H19) / H26 * 100` (adjust row references based on which block maps to which metric)
   
   The six campuses in rows 35-40 should correspond to the same six campuses in rows 12-17, 19-24, 26-31. Match them by checking the labels.

8. For H42:L47, calculate column-wise statistics over H35:L40:
   - Row 42 (minimum): `=MIN(H35:H40)` for each column H-L
   - Row 43 (maximum): `=MAX(H35:H40)` for each column H-L
   - Row 44 (median): `=MEDIAN(H35:H40)` for each column H-L
   - Row 45 (simple mean): `=AVERAGE(H35:H40)` for each column H-L
   - Row 46 (25th percentile): `=PERCENTILE(H35:H40, 0.25)` for each column H-L
   - Row 47 (75th percentile): `=PERCENTILE(H35:H40, 0.75)` for each column H-L
   
   **IMPORTANT**: Check the labels in column D/E for rows 42-47 to verify which row is which statistic. Assign formulas to match the actual labels, not assumed order.

## Step 3: Weighted mean in H50:L50
9. For each column H through L in row 50:
   `=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)`
   This uses the Net renewable balance percentages as values and Baseline Energy Demand as weights.
   
   **IMPORTANT**: The instruction says to use SUMPRODUCT. The weighted mean formula should be:
   `=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)`

## Validation
10. After writing all formulas, re-read the workbook and print:
    - A sample of cells from each block to confirm formulas are present (not just values).
    - Verify no extra sheets were added.
    - Verify the file saves without errors.

11. Save the workbook to `/root/output/result.xlsx`.

## Critical Notes
- Use openpyxl to write Excel formulas as strings (e.g., cell.value = '=INDEX(...)').
- Do NOT use data_only=True when opening (that strips formulas).
- Preserve all existing formatting - do not clear or overwrite cells outside the specified ranges.
- The exact formula references MUST be adjusted based on what you observe in the inspection step. Do not blindly use the example references - inspect first, then construct correct formulas.
- When writing formulas with sheet references, use the exact sheet name. If the Data sheet name contains spaces, wrap in single quotes: `'Data'!`.
- After writing, save with `wb.save('/root/output/result.xlsx')`.
- Double-check that the statistics rows (42-47) match the actual labels in the workbook.

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