# Task Instruction

Execute the following steps precisely to complete the hospital capacity risk assessment workbook.

## Setup
1. `cp /root/data/workbook.xlsx /root/output/result.xlsx`
2. Install openpyxl if needed: `pip install openpyxl`

## Inspection
3. Open `/root/output/result.xlsx` with openpyxl (data_only=False to preserve formulas).
4. Read sheet `Task`:
   - Print the contents of rows 10-50, columns D through L, to understand the layout: series codes in column D, years in row 10, yellow target cells, and any existing content.
   - Specifically print D12:D17, D19:D24, D26:D31 to see the series codes for each block.
   - Print H10:L10 to see the year headers.
5. Read sheet `Data`:
   - Print rows 21-38 completely (all columns) to understand the data layout: which row contains headers, how series codes and years are arranged, and the exact cell references.
   - Determine the data range: first row, last row, first column, last column.
   - Note whether Data is organized with series codes in a column and years across columns, or vice versa.

## Planning Formulas
6. Based on inspection, determine:
   - On sheet `Data`, which column contains the series code (the lookup key matching column D on Task).
   - On sheet `Data`, which row contains the year headers (matching row 10 on Task).
   - The exact data range for the lookup (rows 21:38, but identify the column span).

## Step 1: Populate H12:L17, H19:L24, H26:L31 with lookup formulas
7. For each cell in the three blocks (H12:L17, H19:L24, H26:L31), write a formula using INDEX/MATCH pattern:
   - The formula should look up the series code from column D of the current row in the Data sheet's series code column (within rows 21:38), and the year from row 10 of the Task sheet in the Data sheet's year header row.
   - Use absolute references for the Data range and mixed references appropriately so the formula works for each cell.
   - Example pattern (adjust based on actual Data layout): `=INDEX(Data!$B$21:$Z$38,MATCH($D12,Data!$A$21:$A$38,0),MATCH(H$10,Data!$B$20:$Z$20,0))`
   - Adjust the column/row references based on what you find in the inspection step. The key is: MATCH on series code in the appropriate column of Data rows 21:38, MATCH on year in the appropriate row of Data, then INDEX into the data block.
   - Write these formulas using openpyxl by assigning the formula string to each cell.

## Step 2: Net capacity headroom in H35:L40
8. For each cell in H35:L40, write a formula:
   `=(H12 - H19) / H26 * 100`
   where H12 corresponds to Available Care Slots (rows 12-17), H19 to Occupied Care Slots (rows 19-24), and H26 to Staffed Bed Capacity (rows 26-31). Adjust row references per the row offset within each block (row 35 uses rows 12, 19, 26; row 36 uses rows 13, 20, 27; etc.).
   - Specifically: for row 35+i (i=0..5), the formula in column col is: `=(col_letter(row 12+i) - col_letter(row 19+i)) / col_letter(row 26+i) * 100`

9. For H42:L47, write summary statistics formulas over H35:L40 (each column):
   - H42 (minimum): `=MIN(H35:H40)`
   - H43 (maximum): `=MAX(H35:H40)`
   - H44 (median): `=MEDIAN(H35:H40)`
   - H45 (mean): `=AVERAGE(H35:H40)`
   - H46 (25th percentile): `=PERCENTILE(H35:H40,0.25)`
   - H47 (75th percentile): `=PERCENTILE(H35:H40,0.75)`
   - Check the labels in column D or G for rows 42-47 to confirm the correct order of min/max/median/mean/25th/75th. Adjust assignment accordingly.

## Step 3: Weighted mean in H50:L50
10. For each column H through L in row 50, write:
    `=SUMPRODUCT(H35:H40,H26:H31)/SUM(H26:H31)`
    This computes the weighted mean of the net capacity headroom percentages using Staffed Bed Capacity as weights.

## Validation
11. Re-read the workbook and print all cells you wrote to confirm they contain formula strings (not values or None).
12. Verify no new sheets were added.
13. Save and close.

## Critical Notes
- Use openpyxl with `load_workbook(filename, data_only=False)` to preserve existing formulas.
- Do NOT use `data_only=True`.
- When writing formulas, they must start with `=`.
- Do NOT change any existing formatting, cell styles, or content outside the target ranges.
- Do NOT add sheets, macros, VBA, or external links.
- The inspection in steps 4-5 is critical. Print enough of the Data sheet to understand the exact layout before writing any formulas. If the Data layout differs from assumptions, adapt the formulas accordingly.
- For the summary stats rows 42-47, read the labels carefully to determine which row gets which function. Do not assume the order.

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