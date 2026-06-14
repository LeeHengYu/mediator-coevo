# Task Instruction

Execute the following steps carefully to complete the hospital capacity workbook task.

## Setup
1. `cp /root/data/workbook.xlsx /root/output/result.xlsx`
2. Install openpyxl if needed: `pip install openpyxl`

## Inspection
3. Open `/root/output/result.xlsx` with openpyxl (data_only=False) and inspect:
   - Sheet names (confirm `Task` and `Data` exist)
   - On `Task` sheet: read row 10 to see the year headers in columns H through L (columns 8-12). Print them.
   - Read column D rows 12-17, 19-24, 26-31 to see the series codes. Print them.
   - Read column D rows 35-40 to see cluster names or codes. Print them.
   - Read rows 42-47 column A-G to see labels for min/max/median/mean/percentiles. Print them.
   - Read row 50 columns A-G to see the weighted mean label. Print it.
   - On `Data` sheet: read rows 21-38, printing all content (columns A through at least column Z or wherever data ends) to understand the data layout — identify which column has series codes, which row has years, and where the values are.
   - Also check what's currently in the yellow cells (H12:L17 etc.) — are they empty or do they have values?

## Step 1: Populate lookup formulas in H12:L17, H19:L24, H26:L31
4. Based on the inspection, determine the exact layout of the Data sheet rows 21-38. Identify:
   - Which column contains the series/indicator codes
   - Which row contains the year headers
   - The data range for VLOOKUP or INDEX/MATCH

5. Using openpyxl, write formulas into each cell in the three blocks. For each cell at row `r`, column `c` (H=8, I=9, J=10, K=11, L=12):
   - The series code reference is `$D{r}` (column D of the current row on Task sheet)
   - The year reference is the cell in row 10 at column `c`, e.g., `H$10`, `I$10`, etc.
   - Use an INDEX/MATCH/MATCH pattern referencing Data!rows21:38. The exact formula depends on the Data sheet layout discovered in step 3.
   - Example pattern (adjust based on actual layout): `=INDEX(Data!$B$22:$Z$38, MATCH($D12, Data!$A$22:$A$38, 0), MATCH(H$10, Data!$B$21:$Z$21, 0))`
   - Adjust column/row references to match the actual data range discovered.

6. After writing all formulas, re-read a few cells to confirm the formulas are stored correctly.

## Step 2: Net capacity headroom in H35:L40 and summary stats in H42:L47
7. For H35:L40, determine which rows contain "Available Care Slots", "Occupied Care Slots", and "Staffed Bed Capacity" for each of the six clusters. These should correspond to the three blocks H12:L17 (one metric), H19:L24 (another metric), H26:L31 (third metric). Check the labels in column A-G for rows 12-17, 19-24, 26-31 to identify which block is which metric.

8. Write formulas for each cell in H35:L40. For cluster i (i=0..5), the formula is:
   `=(AvailableCareSlots_cell - OccupiedCareSlots_cell) / StaffedBedCapacity_cell * 100`
   where the three cells come from the corresponding row in each of the three blocks (row 12+i, 19+i, 26+i) at the same column.

9. For H42:L47, write column-wise summary statistics over H35:L40. Based on the labels discovered in inspection:
   - MIN: `=MIN(H35:H40)` (for column H, similarly for I-L)
   - MAX: `=MAX(H35:H40)`
   - MEDIAN: `=MEDIAN(H35:H40)`
   - AVERAGE: `=AVERAGE(H35:H40)`
   - 25th percentile: `=PERCENTILE(H35:H40, 0.25)`
   - 75th percentile: `=PERCENTILE(H35:H40, 0.75)`
   Match the row order to the labels found in the inspection.

## Step 3: Weighted mean in H50:L50
10. Write SUMPRODUCT-based weighted mean formulas. For each column c (H-L):
    `=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)` (adjusting column letter)
    This uses the Step 2 percentages as values and Staffed Bed Capacity as weights.

## Save and Validate
11. Save the workbook (keep_vba=False is fine since no macros).
12. Re-open the file and print all formulas in the modified cells to confirm they look correct.
13. Also open with data_only=True (after saving) — note that openpyxl won't calculate formulas, but confirm the formulas are present.

## Critical Notes
- Do NOT modify any existing formatting, styles, or cell values outside the specified ranges.
- Do NOT add new sheets, macros, VBA, external links, or helper tabs.
- When writing formulas with openpyxl, prefix them with `=`.
- Use absolute/mixed references appropriately: lock the series code column ($D) and the year row ($10) as shown.
- The Data sheet reference ranges must use absolute references ($ signs) so formulas work across all cells.
- Double-check the Data sheet layout carefully before writing any formulas — the exact row/column ranges are critical.

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