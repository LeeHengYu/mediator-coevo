# Task Instruction

Execute the following steps precisely to complete the weighted campus energy balance workbook task.

## Setup
1. `cp /root/data/workbook.xlsx /root/output/result.xlsx`
2. Install openpyxl if needed: `pip install openpyxl`

## Inspection
3. Open `/root/output/result.xlsx` with openpyxl (data_only=False to preserve formulas).
4. Read and print the following from sheet `Task`:
   - Row 10 (headers/years) — especially columns H through L (cols 8–12).
   - Column D for rows 12–17, 19–24, 26–31 (series codes).
   - Rows 35–40 labels, row 41 area, rows 42–47 labels.
   - Row 50 label.
   - Any existing content in the yellow cell ranges.
5. Read and print from sheet `Data`:
   - Row 20 or 21 headers and a few sample rows (rows 21–38) to understand the data layout: which row has which series code, which columns have which years.
   - Identify the exact structure: Is the data organized with series codes in a column and years across columns? Or years in a column and series codes across columns?

## Step 1: Populate lookup formulas in H12:L17, H19:L24, H26:L31
6. Based on the inspection, construct formulas using INDEX/MATCH (safest cross-version pattern).
   - For each cell in the three blocks (rows 12–17, 19–24, 26–31) and columns H–L:
     - The formula should look up the value from `Data!$21:$38` using:
       - The series code from column D of the current row on `Task` sheet (e.g., `$D12`)
       - The year from row 10 of the current column on `Task` sheet (e.g., `H$10`)
     - Pattern: `=INDEX(Data!$A$21:$XFD$38, MATCH($D12, Data!$A$21:$A$38, 0), MATCH(H$10, Data!$A$20:$XFD$20, 0))` — but adjust the exact ranges based on what you find in inspection. The key is:
       - The MATCH for the series code searches down a column in Data rows 21:38
       - The MATCH for the year searches across a header row in Data (likely row 20 or whichever row contains years)
       - INDEX returns the intersection
   - IMPORTANT: Determine the exact column that contains series codes on Data sheet and the exact row that contains year headers. Adjust ranges accordingly.
   - Write these formulas using openpyxl by assigning formula strings to cells. Use absolute references for the data range and mixed references ($D12 for row-relative column-absolute, H$10 for column-relative row-absolute).

## Step 2: Net renewable balance formulas in H35:L40 and statistics in H42:L47
7. For H35:L40 (6 campus rows, 5 year columns):
   - Determine which rows in the blocks above correspond to Renewable Generation, Grid Consumption, and Baseline Energy Demand. From the task description:
     - H12:L17 likely = one metric (e.g., Renewable Generation)
     - H19:L24 likely = another metric (e.g., Grid Consumption)  
     - H26:L31 likely = Baseline Energy Demand
   - Verify by reading the labels in the Task sheet (likely in columns A-G for each block).
   - Formula: `=(RenewableGen - GridConsumption) / BaselineEnergyDemand * 100`
   - E.g., if Renewable Gen is rows 12–17, Grid Consumption is rows 19–24, Baseline is rows 26–31: `=(H12-H19)/H26*100` for H35, etc. Map campus order carefully — verify that row 35 campus matches row 12, row 19, row 26 for the same campus.

8. For H42:L47 (column-wise statistics over H35:L40):
   - Determine which row is min, max, median, mean, 25th percentile, 75th percentile by reading labels in rows 42–47.
   - Use: `=MIN(H35:H40)`, `=MAX(H35:H40)`, `=MEDIAN(H35:H40)`, `=AVERAGE(H35:H40)`, `=PERCENTILE(H35:H40,0.25)`, `=PERCENTILE(H35:H40,0.75)` — assign to the correct rows based on labels.

## Step 3: Weighted mean in H50:L50
9. For each column H–L in row 50:
   - `=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)`
   - This computes the weighted mean of the net renewable balance percentages weighted by baseline energy demand.
   - Note: The task says use SUMPRODUCT. The formula should be: `=SUMPRODUCT(H35:H40, H26:H31)/SUM(H26:H31)`

## Save and Validate
10. Save the workbook (keep_vba=False is fine since no macros).
11. Re-open the file and print out a sample of the formulas you wrote to verify they are correctly stored.
12. Verify: no new sheets were added, no macros, no external links.

## Critical Notes
- Use openpyxl with data_only=False throughout to preserve and write formulas.
- Do NOT use data_only=True (that would lose existing formulas).
- When writing formulas, they must start with '=' and be plain strings assigned to cell.value.
- Preserve all existing formatting — do not clear cells, do not change fonts/colors/borders.
- Match campus ordering exactly between the three data blocks and the calculation block.
- Double-check that the series codes in column D match what's in the Data sheet.

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