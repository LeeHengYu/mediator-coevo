# Task Instruction

## Task: Weighted Campus Energy Balance Calculation

You must update an Excel workbook with formulas and save it. Follow these steps precisely.

### Step 0: Inspect the workbook
1. Copy `/root/data/workbook.xlsx` to `/root/output/result.xlsx`.
2. Use `openpyxl` to open `/root/output/result.xlsx` and inspect:
   - Sheet `Task`: Read rows 10-50, focusing on columns D and H-L. Print the contents of row 10 (the year headers), column D rows 12-31 (series codes), and the structure of rows 35-50 (labels, any existing content).
   - Sheet `Data`: Read rows 21-38 to understand the data layout. Print the first row (headers) and a few data rows to understand the column structure. Identify where series codes are and where year data is.
3. Print all findings before proceeding.

### Step 1: Populate lookup formulas in H12:L17, H19:L24, H26:L31

Using `openpyxl`, write spreadsheet formulas (not computed values) into the yellow cells.

For each cell in these ranges, the formula must:
- Look up the value from sheet `Data` rows 21:38
- Use the series code from column D of the current row on sheet `Task`
- Use the year from row 10 of the current column on sheet `Task`
- Use one of the allowed patterns: INDEX/MATCH, VLOOKUP/MATCH, HLOOKUP/MATCH, or XLOOKUP/MATCH

IMPORTANT: When writing formulas with openpyxl, you must understand the exact layout of the Data sheet first. The formula pattern should be something like:
- `=INDEX(Data!$B$21:$XX$38, MATCH($D12, Data!$A$21:$A$38, 0), MATCH(H$10, Data!$B$20:$XX$20, 0))` 
- Adjust the exact cell references based on what you find in the Data sheet inspection.

Make sure:
- The row reference for the series code (column D) uses `$D` with the row number (absolute column, relative row)
- The column reference for the year (row 10) uses the column letter with `$10` (relative column, absolute row)
- The Data sheet ranges are fully anchored with `$`

### Step 2: Net Renewable Balance and Statistics (H35:L40, H42:L47)

For H35:L40 — Net renewable balance for six campuses:
- Formula: `(Renewable Generation - Grid Consumption) / Baseline Energy Demand * 100`
- Renewable Generation values are in H12:L17
- Grid Consumption values are in H19:L24  
- Baseline Energy Demand values are in H26:L31
- So for cell H35: `=(H12-H19)/H26*100`, and similarly for the rest of the 6×5 block.
- Use relative references so the formula naturally extends across the block.

For H42:L47 — Column-wise statistics over H35:L40:
- H42: `=MIN(H$35:H$40)` (minimum)
- H43: `=MAX(H$35:H$40)` (maximum)
- H44: `=MEDIAN(H$35:H$40)` (median)
- H45: `=AVERAGE(H$35:H$40)` (simple mean)
- H46: `=PERCENTILE(H$35:H$40,0.25)` (25th percentile)
- H47: `=PERCENTILE(H$35:H$40,0.75)` (75th percentile)
- Check the labels in column D or nearby columns for rows 42-47 to confirm the correct order of statistics. Adjust the row assignments if the labels indicate a different order.

### Step 3: Weighted Mean in H50:L50

For H50: `=SUMPRODUCT(H35:H40,H26:H31)/SUM(H26:H31)`
- This uses Net renewable balance (H35:H40) as values and Baseline Energy Demand (H26:H31) as weights.
- Repeat pattern for columns I through L.

### Step 4: Save and Validate
1. Save the workbook (keep formatting intact — do NOT use `data_only` mode, do NOT delete sheets).
2. Re-open the saved file and verify:
   - Cells H12, L17, H19, L24, H26, L31 contain formula strings (not None or numbers)
   - Cells H35, L40 contain formulas
   - Cells H42, H47 contain formulas
   - Cell H50 contains a SUMPRODUCT formula
3. Print representative formulas from each section to confirm correctness.

### Critical Constraints
- Do NOT use `data_only=True` when opening for writing.
- Do NOT add new sheets, macros, VBA, or external links.
- Do NOT modify existing formatting.
- Preserve all existing content outside the target cells.
- Write Excel formula strings (starting with `=`), not Python-computed values.
- Save to `/root/output/result.xlsx`.

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