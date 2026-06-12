# Task Instruction

Execute the following steps to produce /root/output/result.xlsx from /root/data/workbook.xlsx.

## Pre-work
1. `mkdir -p /root/output`
2. `pip install openpyxl` (if not already installed).

## Inspection
3. Open `/root/data/workbook.xlsx` with openpyxl (`data_only=False`) to preserve all existing formulas and formatting.
4. On sheet **Task**:
   - Read the year headers in row 10, columns H–L (cols 8–12). Record the exact year values (e.g., 2019, 2020, …).
   - Read the series codes in column D for rows 12–17 (block 1), 19–24 (block 2), 26–31 (block 3). Record each code string exactly.
   - Read the label in cell around row 35 area to confirm the Net capacity headroom block rows 35–40 and the six cluster names.
   - Read the stats labels in rows 42–47 (Min, Max, Median, Mean, 25th‰, 75th‰).
   - Read row 50 label to confirm "Regional Care Grid" weighted-mean row.
5. On sheet **Data**:
   - Inspect rows 21–38 to understand the layout: which column holds the series code, and which columns/rows hold year-indexed values.
   - Determine the exact column letter of the series-code key column and the row that holds year headers in the Data sheet.
   - Record the data range boundaries (e.g., Data!A20:S38 or similar).

## Formula Writing (use Python + openpyxl)
6. Write a Python script that:

### Step 1 – Lookup formulas in H12:L17, H19:L24, H26:L31
For each cell (row r, column c) in those three blocks:
- Let `series_cell` = the cell reference for the series code in column D of that row on sheet Task (e.g., `$D12`).
- Let `year_cell` = the cell reference for the year in row 10 of that column on sheet Task (e.g., `H$10`).
- Determine the Data sheet's lookup range. For example if Data has series codes in column A rows 21–38 and year headers in row 20 columns B onward, the formula pattern would be:
  `=INDEX(Data!B21:XX38, MATCH($D12,Data!$A$21:$A$38,0), MATCH(H$10,Data!$B$20:$XX$20,0))`
- Adjust the exact range references based on what you found in the inspection step. Use absolute references for the data range and mixed references for the series code column ($D) and year row ($10).
- Write the formula string into each cell.

### Step 2a – Net capacity headroom in H35:L40
For each cell (row r, column c) in H35:L40:
- The six clusters correspond positionally to the six rows in each lookup block.
- Let `avail` = cell from block 1 (H12:L17) at the matching position (same column, offset row).
- Let `occup` = cell from block 2 (H19:L24) at the matching position.
- Let `staffed` = cell from block 3 (H26:L31) at the matching position.
- Formula: `=(avail_cell - occup_cell) / staffed_cell * 100`
- Example for H35: `=(H12-H19)/H26*100`

### Step 2b – Summary statistics in H42:L47
For each column c (H through L):
- Row 42 (Min):    `=MIN(H35:H40)` (adjusted per column)
- Row 43 (Max):    `=MAX(H35:H40)`
- Row 44 (Median): `=MEDIAN(H35:H40)`
- Row 45 (Mean):   `=AVERAGE(H35:H40)`
- Row 46 (25th):   `=PERCENTILE(H35:H40,0.25)`
- Row 47 (75th):   `=PERCENTILE(H35:H40,0.75)`

### Step 3 – Weighted mean in H50:L50
For each column c (H through L):
- Formula: `=SUMPRODUCT(H35:H40,H26:H31)/SUM(H26:H31)` (adjusted per column)

7. After writing all formulas, verify by reading back a sample of cells to confirm the formula strings are present (not None).
8. Save the workbook to `/root/output/result.xlsx`.
9. Re-open the saved file and spot-check that:
   - Cells H12, L17, H19, L24, H26, L31 contain formula strings (start with '=').
   - Cells H35, L40 contain formula strings.
   - Cells H42, L47 contain formula strings.
   - Cell H50 and L50 contain formula strings.
   - No extra sheets were added.
   - The file loads without errors.

## Critical Constraints
- Do NOT use `data_only=True` when loading.
- Do NOT add sheets, macros, VBA, external links, or helper tabs.
- Do NOT modify any existing formatting, values outside the specified yellow cells, or sheet structure.
- Use absolute/mixed references correctly so formulas work across the row/column ranges.
- Double-check the Data sheet layout carefully before constructing formulas; a wrong range will cause None values in the verifier (this was the failure mode in the related hospital-bedflow task).

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