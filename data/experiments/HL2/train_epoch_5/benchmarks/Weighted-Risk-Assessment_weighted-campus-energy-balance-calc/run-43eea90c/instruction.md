# Task Instruction

## Task: Weighted Campus Energy Balance Calculation

You must update an Excel workbook with formulas and save the result. Follow these steps precisely.

### Step 0: Inspect the workbook
1. Copy `/root/data/workbook.xlsx` to `/root/output/result.xlsx`.
2. Use `openpyxl` to open `/root/output/result.xlsx` and inspect:
   - Sheet `Task`: Read rows 10-50, especially columns D and H-L. Note the series codes in column D for rows 12-17, 19-24, 26-31, 35-40, 42-47, 50. Note the years in row 10 columns H-L.
   - Sheet `Data`: Read rows 21-38 to understand the data layout (what's in each column/row, where series codes and years appear, how data is organized).
3. Print all of this information so you understand the exact structure before writing any formulas.

### Step 1: Populate lookup formulas in yellow cells

For each cell in the ranges `H12:L17`, `H19:L24`, and `H26:L31` on sheet `Task`:
- Write a spreadsheet formula (not a Python-computed value) that looks up data from sheet `Data` rows 21:38.
- Each formula must use TWO inputs: (a) the series code from column D of that row, and (b) the year from row 10 of that column.
- Use one of these lookup patterns: `VLOOKUP` with `MATCH`, `HLOOKUP` with `MATCH`, `XLOOKUP` with `MATCH`, or `INDEX` with `MATCH`.
- IMPORTANT: Examine the Data sheet layout carefully to determine which lookup pattern is appropriate. If data is arranged with series codes in a column and years in a header row, INDEX/MATCH or VLOOKUP/MATCH would work. If arranged differently, adapt accordingly.
- Use appropriate absolute/relative references so formulas can be understood. Lock row references for the year lookup and column references for the series code as needed.
- Make sure the formula references to the Data sheet use the correct sheet name syntax: `Data!` prefix.

### Step 2: Net renewable balance and statistics

For cells `H35:L40` (Net renewable balance for 6 campuses):
- The formula is: `(Renewable Generation - Grid Consumption) / Baseline Energy Demand * 100`
- Renewable Generation values are in `H12:L17`, Grid Consumption in `H19:L24`, Baseline Energy Demand in `H26:L31`.
- So for cell H35: `=(H12-H19)/H26*100`, and similarly for the rest of the 6×5 block.
- These must be spreadsheet formulas, not computed values.

For cells `H42:L47` (column-wise statistics over H35:L40):
- Row 42: Minimum → `=MIN(H35:H40)` for each column
- Row 43: Maximum → `=MAX(H35:H40)` for each column  
- Row 44: Median → `=MEDIAN(H35:H40)` for each column
- Row 45: Simple mean → `=AVERAGE(H35:H40)` for each column
- Row 46: 25th percentile → `=PERCENTILE(H35:H40,0.25)` for each column
- Row 47: 75th percentile → `=PERCENTILE(H35:H40,0.75)` for each column
- IMPORTANT: Check the labels in column D (or nearby) for rows 42-47 to confirm which row corresponds to which statistic. Adjust the row assignments if the labels differ from the order above.

### Step 3: Weighted mean for MCEC

For cells `H50:L50`:
- Use `SUMPRODUCT` to calculate the weighted mean.
- Values are the Net renewable balance percentages in `H35:H40` (for column H), weights are the Baseline Energy Demand in `H26:H31`.
- Formula: `=SUMPRODUCT(H35:H40,H26:H31)/SUM(H26:H31)` for each column H through L.
- These must be spreadsheet formulas.

### Step 4: Save and validate
1. Save the workbook to `/root/output/result.xlsx` preserving all existing formatting.
2. Re-open the saved file and verify:
   - Cells in H12:L17, H19:L24, H26:L31 contain formula strings (start with `=`).
   - Cells in H35:L40 contain formula strings.
   - Cells in H42:L47 contain formula strings.
   - Cells in H50:L50 contain formula strings.
   - No new sheets were added.
   - Print a sample of the formulas to confirm correctness.

### Critical constraints
- Do NOT add sheets, macros, VBA, external links, or helper tabs.
- Do NOT change existing formatting.
- All values in the specified ranges must be Excel formulas, not hardcoded values.
- Use `openpyxl` for all Excel operations. When writing formulas, just assign the formula string to the cell's value (e.g., `ws['H12'] = '=INDEX(...)'`).
- Make sure to create the output directory if it doesn't exist: `os.makedirs('/root/output', exist_ok=True)`

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