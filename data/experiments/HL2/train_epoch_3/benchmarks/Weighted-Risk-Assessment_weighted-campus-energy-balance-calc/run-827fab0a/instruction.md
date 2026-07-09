# Task Instruction

## Task: Weighted Campus Energy Balance Calculation

You must update an Excel workbook with formulas and save the result. Follow these steps precisely.

### Step 0: Inspect the workbook
1. Copy `/root/data/workbook.xlsx` to `/root/output/result.xlsx`.
2. Use `openpyxl` to open `/root/output/result.xlsx` and inspect:
   - Sheet `Task`: Read rows 10-50, columns D through L. Print cell values to understand the layout: what's in column D (series codes), what's in row 10 (years), what yellow cells need formulas.
   - Sheet `Data`: Read rows 21-38 to understand the data structure — identify where series codes are, where years are as column headers, and where values live.
3. Print out enough detail to understand the exact layout before writing any formulas.

### Step 1: Populate lookup formulas in H12:L17, H19:L24, H26:L31

For each cell in the three blocks (H12:L17, H19:L24, H26:L31):
- The cell's row has a series code in column D of that same row.
- The cell's column corresponds to a year in row 10 of that same column.
- Write a formula that looks up the value from sheet `Data` rows 21:38.
- Use one of the allowed patterns: `INDEX/MATCH`, `VLOOKUP/MATCH`, `HLOOKUP/MATCH`, or `XLOOKUP/MATCH`.

IMPORTANT: You need to understand the Data sheet layout first:
- If Data has series codes in a column and years across a row, use INDEX(MATCH, MATCH) or similar 2D lookup.
- Anchor references appropriately (use $ signs) so the formula correctly picks up the series code from column D and the year from row 10.
- The lookup range on Data should cover rows 21:38 as specified.

Write actual Excel formula strings into the cells (do NOT write computed values). Use `openpyxl` and set `cell.value = '=FORMULA...'`.

### Step 2: Net renewable balance in H35:L40 and statistics in H42:L47

For H35:L40 (6 campuses × 5 years):
- Formula: `(Renewable Generation - Grid Consumption) / Baseline Energy Demand * 100`
- Renewable Generation values are in H12:L17
- Grid Consumption values are in H19:L24  
- Baseline Energy Demand values are in H26:L31
- So for cell H35: `=(H12-H19)/H26*100`, H36: `=(H13-H20)/H27*100`, etc.
- Verify the row mapping by checking that each campus lines up correctly across all three blocks and the result block.

For H42:L47 (column-wise statistics over H35:L40):
- Check what labels are in column D (or nearby) for rows 42-47 to determine which statistic goes where.
- The six statistics are: minimum, maximum, median, simple mean, 25th percentile, 75th percentile.
- Use Excel functions: MIN, MAX, MEDIAN, AVERAGE, PERCENTILE (or PERCENTILE.INC) accordingly.
- Each formula should reference the column range, e.g., for column H: `=MIN(H35:H40)`, `=MAX(H35:H40)`, etc.
- Match the statistic to the correct row based on the labels you find.

### Step 3: Weighted mean in H50:L50

For each cell in H50:L50:
- Use SUMPRODUCT to calculate weighted mean.
- Values = the Net renewable balance percentages for that column (e.g., H35:H40)
- Weights = the Baseline Energy Demand for that column (e.g., H26:H31)
- Formula: `=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)`

### Step 4: Save and verify
1. Save the workbook to `/root/output/result.xlsx`.
2. Re-open it and verify:
   - Cells H12:L17, H19:L24, H26:L31 contain formula strings (not plain values).
   - Cells H35:L40 contain formula strings.
   - Cells H42:L47 contain formula strings.
   - Cells H50:L50 contain formula strings.
   - Print a sample of formulas to confirm correctness.
3. Confirm no new sheets were added, no macros, no external links.
4. Confirm the file is saved at `/root/output/result.xlsx`.

### Critical constraints
- Do NOT add sheets, macros, VBA, external links, or helper tabs.
- Do NOT change existing formatting.
- All populated cells must contain Excel formulas, not hardcoded values.
- The lookup formulas MUST use one of: VLOOKUP+MATCH, HLOOKUP+MATCH, XLOOKUP+MATCH, or INDEX+MATCH.
- Inspect the actual workbook structure before writing any formulas — do not assume layouts.

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