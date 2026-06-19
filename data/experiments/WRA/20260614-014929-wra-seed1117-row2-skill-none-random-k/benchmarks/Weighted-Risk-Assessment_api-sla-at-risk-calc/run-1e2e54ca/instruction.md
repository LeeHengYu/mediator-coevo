# Task Instruction

## Task: Update `/root/data/workbook.xlsx` with formulas and save to `/root/output/result.xlsx`

### Pre-work: Inspect the workbook
1. Create `/root/output/` directory if it doesn't exist.
2. Use `openpyxl` to open `/root/data/workbook.xlsx` and inspect:
   - Sheet `Task`: Read cells D12:D17, D19:D24, D26:D31 to get the series codes for each row. Read row 10 (H10:L10) to get the year headers. Read H35:H40 row labels or D35:D40 if present. Read any existing content in H42:L47 row labels. Read H50 row label and any weights reference.
   - Sheet `Data`: Read rows 21:38 to understand the data layout — specifically which row contains headers, which column has series codes, and how data is arranged. Note the exact column letters and row numbers.
   - Check cell fill colors to confirm yellow cells match the ranges specified.
   - Print all findings before proceeding.

### Step 1: Populate lookup formulas in H12:L17, H19:L24, H26:L31

For each cell in these ranges, write a formula that looks up data from sheet `Data` rows 21:38. The formula must use one of these patterns: VLOOKUP+MATCH, HLOOKUP+MATCH, XLOOKUP+MATCH, or INDEX+MATCH.

- The two inputs are: (a) the series code from column D of the same row on sheet `Task`, and (b) the year from row 10 on sheet `Task`.
- Based on the Data sheet layout you inspected, choose the appropriate lookup pattern. For example, if Data has series codes in a column and years across a row header, INDEX(MATCH,MATCH) is likely cleanest.
- Use absolute references for the Data range and MATCH ranges, but relative references for the series code (column D, same row) and year (row 10, same column).
- Write the formula as a string (e.g., `ws['H12'] = '=INDEX(...)'`). Make sure the formula references are correct for the Data sheet (e.g., `Data!A21:Z38` or whatever the actual range is).
- Verify that when you change columns H→I→J→K→L, the year reference shifts, and when you change rows, the series code reference shifts.

### Step 2: Net SLA buffer in H35:L40 and statistics in H42:L47

For H35:L40:
- Formula: `(Latency Budget Preserved - Latency Budget Consumed) / Covered Request Capacity * 100`
- Identify which of the three blocks (H12:L17, H19:L24, H26:L31) corresponds to "Latency Budget Preserved", "Latency Budget Consumed", and "Covered Request Capacity" by reading the block headers/labels on the Task sheet.
- For each cell, reference the corresponding cells from those blocks. For example, if row 35 corresponds to the first service: `=(H12-H19)/H26*100` (adjust based on actual block mapping).

For H42:L47 (column-wise statistics over H35:L40):
- Row 42: `=MIN(H35:H40)` (or whichever row is MIN — check the labels)
- Row 43: `=MAX(H35:H40)`
- Row 44: `=MEDIAN(H35:H40)`
- Row 45: `=AVERAGE(H35:H40)`
- Row 46: `=PERCENTILE(H35:H40,0.25)`
- Row 47: `=PERCENTILE(H35:H40,0.75)`
- Match the statistic to the correct row based on the labels you read from the sheet. The labels might be in column D or G or similar.

### Step 3: Weighted mean in H50:L50
- Formula: `=SUMPRODUCT(H35:H40,H26:H31)/SUM(H26:H31)` for each column H through L.
- This computes the weighted mean of Net SLA buffer percentages weighted by Covered Request Capacity.

### Saving
- Save the workbook to `/root/output/result.xlsx` using openpyxl.
- Do NOT change formatting, do NOT add sheets, macros, VBA, external links, or helper tabs.
- After saving, reopen the file and verify:
  - Formulas exist in all target cells (spot-check H12, L17, H19, L24, H26, L31, H35, H40, H42, H47, H50, L50).
  - No sheets were added or removed.
  - The formulas reference the correct sheets and ranges.

### Critical Notes
- Use `openpyxl` with `data_only=False` (the default) so formulas are preserved as formulas, not evaluated values.
- When writing formulas, they must be Excel formula strings starting with `=`.
- Do NOT use `load_workbook(..., data_only=True)` — that would strip formulas.
- Carefully inspect the Data sheet layout before writing any formulas. The exact row/column ranges in formulas must match the actual data positions.

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