# Task Instruction

## Task: Update `/root/data/workbook.xlsx` with formulas and save to `/root/output/result.xlsx`

### Phase 0: Inspect the workbook
1. Create `/root/output/` directory if it doesn't exist.
2. Use `openpyxl` to open `/root/data/workbook.xlsx` and inspect:
   - Sheet `Task`: Read cells in column D rows 12-17, 19-24, 26-31 to understand the series codes. Read row 10 columns H-L to understand the year headers. Read any labels in rows 35-40, 42-47, 50. Note the exact text of series codes and year values.
   - Sheet `Data`: Read rows 21-38 to understand the data layout — specifically which row/column contains what, what the header row is, how series codes appear, and how years are arranged (are years in columns or rows?).
3. Print all of this information so you can construct correct formulas.

### Phase 1: Populate lookup formulas in H12:L17, H19:L24, H26:L31

For each yellow cell in these three blocks, write a spreadsheet formula (not a Python-computed value) that looks up data from the `Data` sheet rows 21-38.

**Important formula construction rules:**
- Each formula must use TWO inputs: (a) the series code from column D of the current row on `Task`, and (b) the year from row 10 on `Task`.
- Use one of these patterns: `INDEX(MATCH,MATCH)`, `VLOOKUP+MATCH`, `HLOOKUP+MATCH`, or `XLOOKUP+MATCH`.
- Reference the Data sheet appropriately (e.g., `Data!A21:Z38` or whatever the actual range is).
- Use absolute references (`$`) where appropriate so that the series code reference locks to column D and the year reference locks to row 10, while allowing the formula to vary correctly across the grid.
- When writing formulas with openpyxl, assign them as strings starting with `=` to the cell's `.value` property. Do NOT use `data_only` mode for writing.

**Before writing formulas**, verify:
- The exact column in Data sheet that contains the series codes (to use as the lookup column).
- The exact row in Data sheet that contains years (to use as the match row).
- Whether years are stored as numbers or strings — match the type used in row 10 of Task sheet.
- The exact range boundaries for Data rows 21:38.

### Phase 2: Net reliability gap formulas in H35:L40

For each of the six regions (rows 35-40) and each year column (H-L):
- Write a formula: `=(SuccessfulAPIRequests - FailedAPIRequests) / ComputeCapacity * 100`
- The Successful API Requests values are in H12:L17, Failed API Requests in H19:L24, and Compute Capacity in H26:L31.
- Match each region row correctly (row 35 corresponds to row 12, 19, 26; row 36 to rows 13, 20, 27; etc.).

### Phase 3: Summary statistics in H42:L47

For each year column (H-L), write column-wise formulas over the Net reliability gap block (rows 35-40):
- Row 42: `=MIN(H35:H40)` (adjust column letter per column)
- Row 43: `=MAX(H35:H40)`
- Row 44: `=MEDIAN(H35:H40)`
- Row 45: `=AVERAGE(H35:H40)`
- Row 46: `=PERCENTILE(H35:H40,0.25)` (25th percentile)
- Row 47: `=PERCENTILE(H35:H40,0.75)` (75th percentile)

**Check the row labels** in column A-G for rows 42-47 to confirm which row is MIN, MAX, MEDIAN, MEAN, 25th percentile, 75th percentile, and assign formulas accordingly.

### Phase 4: Weighted mean in H50:L50

For each year column (H-L), write:
`=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)`
(using the Net reliability gap percentages as values and Compute Capacity as weights)

This is the weighted mean for Global Cloud Mesh (GCM).

### Phase 5: Save and validate
1. Save the workbook to `/root/output/result.xlsx`.
2. Re-open the saved file and verify:
   - Cells H12, L17, H19, L24, H26, L31 contain formula strings (start with `=`).
   - Cells H35, L40 contain formula strings.
   - Cells H42, L47 contain formula strings.
   - Cells H50, L50 contain formula strings.
   - No new sheets were added.
   - Print a sample of formulas from each block to confirm correctness.
3. Do NOT use `data_only=True` when saving or verifying formula presence.

### Critical constraints
- Do NOT add sheets, macros, VBA, external links, or helper tabs.
- Do NOT change existing formatting (don't modify fonts, fills, borders, number formats, column widths, etc.).
- Write Excel formulas as cell values, not computed Python results.
- Use `openpyxl` for all operations. If openpyxl is not available, install it with pip.

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
Task metadata: author_email=catpaw@meituan.com, author_name=CatPaw Task Engineer, category=spreadsheet-formula-reuse, difficulty=easy, tags=[excel, formulas, lookup, statistics, weighted-mean].
Verifier config: timeout_sec=600.0.