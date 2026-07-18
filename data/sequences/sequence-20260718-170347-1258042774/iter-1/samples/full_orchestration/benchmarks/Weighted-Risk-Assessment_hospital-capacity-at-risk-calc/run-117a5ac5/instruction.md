# Task Instruction

## Task: Populate formulas and calculations in hospital capacity workbook

### Overview
You must update `/root/data/workbook.xlsx` by populating specific cells with spreadsheet formulas, then save the result to `/root/output/result.xlsx`. Work only inside the existing `Task` and `Data` sheets. Do not add sheets, macros, VBA, external links, or helper tabs. Preserve all existing formatting.

### Step 0: Inspect the workbook structure
1. Create `/root/output/` directory if it doesn't exist.
2. Open `/root/data/workbook.xlsx` with openpyxl and inspect:
   - Sheet `Task`: Read row 10 to find the year headers in columns H through L. Read column D rows 12-17, 19-24, 26-31 to find series codes. Read the labels in rows 35-40, 42-47, and 50. Note the exact text/values in all these cells.
   - Sheet `Data`: Read rows 21-38 to understand the data layout — identify which row has headers, which column has series codes, and how years map to columns.
3. Print all of this information so you understand the exact structure before writing any formulas.

### Step 1: Populate lookup formulas in H12:L17, H19:L24, H26:L31
For each cell in these three blocks, write a formula that looks up data from `Data!$21:$38` using:
- The series code from column D of the current row on sheet `Task`
- The year from row 10 of the current column on sheet `Task`

Use one of these patterns: `INDEX/MATCH`, `VLOOKUP/MATCH`, `HLOOKUP/MATCH`, or `XLOOKUP/MATCH`. The formula must use two MATCH functions (or equivalent) — one for the row dimension (series code) and one for the column dimension (year).

IMPORTANT: When constructing formulas, ensure:
- References to the Data sheet use the correct sheet name syntax: `Data!` prefix
- Row/column references match the actual data layout you discovered in Step 0
- Use appropriate absolute references ($) where needed for anchoring lookup ranges
- The formulas should work when dragged/copied across the H:L columns and down the rows

### Step 2: Calculate Net capacity headroom in H35:L40
For each of the 6 hospital clusters (rows 35-40) and each year column (H-L), compute:
```
(Available Care Slots - Occupied Care Slots) / Staffed Bed Capacity * 100
```
The three input blocks are:
- H12:L17 = one metric (check which one by reading the label)
- H19:L24 = another metric
- H26:L31 = another metric

Map the correct blocks to `Available Care Slots`, `Occupied Care Slots`, and `Staffed Bed Capacity` based on the labels/series codes you found in Step 0. Write cell formulas (not hardcoded values).

Then in H42:L47, compute column-wise summary statistics using formulas:
- Minimum: `=MIN(H35:H40)` pattern
- Maximum: `=MAX(H35:H40)` pattern  
- Median: `=MEDIAN(H35:H40)` pattern
- Simple mean: `=AVERAGE(H35:H40)` pattern
- 25th percentile: `=PERCENTILE(H35:H40,0.25)` or `=PERCENTILE.INC(H35:H40,0.25)` pattern
- 75th percentile: `=PERCENTILE(H35:H40,0.75)` or `=PERCENTILE.INC(H35:H40,0.75)` pattern

Check the labels in column D (or nearby) for rows 42-47 to determine which row gets which statistic.

### Step 3: Weighted mean in H50:L50
For each year column, compute the weighted mean of the Net capacity headroom percentages (H35:L40) weighted by the Staffed Bed Capacity values (H26:L31):
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```
Use `SUMPRODUCT` as required by the task.

### Step 4: Save
Save the workbook to `/root/output/result.xlsx`. Do NOT change any existing formatting, sheet names, or structure.

### Step 5: Validate
1. Reopen `/root/output/result.xlsx` and verify:
   - Cells H12:L17, H19:L24, H26:L31 contain formula strings (not None/empty)
   - Cells H35:L40 contain formulas
   - Cells H42:L47 contain formulas
   - Cells H50:L50 contain formulas
2. Print a sample of the formulas to confirm they reference the correct ranges.
3. Check that no new sheets were added.

### Critical Notes
- Use `openpyxl` to read and write the workbook.
- Write Excel formula strings (starting with `=`) into cells, not computed Python values.
- Be very careful about the Data sheet layout — inspect it thoroughly before writing formulas.
- If the Data sheet has series codes in a column and years in a row, your INDEX/MATCH should match accordingly.
- Ensure row 10 year references and column D series code references use the correct cell addresses.

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