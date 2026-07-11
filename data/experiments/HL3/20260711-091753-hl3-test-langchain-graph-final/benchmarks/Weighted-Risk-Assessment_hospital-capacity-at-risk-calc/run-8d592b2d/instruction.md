# Task Instruction

## Task: Update hospital capacity workbook with formulas

### Overview
You must read, understand, and update `/root/data/workbook.xlsx` by populating specific cells with spreadsheet formulas, then save to `/root/output/result.xlsx`.

### Step 0: Inspect the workbook structure
1. `mkdir -p /root/output`
2. Use `openpyxl` to open `/root/data/workbook.xlsx` and inspect:
   - Sheet `Task`: Read row 10 (the year headers in columns H through L). Read column D for rows 12-17, 19-24, 26-31 to understand the series codes. Read any labels in rows 35-40 (cluster names), rows 42-47 (stat labels like min, max, median, mean, 25th, 75th percentile), and row 50 (Regional Care Grid label). Print all of these values.
   - Sheet `Data`: Read rows 21-38 completely to understand the data layout — identify where series codes are, where years are (likely in a header row above or within this range), and how data is organized (rows vs columns). Print the first few columns and the header row.
   - Also check what's currently in cells H12:L17, H19:L24, H26:L31 to confirm they are empty/yellow.
   - Print the exact content of rows 10-11 on Task sheet to see year headers and any other context.

### Step 1: Populate lookup formulas in H12:L17, H19:L24, H26:L31
For each cell in these ranges, write a formula that:
- Takes the series code from column D of that row
- Takes the year from row 10 of that column (H10, I10, J10, K10, or L10)
- Looks up the value from sheet `Data` rows 21:38

Use one of the allowed patterns: `INDEX/MATCH`, `VLOOKUP/MATCH`, `HLOOKUP/MATCH`, or `XLOOKUP/MATCH`.

**Important formula construction notes:**
- You must write these as Excel formula strings (not computed values). Use `openpyxl` and set `cell.value = '=FORMULA...'`.
- Reference the Data sheet properly, e.g., `Data!A21:A38` for series codes column, etc.
- Determine the exact layout of the Data sheet before writing formulas. The lookup approach depends on whether data is organized with series codes in a column and years in a row, or vice versa.
- A common pattern if series codes are in column A and years are in a header row on Data sheet:
  `=INDEX(Data!$B$21:$Z$38, MATCH($D12, Data!$A$21:$A$38, 0), MATCH(H$10, Data!$B$20:$Z$20, 0))`
  Adjust column/row references based on actual Data sheet layout.
- Use absolute references (`$`) appropriately so formulas can be consistent across the range.
- Make sure the series code reference locks the column (`$D12`) and the year reference locks the row (`H$10`).

### Step 2: Net capacity headroom formulas in H35:L40
For each of the six hospital clusters (rows 35-40), calculate:
`=(Available_Care_Slots - Occupied_Care_Slots) / Staffed_Bed_Capacity * 100`

Determine which rows correspond to:
- Available Care Slots: rows 12-17
- Occupied Care Slots: rows 19-24  
- Staffed Bed Capacity: rows 26-31

So for cell H35: `=(H12-H19)/H26*100` (adjust row numbers based on actual mapping — the first cluster in row 35 uses data from row 12, 19, 26; second cluster in row 36 uses rows 13, 20, 27; etc.)

Verify this mapping by checking that the cluster names/order in rows 35-40 match those in rows 12-17.

### Step 2b: Summary statistics in H42:L47
For each column H through L, calculate column-wise statistics over H35:L40:
- Identify which row is which statistic by reading the labels in column D (or wherever) for rows 42-47.
- Use appropriate Excel functions:
  - Minimum: `=MIN(H35:H40)`
  - Maximum: `=MAX(H35:H40)`
  - Median: `=MEDIAN(H35:H40)`
  - Mean: `=AVERAGE(H35:H40)`
  - 25th percentile: `=PERCENTILE(H35:H40, 0.25)` or `=PERCENTILE.INC(H35:H40, 0.25)`
  - 75th percentile: `=PERCENTILE(H35:H40, 0.75)` or `=PERCENTILE.INC(H35:H40, 0.75)`
- Match the function to the correct row based on the labels.

### Step 3: Weighted mean in H50:L50
For each column, use SUMPRODUCT to calculate weighted mean:
`=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)`

This weights the Net capacity headroom percentages by Staffed Bed Capacity.

### Step 4: Save
- Do NOT change any existing formatting, sheet names, or structure.
- Save the workbook to `/root/output/result.xlsx`.

### Validation
After saving, re-open `/root/output/result.xlsx` and:
1. Confirm that cells H12, L17, H19, L24, H26, L31 contain formula strings (start with `=`).
2. Confirm that cells H35, L40 contain formula strings.
3. Confirm that cells H42:L47 contain formula strings.
4. Confirm that cells H50:L50 contain formula strings.
5. Confirm that no new sheets were added.
6. Print a sample of the formulas to verify correctness.

### Critical Reminders
- Read the actual Data sheet layout carefully before constructing any formulas.
- All values in the target cells must be Excel formulas, not hardcoded numbers.
- Preserve all existing formatting — do not clear or overwrite non-target cells.
- The label/stat mapping in rows 42-47 must be read from the sheet, not assumed.

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