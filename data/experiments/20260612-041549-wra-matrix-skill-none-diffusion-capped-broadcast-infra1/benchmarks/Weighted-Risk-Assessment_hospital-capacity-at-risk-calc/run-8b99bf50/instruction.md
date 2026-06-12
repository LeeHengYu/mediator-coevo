# Task Instruction

## Task: Update hospital capacity workbook with formulas

### Overview
You must read, understand, and update `/root/data/workbook.xlsx` by populating specific cells with spreadsheet formulas, then save the result to `/root/output/result.xlsx`.

### Step 0: Inspect the workbook
1. `mkdir -p /root/output`
2. Use `openpyxl` to open `/root/data/workbook.xlsx` and inspect:
   - Sheet `Task`: Read the structure carefully. Print rows 1–55 or so, paying special attention to:
     - Column D (series codes for each row)
     - Row 10 (years in columns H through L)
     - The yellow cell ranges: H12:L17, H19:L24, H26:L31
     - Row 35-40 labels and any existing content
     - Rows 42-47 labels (min, max, median, mean, 25th, 75th percentile)
     - Row 50 label
   - Sheet `Data`: Read rows 21–38 to understand the data layout. Print the header row and several data rows. Identify:
     - Which row/column contains series codes
     - Which row/column contains years
     - The orientation of the data (is it vertical with series codes in a column and years across columns, or horizontal?)
   - Print cell values, not just a summary. You need exact cell references to write correct formulas.

3. Determine the exact lookup structure:
   - Where are series codes in the `Data` sheet (which column)?
   - Where are years in the `Data` sheet (which row)?
   - What is the data range?
   This determines whether to use VLOOKUP+MATCH, HLOOKUP+MATCH, INDEX+MATCH, or XLOOKUP+MATCH.

### Step 1: Populate lookup formulas in H12:L17, H19:L24, H26:L31
For each cell in these ranges, write a formula that:
- Takes the series code from column D of that row
- Takes the year from row 10 of that column
- Looks up the corresponding value from `Data!` rows 21:38
- Uses one of the allowed patterns: INDEX+MATCH, VLOOKUP+MATCH, HLOOKUP+MATCH, or XLOOKUP+MATCH

IMPORTANT: Use `openpyxl` to write Excel formula strings (e.g., `=INDEX(...)`) into the cells. Do NOT compute values in Python. The cells must contain live spreadsheet formulas.

When writing formulas, be precise about:
- Absolute vs relative references where needed
- The exact range references on the Data sheet
- The match type (exact match = 0)

### Step 2a: Net capacity headroom in H35:L40
For each of the 6 hospital clusters (rows 35-40) and each year (columns H-L), write a formula:
`= (Available Care Slots - Occupied Care Slots) / Staffed Bed Capacity * 100`

The three input blocks are:
- H12:L17 = one metric (check which one by reading the labels)
- H19:L24 = another metric
- H26:L31 = another metric

You must identify which block corresponds to which metric (Available Care Slots, Occupied Care Slots, Staffed Bed Capacity) by reading the section headers/labels on the Task sheet. Then write the formula accordingly.

For example, if row 12 corresponds to the same cluster as row 35, and H12:L17 is Available Care Slots, H19:L24 is Occupied Care Slots, H26:L31 is Staffed Bed Capacity, then:
`H35 = (H12 - H19) / H26 * 100`

Verify the row correspondence between the lookup blocks and the headroom block.

### Step 2b: Summary statistics in H42:L47
For each column (H through L), calculate column-wise statistics over the 6 headroom values (rows 35-40):
- Row 42: MIN
- Row 43: MAX
- Row 44: MEDIAN
- Row 45: AVERAGE (simple mean)
- Row 46: 25th percentile (use PERCENTILE or PERCENTILE.INC with 0.25)
- Row 47: 75th percentile (use PERCENTILE or PERCENTILE.INC with 0.75)

IMPORTANT: Check the actual labels in column A/B/C/D for rows 42-47 to confirm which statistic goes in which row. Adjust accordingly.

### Step 3: Weighted mean in H50:L50
For each column H through L, write a SUMPRODUCT formula:
`=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)`

This computes the weighted mean of the headroom percentages (H35:H40) weighted by Staffed Bed Capacity (H26:H31).

### Step 4: Save
- Do NOT change any existing formatting, sheet names, or structure.
- Save to `/root/output/result.xlsx`

### Validation
After saving, reopen the file and verify:
1. Cells H12, L17, H19, L24, H26, L31 contain formula strings (start with `=`)
2. Cells H35, L40 contain formula strings
3. Cells H42, L47 contain formula strings
4. Cell H50 and L50 contain formula strings
5. No new sheets were added
6. Print a sample of the formulas to confirm correctness

### Critical Notes
- All cells must contain Excel formulas, not computed Python values
- Preserve all existing formatting (use `openpyxl` with `load_workbook` keeping styles)
- Do not add sheets, macros, VBA, external links, or helper tabs
- Read the actual workbook structure before writing any formulas — do not assume cell positions

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