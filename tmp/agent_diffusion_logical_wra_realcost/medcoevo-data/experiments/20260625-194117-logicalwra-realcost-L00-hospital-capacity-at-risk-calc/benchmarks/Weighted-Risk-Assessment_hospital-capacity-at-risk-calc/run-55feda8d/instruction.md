# Task Instruction

## Task: Update hospital capacity workbook with formulas

### Overview
You need to read, understand, and update `/root/data/workbook.xlsx` by populating specific cells with spreadsheet formulas, then save the result to `/root/output/result.xlsx`.

### Step 0: Inspect the workbook structure
1. `mkdir -p /root/output`
2. Use `openpyxl` to open `/root/data/workbook.xlsx` and inspect:
   - Sheet `Task`: Print rows 1-55 to understand the layout. Pay special attention to:
     - Column D (series codes) for rows 12-17, 19-24, 26-31
     - Row 10 (years in columns H-L)
     - The labels/structure around H35:L40, H42:L47, H50:L50
     - Which cells are yellow (check fill colors in H12:L17, H19:L24, H26:L31)
   - Sheet `Data`: Print rows 21-38 to understand the data layout. Note:
     - How the data is organized (rows vs columns)
     - Where series codes appear
     - Where years appear
     - The exact range structure
3. Print the exact series codes in column D for rows 12-17, 19-24, 26-31.
4. Print the exact years in row 10, columns H-L.
5. Print the data layout in `Data` sheet rows 21-38 including headers/labels.

### Step 1: Populate lookup formulas in H12:L17, H19:L24, H26:L31

For each yellow cell in these ranges, write a spreadsheet **formula** (not a computed value) that:
- Uses the series code from column D of that row
- Uses the year from row 10 of that column
- Looks up the value from `Data` sheet rows 21:38
- Uses one of the allowed patterns: INDEX/MATCH, VLOOKUP/MATCH, HLOOKUP/MATCH, or XLOOKUP/MATCH

**Critical formula details:**
- You must write these as Excel formula strings (starting with `=`), NOT Python-computed values.
- When referencing the Data sheet, use `Data!` prefix.
- Determine from inspection whether the Data sheet is organized with series codes in rows or columns, and years in rows or columns, then choose the appropriate lookup pattern.
- A common pattern if data has series codes in one column and years across columns:
  `=INDEX(Data!$B$21:$Z$38, MATCH($D12, Data!$A$21:$A$38, 0), MATCH(H$10, Data!$B$20:$Z$20, 0))`
  But you MUST adjust the exact ranges based on your inspection of the actual data layout.
- Use `$` for absolute references on the data range and lookup arrays. Use mixed references so the formula can be consistent across the block (e.g., `$D12` locks column D, `H$10` locks row 10).

### Step 2: Net capacity headroom formulas in H35:L40

For each cell in H35:L40 (6 hospital clusters × 5 years):
- Formula: `=(Available_Care_Slots - Occupied_Care_Slots) / Staffed_Bed_Capacity * 100`
- The three blocks from Step 1 correspond to three data series. From inspection, determine which block (rows 12-17, 19-24, 26-31) corresponds to which metric.
- For example, if rows 12-17 = Available Care Slots, rows 19-24 = Occupied Care Slots, rows 26-31 = Staffed Bed Capacity, then:
  `=( H12 - H19 ) / H26 * 100`
  Adjust based on actual labels found during inspection.

### Step 2 continued: Summary statistics in H42:L47

For each column (H through L), calculate these six statistics over the 6 values in H35:L40 (or the corresponding column):
- Row 42: `=MIN(H35:H40)` (minimum)
- Row 43: `=MAX(H35:H40)` (maximum)  
- Row 44: `=MEDIAN(H35:H40)` (median)
- Row 45: `=AVERAGE(H35:H40)` (simple mean)
- Row 46: `=PERCENTILE(H35:H40, 0.25)` (25th percentile)
- Row 47: `=PERCENTILE(H35:H40, 0.75)` (75th percentile)

**IMPORTANT:** Check the actual labels in column D/E/F/G for rows 42-47 to determine which row gets which statistic. Map them correctly based on what you see.

### Step 3: Weighted mean in H50:L50

For each column H-L:
`=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)`

This calculates the weighted mean of Net capacity headroom (Step 2 percentages) weighted by Staffed Bed Capacity.

### Step 3 (saving):
1. Save the workbook to `/root/output/result.xlsx`.
2. Do NOT change any existing formatting, do NOT add sheets, macros, VBA, external links, or helper tabs.

### Validation
After saving, reopen `/root/output/result.xlsx` and:
1. Verify that cells in H12:L17, H19:L24, H26:L31 contain formula strings (start with `=`).
2. Verify that cells in H35:L40 contain formulas.
3. Verify that cells in H42:L47 contain formulas.
4. Verify that cells in H50:L50 contain formulas.
5. Print a few sample formulas to confirm they reference the correct ranges.
6. Confirm no new sheets were added.
7. Confirm the file exists at the output path.

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