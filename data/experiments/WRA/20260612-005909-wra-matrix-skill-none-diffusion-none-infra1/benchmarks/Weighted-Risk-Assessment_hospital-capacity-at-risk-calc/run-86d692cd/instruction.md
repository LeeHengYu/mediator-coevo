# Task Instruction

## Task: Update hospital capacity workbook with formulas

### Overview
You need to read, understand, and update `/root/data/workbook.xlsx` by populating specific cells with spreadsheet formulas, then save to `/root/output/result.xlsx`.

### Step 0: Inspect the workbook
1. `mkdir -p /root/output`
2. Use `openpyxl` to open `/root/data/workbook.xlsx` and inspect both sheets:
   - On sheet `Task`: print the layout of rows 1-55, focusing on columns A-L. Pay special attention to:
     - Column D (series codes) for rows 12-17, 19-24, 26-31
     - Row 10 (years in columns H-L)
     - The labels/structure of rows 35-50
     - Any existing formulas or values already present
     - Cell fill colors to confirm yellow cells
   - On sheet `Data`: print rows 21-38 fully, noting the structure (which row has headers, which column has series codes, how years map to columns). Print the first few rows and column headers to understand the data layout.
3. Print cell values for key reference points: D12:D17, D19:D24, D26:D31, H10:L10, and the header row of the Data sheet's rows 21-38 area.

### Step 1: Populate lookup formulas in H12:L17, H19:L24, H26:L31
For each cell in these ranges, write a spreadsheet formula (not a Python-computed value) that:
- Takes TWO inputs: the series code from column D of that row, and the year from row 10 of that column
- Looks up the value from sheet `Data` rows 21:38
- Uses one of these patterns: `INDEX/MATCH`, `VLOOKUP/MATCH`, `HLOOKUP/MATCH`, or `XLOOKUP/MATCH`

IMPORTANT: You must write actual Excel formulas as strings (e.g., `=INDEX(Data!B21:B38,MATCH(...))`). Do NOT compute values in Python and write them as constants.

To build correct formulas:
- First understand the Data sheet layout: which column contains series codes, which row contains years, and where the data values are.
- Determine the correct ranges for the lookup. The formula must reference `Data!` sheet.
- Use absolute references where appropriate to allow the formula pattern to work across the H-L columns and the relevant rows.
- Test one formula mentally before applying the pattern.

### Step 2a: Net capacity headroom in H35:L40
Write spreadsheet formulas in H35:L40 that compute:
`(Available Care Slots - Occupied Care Slots) / Staffed Bed Capacity * 100`

where:
- Available Care Slots = H12:L17 block (rows 12-17 correspond to the 6 hospital clusters)
- Occupied Care Slots = H19:L24 block (rows 19-24)
- Staffed Bed Capacity = H26:L31 block (rows 26-31)

So for cell H35: `=(H12-H19)/H26*100` and similarly for the rest of the 6×5 grid.

Verify the row mapping: row 35 maps to cluster 1 (rows 12, 19, 26), row 36 maps to cluster 2 (rows 13, 20, 27), etc.

### Step 2b: Summary statistics in H42:L47
Write spreadsheet formulas for column-wise statistics over H35:L40:
- H42:L42 = MIN of H35:H40 (for each column)
- H43:L43 = MAX
- H44:L44 = MEDIAN
- H45:L45 = AVERAGE
- H46:L46 = PERCENTILE (25th) or PERCENTILE.INC(..., 0.25)
- H47:L47 = PERCENTILE (75th) or PERCENTILE.INC(..., 0.75)

CHECK: Look at the labels in column A/B/C/D for rows 42-47 to confirm which row is which statistic. Map accordingly - do NOT assume the order above; use the actual labels.

### Step 3: Weighted mean in H50:L50
Write a SUMPRODUCT-based formula for weighted mean:
`=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)`
for each column H through L.

### Step 4: Save
- Save to `/root/output/result.xlsx` using openpyxl, preserving all formatting.
- Use `keep_vba=False`, and do NOT modify any sheet names, cell formats, or other content.

### Step 5: Validate
- Reopen `/root/output/result.xlsx` and verify:
  1. Cells H12:L31 contain formula strings (check `.value` starts with `=`)
  2. Cells H35:L40 contain formula strings
  3. Cells H42:L47 contain formula strings
  4. Cells H50:L50 contain formula strings
  5. No new sheets were added
  6. Print a sample of formulas to confirm correctness

### Critical Notes
- Use `openpyxl` to read and write. When writing formulas, assign them as strings starting with `=`.
- Do NOT use `data_only=True` when reading (you need to preserve existing formulas).
- The row-to-statistic mapping in rows 42-47 MUST match the actual labels in the workbook. Inspect before writing.
- Keep all existing formatting. Do not clear or overwrite cells outside the specified ranges.
- Double-check the Data sheet structure before writing lookup formulas. The exact column/row ranges matter.

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