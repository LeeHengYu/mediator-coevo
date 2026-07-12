# Task Instruction

You must update the Excel workbook `/root/data/workbook.xlsx` and save the result to `/root/output/result.xlsx`. Follow these steps precisely.

## Preparation
1. `mkdir -p /root/output`
2. Install openpyxl if needed: `pip install openpyxl`
3. Open `/root/data/workbook.xlsx` and inspect:
   - Sheet `Task`: read the layout of rows 10-50, columns D and H-L. Identify the series codes in column D for rows 12-17, 19-24, 26-31, 35-40, 42-47, 50. Identify the years in row 10 columns H-L. Print all of these values.
   - Sheet `Data`: read rows 21-38 to understand the data layout (which row has headers, which column has series codes, how years map to columns). Print the first few rows so you understand the structure.
   - Also inspect what labels are in cells around rows 35-47 and row 50 on sheet `Task` to understand what each row represents.

## Step 1: Populate lookup formulas in H12:L17, H19:L24, H26:L31

For each cell in these three blocks, write a spreadsheet formula (not a Python-computed value) that looks up data from sheet `Data` rows 21:38.

The formula must use the series code from column D of that row and the year from row 10 of that column. Use one of these patterns: VLOOKUP+MATCH, HLOOKUP+MATCH, XLOOKUP+MATCH, or INDEX+MATCH.

IMPORTANT: Before writing formulas, determine the exact layout of the Data sheet rows 21:38:
- Which column contains the series codes (lookup keys)?
- Are years in a header row, and if so which row?
- Is the data arranged with series codes in rows and years in columns, or transposed?

Based on the layout, choose the most natural lookup pattern. For example, if Data has series codes in column A and years across a header row, then INDEX(MATCH(series_code, ...), MATCH(year, ...)) is natural.

Make sure to use absolute references for the Data range and relative/mixed references appropriately so the formula works across the entire block. Use the sheet reference `Data!` in the formula.

Write the formula into each cell using openpyxl (set `cell.value = '=FORMULA...'`). Do NOT set `cell.data_type` manually; just assign the string starting with `=`.

## Step 2: Net reliability gap (H35:L40) and statistics (H42:L47)

For H35:L40 (6 regions × 5 years):
- The formula is: `(Successful API Requests - Failed API Requests) / Compute Capacity * 100`
- Identify which of the three blocks (H12:L17, H19:L24, H26:L31) corresponds to "Successful API Requests", "Failed API Requests", and "Compute Capacity" by reading the labels on the Task sheet.
- Write a cell formula referencing the appropriate cells from those blocks. For example, if row 12 aligns with the same region as row 35, then H35 = (H12 - H19) / H26 * 100 (adjust based on actual layout).

For H42:L47 (statistics), write column-wise formulas over the H35:L40 range:
- Row for minimum: `=MIN(H35:H40)` (and similarly for columns I-L)
- Row for maximum: `=MAX(H35:H40)`
- Row for median: `=MEDIAN(H35:H40)`
- Row for simple mean: `=AVERAGE(H35:H40)`
- Row for 25th percentile: `=PERCENTILE(H35:H40,0.25)`
- Row for 75th percentile: `=PERCENTILE(H35:H40,0.75)`

Match each statistic to the correct row by reading the labels in column D (or nearby) for rows 42-47.

## Step 3: Weighted mean in H50:L50

For each column (H through L), write a SUMPRODUCT formula:
`=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)`

This computes the weighted mean of the Net reliability gap percentages weighted by Compute Capacity.

## Final checks
1. Do NOT modify any formatting, do not add sheets, macros, VBA, external links, or helper tabs.
2. Save to `/root/output/result.xlsx`.
3. Re-open the saved file and verify:
   - Sheets are exactly `Task` and `Data` (no extra sheets).
   - Cells H12, L31, H35, L40, H42, L47, H50, L50 all contain formula strings (start with `=`).
   - Print a sample of formulas to confirm correctness.

Critical: Read the actual workbook structure carefully before writing any formulas. The exact row/column references in the Data sheet and the mapping between blocks and metric names must come from inspecting the file, not from assumptions.

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