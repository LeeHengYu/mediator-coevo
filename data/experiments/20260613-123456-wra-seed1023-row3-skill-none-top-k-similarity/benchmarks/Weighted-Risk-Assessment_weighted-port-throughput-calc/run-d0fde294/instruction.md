# Task Instruction

You must update an Excel workbook at `/root/data/workbook.xlsx` and save the result to `/root/output/result.xlsx`. Work only inside the existing sheets `Task` and `Data`. Do not add sheets, macros, VBA, external links, or helper tabs. Preserve all existing formatting.

## Preliminary Investigation

1. Create `/root/output/` if it doesn't exist.
2. Read the workbook with openpyxl (keep formatting: do NOT use data_only). Inspect:
   - Sheet `Task`: Print the contents of rows 10-50, columns D through L (values and any existing formulas). Pay special attention to:
     - Row 10 (year headers in H10:L10)
     - Column D rows 12-17, 19-24, 26-31 (series codes)
     - Column C or B rows 12-17, 19-24, 26-31 (port names or labels)
     - Rows 35-40 (what labels/structure exists)
     - Rows 42-47 (statistic labels: min, max, median, mean, 25th, 75th percentile)
     - Row 50 (CPA weighted mean row)
   - Sheet `Data`: Print rows 21-38 to understand the data layout. Identify:
     - Which row contains headers (column labels)
     - How series codes appear (which column)
     - How years appear (which row)
     - The overall structure (is data arranged with series codes in a column and years across columns, or transposed?)
3. Print the exact cell references and values so you can construct correct formulas.

## Step 1: Populate H12:L17, H19:L24, H26:L31 with lookup formulas

For each cell in these three blocks, write a spreadsheet formula (as a string starting with `=`) that looks up data from sheet `Data` rows 21:38. The formula must use:
- The series code from column D of the same row on sheet `Task`
- The year from row 10 of the same column on sheet `Task`
- One of these lookup patterns: VLOOKUP+MATCH, HLOOKUP+MATCH, XLOOKUP+MATCH, or INDEX+MATCH

IMPORTANT: Determine the correct lookup pattern by examining the Data sheet layout:
- If series codes are in a column and years are in a row header → INDEX(MATCH for row, MATCH for column) or VLOOKUP with MATCH for column are natural.
- Use absolute references (with $) for the data range and for the row/column anchors that should not shift when the formula is copied across the block, but use relative references for the parts that should change (the series code cell should be anchored to column D, the year cell should be anchored to row 10).

Write the formula for cell H12 first, verify it looks correct by reasoning about what it should return, then apply the analogous formula to all cells in the three blocks. Use `$` signs carefully:
- Lock the lookup column reference (e.g., `$D12` so column D is fixed but row varies)
- Lock the year row reference (e.g., `H$10` so row 10 is fixed but column varies)
- Lock the data range entirely with `$` on both row and column

## Step 2: Net container flow (H35:L40) and statistics (H42:L47)

The three blocks from Step 1 correspond to three metrics for six ports. Based on the row labels, identify which block is:
- Loaded Containers Inbound (likely H12:L17)
- Loaded Containers Outbound (likely H19:L24)  
- Terminal Throughput Capacity (likely H26:L31)

Verify this by checking the labels in column B or C for rows 12, 19, 26.

For H35:L40, write formulas computing:
`(Loaded Containers Inbound - Loaded Containers Outbound) / Terminal Throughput Capacity * 100`

For example, H35 = (H12 - H19) / H26 * 100, H36 = (H13 - H20) / H27 * 100, etc.

For H42:L47, write column-wise aggregate formulas over H35:L40:
- Row 42: MIN(H35:H40) — adjust column for each
- Row 43: MAX(H35:H40)
- Row 44: MEDIAN(H35:H40)
- Row 45: AVERAGE(H35:H40)
- Row 46: PERCENTILE(H35:H40, 0.25) or PERCENTILE.INC(H35:H40, 0.25)
- Row 47: PERCENTILE(H35:H40, 0.75) or PERCENTILE.INC(H35:H40, 0.75)

Check the labels in column B/C/D for rows 42-47 to match the correct statistic to the correct row. The order might differ from what I listed above.

## Step 3: Weighted mean for CPA (H50:L50)

For each column H through L, write a SUMPRODUCT formula:
`=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)`

This computes the weighted mean of the net container flow percentages (Step 2 values) weighted by Terminal Throughput Capacity.

## Final Steps

1. After writing all formulas, save the workbook to `/root/output/result.xlsx`.
2. Reopen the saved file and verify:
   - Formulas exist (not just values) in the target cells
   - Spot-check a few formula strings to confirm correct structure
   - The file has exactly the same sheets as the original (Task and Data, no extras)
3. Print a summary of what was done.

## Critical Notes
- Use `openpyxl` to read and write. Do NOT use `data_only=True` when loading (that strips formulas).
- Write formulas as strings (e.g., `ws['H12'] = '=INDEX(...)'`). Do not compute values in Python.
- Be very careful about the Data sheet layout — inspect it thoroughly before writing any formula.
- If the statistic rows (42-47) have labels that differ from my assumed order, match the formula to the actual label.

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