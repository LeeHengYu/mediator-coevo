# Task Instruction

Execute the following steps precisely to complete the hospital capacity workbook task.

## Step 0: Inspect the workbook
1. Copy `/root/data/workbook.xlsx` to `/root/output/result.xlsx`.
2. Open `/root/output/result.xlsx` with openpyxl (with `data_only=False` so you can read and write formulas).
3. Print out the `Task` sheet structure:
   - Print rows 1–55, all columns A–L, showing cell values (formulas or constants).
   - Pay special attention to:
     - Column D rows 12–31 (series codes for each row)
     - Row 10 columns H–L (years)
     - The yellow cell ranges: H12:L17, H19:L24, H26:L31
     - Rows 35–50 for labels and any existing content
4. Print out the `Data` sheet rows 18–40, all populated columns, to understand the data layout:
   - Identify whether Data is organized with series codes in a column and years in a row, or vice versa.
   - Identify exactly which column contains the series codes and which row contains the years.
   - Note the exact row/column ranges.

## Step 1: Populate lookup formulas in H12:L17, H19:L24, H26:L31

Based on the Data sheet layout discovered in Step 0, write INDEX/MATCH formulas (or VLOOKUP with MATCH, etc.) into each yellow cell. The formula pattern for each cell should:
- Use the series code from column D of that row (e.g., `$D12` for row 12, with $ on column to allow horizontal copying)
- Use the year from row 10 of that column (e.g., `H$10` for column H, with $ on row)
- Look up in the Data sheet rows 21:38

For example, if Data has series codes in column A (rows 21:38) and year headers in some row, and data values spread across columns, use a pattern like:
`=INDEX(Data!$B$21:$Z$38, MATCH($D12, Data!$A$21:$A$38, 0), MATCH(H$10, Data!$B$20:$Z$20, 0))`

Adjust the exact column/row references based on what you discovered in Step 0. The key requirements:
- The lookup must use TWO inputs: series code from column D and year from row 10.
- The lookup range must be within Data rows 21:38.
- Use one of: VLOOKUP+MATCH, HLOOKUP+MATCH, XLOOKUP+MATCH, or INDEX+MATCH.
- Make references absolute where needed so the formula pattern is consistent across the block.

Write formulas into all 30 cells across the three blocks (6 rows × 5 columns each).

## Step 2: Net capacity headroom and statistics

### Step 2a: H35:L40 — Net capacity headroom
For each of the 6 hospital clusters (rows 35–40) and each year (columns H–L), write:
`=(H12 - H19) / H26 * 100`

where:
- H12:L17 = Available Care Slots (rows 12–17)
- H19:L24 = Occupied Care Slots (rows 19–24)
- H26:L31 = Staffed Bed Capacity (rows 26–31)

So row 35 uses rows 12, 19, 26; row 36 uses rows 13, 20, 27; etc.

For cell H35: `=(H12-H19)/H26*100`
For cell H36: `=(H13-H20)/H27*100`
... and so on through row 40 and columns H through L.

### Step 2b: H42:L47 — Summary statistics
Based on the labels in column D (or nearby) for rows 42–47, write column-wise formulas. Check the exact labels, but they should be minimum, maximum, median, mean, 25th percentile, 75th percentile. Use the range H35:H40 (for column H), etc.:
- Minimum: `=MIN(H35:H40)`
- Maximum: `=MAX(H35:H40)`
- Median: `=MEDIAN(H35:H40)`
- Mean: `=AVERAGE(H35:H40)`
- 25th percentile: `=PERCENTILE(H35:H40,0.25)`
- 75th percentile: `=PERCENTILE(H35:H40,0.75)`

Match each formula to the correct row based on the label. Write for all 5 year columns.

## Step 3: Weighted mean in H50:L50
For each year column, compute:
`=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)`

This weights the Net capacity headroom percentages by Staffed Bed Capacity.

Write this formula in H50:L50.

## Step 4: Save
Save the workbook to `/root/output/result.xlsx`. Do NOT change any formatting, do not add sheets, macros, VBA, external links, or helper tabs.

## Validation
After saving, reopen the file and print:
- A sample of the lookup formulas (e.g., H12, L17, H19, L24, H26, L31)
- The headroom formulas (H35, L40)
- The statistics formulas (H42:H47)
- The weighted mean formula (H50)
Confirm they are all present as formulas (not empty, not just values).

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