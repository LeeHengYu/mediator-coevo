# Task Instruction

Execute the following steps to complete the hospital-capacity-at-risk workbook.

## Step 0 – Inspect the workbook structure

1. Copy the source workbook:
   ```bash
   mkdir -p /root/output
   cp /root/data/workbook.xlsx /root/output/result.xlsx
   ```
2. Open `/root/output/result.xlsx` with openpyxl (data_only=False) and inspect:
   - **Task sheet**: Read rows 10-50, columns D-L to understand the layout:
     - Row 10 should contain year headers in columns H-L.
     - Column D rows 12-17, 19-24, 26-31 should contain series codes.
     - Rows 35-40 are for Net capacity headroom.
     - Rows 42-47 are for summary statistics (min, max, median, mean, 25th, 75th percentile).
     - Row 50 is for weighted mean.
   - **Data sheet**: Read rows 21-38 to understand the data layout. Identify:
     - Which column contains the series codes (the lookup key).
     - Which row contains the year headers.
     - The data range for INDEX/MATCH.
   - Print all of this so you can confirm the exact cell references before writing formulas.

## Step 1 – Populate lookup formulas in H12:L31

Using openpyxl, write INDEX/MATCH formulas into each cell in the three blocks: H12:L17, H19:L24, H26:L31.

The formula pattern for each cell should be:
```
=INDEX(Data!<data_columns_range>, MATCH($D12, Data!<series_code_column>, 0), MATCH(H$10, Data!<year_header_row>, 0))
```

Adjust the references based on what you found in Step 0:
- `$D12` locks the column to D but lets the row float (use `$D` prefix for column lock).
- `H$10` locks the row to 10 but lets the column float (use `$10` suffix for row lock).
- The Data sheet ranges should be absolute (e.g., `Data!$B$21:$B$38` for the series code column, `Data!$B$20:$Z$20` for the year header row, `Data!$C$21:$Z$38` for the data area — adjust column letters based on actual inspection).

Write the formula to cell H12 first, then replicate it across H12:L17 adjusting only the cell reference (the `$` locks handle the rest). Do the same for H19:L24 and H26:L31.

Use a loop: for each row in range, for each column H through L, write the formula string.

## Step 2 – Net capacity headroom (H35:L40)

For each cell in H35:L40, write a formula:
```
=(H12 - H19) / H26 * 100
```
where row 12 corresponds to Available Care Slots, row 19 to Occupied Care Slots, row 26 to Staffed Bed Capacity. The exact row offsets should match: row 35 uses rows 12, 19, 26; row 36 uses rows 13, 20, 27; etc.

So for cell at (row_offset, col): `=(<col><12+offset> - <col><19+offset>) / <col><26+offset> * 100`

## Step 2b – Summary statistics (H42:L47)

For each column H through L, write these formulas:
- Row 42 (MIN):    `=MIN(H35:H40)`
- Row 43 (MAX):    `=MAX(H35:H40)`
- Row 44 (MEDIAN): `=MEDIAN(H35:H40)`
- Row 45 (MEAN):   `=AVERAGE(H35:H40)`
- Row 46 (25th):   `=PERCENTILE(H35:H40, 0.25)`
- Row 47 (75th):   `=PERCENTILE(H35:H40, 0.75)`

Check the labels in column D/E for rows 42-47 to confirm the correct order of statistics. Adjust if the order differs.

## Step 3 – Weighted mean (H50:L50)

For each column H through L:
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```
This computes the weighted mean of Net capacity headroom using Staffed Bed Capacity as weights.

## Step 4 – Save and verify

1. Save the workbook with `wb.save('/root/output/result.xlsx')`.
2. Reopen the file and print all formula cells to confirm they were written correctly.
3. Spot-check a few formulas to make sure the references are correct.

## Important constraints
- Do NOT add new sheets, macros, VBA, external links, or helper tabs.
- Do NOT change any existing formatting.
- Only write formula strings; do not compute values in Python.
- Use openpyxl throughout.
- All formulas must use one of the approved lookup patterns (INDEX/MATCH preferred based on prior success).
- Use PERCENTILE (not PERCENTILE.INC or PERCENTILE.EXC) for compatibility.

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