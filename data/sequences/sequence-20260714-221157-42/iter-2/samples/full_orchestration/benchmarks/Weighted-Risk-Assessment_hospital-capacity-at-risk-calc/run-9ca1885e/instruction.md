# Task Instruction

You must update the workbook `/root/data/workbook.xlsx` and save the result to `/root/output/result.xlsx`. Follow these steps carefully:

## Preliminary Investigation

1. Create the output directory: `mkdir -p /root/output`
2. Install openpyxl if needed: `pip install openpyxl`
3. Write and run a Python script to inspect the workbook structure:
   - Read the `Task` sheet: print the contents of rows 10-50, columns D through L (especially column D for series codes and row 10 for years).
   - Read the `Data` sheet: print rows 21-38 completely (all columns with data), paying special attention to how series codes are stored (which column), and how years appear (which row, whether they are headers or in a specific row).
   - Print the exact cell values for Task!D12:D17, Task!D19:D24, Task!D26:D31 (the series codes used for lookup).
   - Print Task row 10 columns H through L (the year headers).
   - Print the Data sheet's row and column structure for rows 21-38: identify which column holds the series codes and which row holds the year headers.
   - Print Task!H35:L40 area and Task!H42:L47 area and Task!H50:L50 to see what's currently there.
   - Also print Data rows 1-5 to understand the header structure.

This investigation is CRITICAL. The previous failed run on a similar task got #N/A errors because the MATCH ranges didn't align with actual data layout.

## Step 1: Write Lookup Formulas in H12:L17, H19:L24, H26:L31

Based on the investigation, write a Python script using openpyxl to insert formulas into the yellow cells.

For each cell in these ranges, the formula must:
- Use the series code from column D of the SAME row on the Task sheet
- Use the year from row 10 of the SAME column on the Task sheet  
- Look up the value from the Data sheet rows 21:38
- Use one of these patterns: INDEX/MATCH, VLOOKUP/MATCH, HLOOKUP/MATCH, or XLOOKUP/MATCH

IMPORTANT: You must determine from the Data sheet inspection:
- Whether series codes are in a column (use VLOOKUP or INDEX/MATCH with vertical lookup) or a row (use HLOOKUP)
- Whether years are in a row (column headers) or a column
- The exact range references for the Data sheet lookup area
- Whether the series codes in Task!D column match EXACTLY (string, spacing, case) with the codes in the Data sheet. Print both side by side to verify.

Use INDEX/MATCH as the preferred pattern since it's most flexible:
`=INDEX(Data!<data_range>, MATCH($D12, Data!<series_code_range>, 0), MATCH(H$10, Data!<year_range>, 0))`

Adjust the exact ranges based on your inspection findings. Use absolute row/column references ($) appropriately so formulas copy correctly across the range.

## Step 2: Net Capacity Headroom (H35:L40) and Statistics (H42:L47)

For H35:L40, the formula is:
`(Available Care Slots - Occupied Care Slots) / Staffed Bed Capacity * 100`

You need to determine which of the three blocks (H12:L17, H19:L24, H26:L31) corresponds to which metric. Check the labels in the Task sheet (likely in column B or C near rows 12, 19, 26) to identify which block is Available Care Slots, Occupied Care Slots, and Staffed Bed Capacity.

For example, if H12:L17 = Available Care Slots, H19:L24 = Occupied Care Slots, H26:L31 = Staffed Bed Capacity, then:
`H35 = (H12 - H19) / H26 * 100`

For H42:L47, calculate column-wise statistics over H35:L40:
- H42: `=MIN(H35:H40)` (or whichever row is minimum - check the labels in column B/C/D near rows 42-47)
- H43: `=MAX(H35:H40)`
- H44: `=MEDIAN(H35:H40)`
- H45: `=AVERAGE(H35:H40)`
- H46: `=PERCENTILE(H35:H40, 0.25)`
- H47: `=PERCENTILE(H35:H40, 0.75)`

Match the order of statistics to whatever labels exist in the Task sheet near rows 42-47.

## Step 3: Weighted Mean (H50:L50)

For H50:L50, use SUMPRODUCT with Staffed Bed Capacity as weights:
`=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)`

This computes the weighted mean of the Net capacity headroom percentages, weighted by Staffed Bed Capacity.

## Final Steps

1. Save the workbook to `/root/output/result.xlsx`
2. Verify by reopening the saved file and printing the formula cells to confirm formulas were written correctly.
3. Do NOT change any formatting, do NOT add sheets, macros, VBA, external links, or helper tabs.
4. When using openpyxl, load with `load_workbook(filename, data_only=False)` to preserve existing formulas elsewhere, and be careful not to overwrite non-yellow cells.

## Key Pitfall to Avoid
The #N/A failure from a similar task was caused by MATCH ranges not aligning with actual data. Triple-check that your MATCH range for series codes contains the actual series codes, and your MATCH range for years contains the actual year values. Print and compare them explicitly before writing formulas.

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