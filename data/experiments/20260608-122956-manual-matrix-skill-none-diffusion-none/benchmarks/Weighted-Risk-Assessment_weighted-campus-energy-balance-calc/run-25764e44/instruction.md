# Task Instruction

Execute the following steps in a single Python script using openpyxl to populate formulas in /root/data/workbook.xlsx and save the result to /root/output/result.xlsx.

## Pre-work: Inspect the workbook

1. Load `/root/data/workbook.xlsx` with `openpyxl.load_workbook('/root/data/workbook.xlsx')` — do NOT use `data_only=True`.
2. Print the sheet names to confirm `Task` and `Data` exist.
3. On the `Data` sheet, print:
   - Row 21 values (columns A through Z) to identify the header row structure (year headers, series code column, etc.)
   - Column A values for rows 21–38 to see series codes
   - Column B values for rows 21–38 to see any labels
   - Print all values in rows 21–38 for columns A–Z to fully understand the data layout.
4. On the `Task` sheet, print:
   - Row 10 values (columns A through L) to see year headers
   - Column D values for rows 12–31 to see the series codes used in lookups
   - Column A–G values for rows 12–50 to understand the full layout
   - The current contents of cells H12, H19, H26, H35, H42, H50 to see if anything is already there

## Step 1: Populate lookup formulas in H12:L17, H19:L24, H26:L31

Based on the inspection, construct INDEX/MATCH formulas. The general pattern should be:

```
=INDEX(Data!<data_range>, MATCH($D12, Data!<series_code_column>, 0), MATCH(H$10, Data!<year_header_row>, 0))
```

Critical details:
- Identify which column on the Data sheet contains the series codes (likely column A or B). Use the inspection output.
- Identify which row on the Data sheet contains the year headers. Use the inspection output.
- The data range for INDEX should cover the full rectangular block of numeric data in rows 21:38.
- Use `$D12` (column-absolute) for the series code reference so it stays fixed when filling across columns.
- Use `H$10` (row-absolute) for the year reference so it stays fixed when filling down rows.
- Make sure the MATCH ranges for series codes and year headers align with the first column/row of the INDEX data range.

Write the formula string into each cell in the three blocks (H12:L17, H19:L24, H26:L31) using a loop. Adjust the row references (e.g., $D12, $D13, ...) and column references (H$10, I$10, ...) appropriately for each cell.

## Step 2: Net renewable balance in H35:L40

The formula for each cell is:
```
=(H12 - H19) / H26 * 100
```
where H12 corresponds to Renewable Generation, H19 to Grid Consumption, and H26 to Baseline Energy Demand for the same campus and year. Adjust row references per campus (rows 35–40 map to the 6 campuses in rows 12–17, 19–24, 26–31 respectively).

For each cell in H35:L40, write the formula referencing the corresponding cells from the three lookup blocks.

## Step 2 continued: Statistics in H42:L47

For each column H through L:
- Row 42: `=MIN(H35:H40)`
- Row 43: `=MAX(H35:H40)`
- Row 44: `=MEDIAN(H35:H40)`
- Row 45: `=AVERAGE(H35:H40)`
- Row 46: `=PERCENTILE(H35:H40,0.25)` — try `PERCENTILE.INC` if the verifier might prefer it; based on cross-task feedback, use `PERCENTILE.INC` for safety.
- Row 47: `=PERCENTILE(H35:H40,0.75)` — similarly use `PERCENTILE.INC`.

IMPORTANT: Check the existing labels in column A or nearby columns for rows 42–47 to confirm the order (min, max, median, mean, 25th, 75th). Adjust the row assignments if the order differs.

## Step 3: Weighted mean in H50:L50

For each column H through L:
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```
This computes the weighted mean of the net renewable balance percentages using Baseline Energy Demand as weights.

## Save

1. Create `/root/output/` directory if it doesn't exist.
2. Save the workbook to `/root/output/result.xlsx`.

## Post-save verification

1. Reload `/root/output/result.xlsx` (without data_only) and print the formula contents of:
   - H12, L17 (first and last of block 1)
   - H19, L24 (first and last of block 2)
   - H26, L31 (first and last of block 3)
   - H35, L40 (first and last of derived block)
   - H42, H47 (stats)
   - H50, L50 (weighted mean)
2. Confirm none are None.

Do NOT add any new sheets, macros, VBA, or external links. Do not modify formatting. Only write formula strings into the specified cells.

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