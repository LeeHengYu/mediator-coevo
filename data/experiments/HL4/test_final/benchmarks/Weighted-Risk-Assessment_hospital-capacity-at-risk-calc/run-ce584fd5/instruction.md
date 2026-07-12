# Task Instruction

Execute the following steps in a single Python script to produce `/root/output/result.xlsx`.

## Preliminary inspection
1. Open `/root/data/workbook.xlsx` with openpyxl (data_only=False) and inspect:
   - Sheet names (confirm `Task` and `Data` exist).
   - On sheet `Task`: read row 10 to find the year headers in columns H–L. Read column D rows 12–17, 19–24, 26–31 to find the series codes. Read the layout of rows 35–50 to understand labels.
   - On sheet `Data`: read rows 21–38 to understand the data layout (which row holds which series code, which columns hold which years).
   - Print all of this so we can see the exact structure before writing formulas.

2. After inspecting, close and re-open the workbook for editing.

## Step 1: Populate H12:L17, H19:L24, H26:L31 with lookup formulas

For each cell in these three blocks (rows 12–17, 19–24, 26–31; columns H–L):
- The lookup key is the series code in column D of the same row.
- The lookup year is in row 10 of the same column.
- The data source is `Data!$21:$38`.
- Use an INDEX/MATCH/MATCH pattern:
  ```
  =INDEX(Data!$A$21:$XFD$38,MATCH($D{row},Data!$A$21:$A$38,0),MATCH(H$10,Data!$A$21:$XFD$21,0))
  ```
  Adjust the exact ranges after inspection. The row-lookup array should be the column containing series codes on sheet Data (likely column A or B). The column-lookup array should be the header row (row 21) on sheet Data. If the series codes are not in column A, adjust accordingly. If the year headers are in a different row on Data, adjust accordingly.

**Important**: After inspecting the Data sheet layout, adapt the formula ranges precisely. The MATCH for the row should search the series-code column across rows 21–38. The MATCH for the column should search the year-header row. Use absolute references for the data range and mixed references ($D{row} for the series code column, {col}$10 for the year row) so formulas are consistent.

## Step 2: Net capacity headroom in H35:L40

For each cell in H35:L40 (6 hospital clusters × 5 years):
- Available Care Slots is in the first data block (H12:L17).
- Occupied Care Slots is in the second data block (H19:L24).
- Staffed Bed Capacity is in the third data block (H26:L31).
- The row offset within each block corresponds to the same hospital cluster.
- Formula: `=(H12-H19)/H26*100` (adjusted for each row/column).

Then in H42:L47, column-wise summary statistics:
- Row 42: `=MIN(H35:H40)` (minimum)
- Row 43: `=MAX(H35:H40)` (maximum)
- Row 44: `=MEDIAN(H35:H40)` (median)
- Row 45: `=AVERAGE(H35:H40)` (simple mean)
- Row 46: `=PERCENTILE(H35:H40,0.25)` (25th percentile)
- Row 47: `=PERCENTILE(H35:H40,0.75)` (75th percentile)

**Important**: Check the labels in column A/B/C/D of rows 42–47 to determine which row is which statistic. Map min/max/median/mean/25th/75th to the correct rows based on the actual labels. Do NOT assume the order above — read the labels first.

## Step 3: Weighted mean in H50:L50

For each column H–L:
```
=SUMPRODUCT(H35:H40,H26:H31)/SUM(H26:H31)
```
This computes the weighted mean of the Net capacity headroom percentages using Staffed Bed Capacity as weights.

## Final steps
- Do NOT change any formatting, do not add sheets, macros, VBA, external links, or helper tabs.
- Save to `/root/output/result.xlsx` (create `/root/output/` if needed).
- Re-open the saved file and print a sample of cells (e.g., H12, H35, H42, H50) to verify formulas were written correctly.

## Key cautions
- Use openpyxl with data_only=False to preserve and write formulas.
- When writing formulas as strings, ensure they start with `=`.
- Use the Translator or manual string formatting to replicate formulas across the range, or write each cell's formula individually.
- After inspection, if the Data sheet's series codes or year headers are in unexpected positions, adapt all formulas accordingly before writing.
- Ensure the INDEX/MATCH formula references are correct for the actual data layout discovered during inspection.

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