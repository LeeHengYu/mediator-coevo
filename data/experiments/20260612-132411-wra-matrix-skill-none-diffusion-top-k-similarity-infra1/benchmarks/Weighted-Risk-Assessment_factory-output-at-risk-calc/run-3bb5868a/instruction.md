# Task Instruction

Complete the following task to populate formulas in an Excel workbook.

## Setup
```bash
mkdir -p /root/output
pip install openpyxl
```

## Step 0: Inspect the workbook structure
Using Python with openpyxl, inspect `/root/data/workbook.xlsx`:
1. Read sheet `Task`: print all cell values for rows 1-55 and columns A-M. Pay special attention to:
   - Column D rows 12-31 (series codes)
   - Row 10 columns H-L (years)
   - Row 35-40 column D or nearby (plant names/identifiers)
   - Rows 42-47 column D or nearby (stat labels: min, max, median, mean, 25th/75th percentile)
   - Row 50 (Regional Output Council row)
   - Any existing formulas or values
2. Read sheet `Data`: print all cell values for rows 1-40 and columns A-Z (or however wide it extends). Focus on rows 21-38 to understand the data layout (column headers, series codes, how years map to columns).
3. Print the exact fill colors of cells in H12:L12 to confirm which cells are yellow.

Print everything clearly so you can understand the full structure before writing any formulas.

## Step 1: Populate lookup formulas in H12:L17, H19:L24, H26:L31

Based on your inspection, write formulas using INDEX+MATCH (or another allowed lookup pattern) in each yellow cell. Each formula should:
- Use the series code from column D of that row (e.g., $D12 for row 12)
- Use the year from row 10 of that column (e.g., H$10 for column H)
- Look up data from sheet `Data` rows 21:38

The exact ranges depend on the Data sheet layout. Determine:
- Which column on Data contains the series codes (for MATCH)
- Which row on Data contains the years (for MATCH)
- The data value range for INDEX

Use absolute/mixed references appropriately so formulas can be understood across the range. Use the pattern: `=INDEX(Data!<value_range>, MATCH(<series_code>, Data!<series_code_column>, 0), MATCH(<year>, Data!<year_row>, 0))`

Adapt the exact cell references based on what you find in the Data sheet.

## Step 2: Net production slack in H35:L40 and statistics in H42:L47

For H35:L40, the formula is:
`(Finished Output - Scrap And Rework) / Rated Production Capacity * 100`

Based on your inspection, identify which row ranges correspond to:
- Finished Output (likely H12:L17)
- Scrap And Rework (likely H19:L24)
- Rated Production Capacity (likely H26:L31)

For each cell in H35:L40, write: `=(H12-H19)/H26*100` (adjusting row references for each plant row).

For H42:L47, write column-wise statistical formulas over H35:L40:
- Row 42: MIN (e.g., `=MIN(H35:H40)`)
- Row 43: MAX (e.g., `=MAX(H35:H40)`)
- Row 44: MEDIAN (e.g., `=MEDIAN(H35:H40)`)
- Row 45: AVERAGE (e.g., `=AVERAGE(H35:H40)`)
- Row 46: PERCENTILE (e.g., `=PERCENTILE(H35:H40,0.25)`)
- Row 47: PERCENTILE (e.g., `=PERCENTILE(H35:H40,0.75)`)

Check the labels in column D/E/F/G of rows 42-47 to confirm the exact order of statistics.

## Step 3: Weighted mean in H50:L50

Use SUMPRODUCT with the Step 2 percentages (H35:H40) as values and Rated Production Capacity (H26:H31) as weights:
`=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)`

Apply this for each column H through L.

## Saving
- Load the workbook with `keep_vba=False` and `data_only=False` to preserve formulas.
- Write all formulas as strings starting with `=`.
- Do NOT change any formatting, sheet names, or structure.
- Save to `/root/output/result.xlsx`.

## Verification
After saving, reload `/root/output/result.xlsx` and print:
1. All formulas in H12:L17, H19:L24, H26:L31
2. All formulas in H35:L40
3. All formulas in H42:L47
4. All formulas in H50:L50

Confirm no cells are empty where formulas should exist. Confirm formulas reference the correct sheets and ranges.

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