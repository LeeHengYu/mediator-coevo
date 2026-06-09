# Task Instruction

## Task: Populate formulas in `/root/data/workbook.xlsx` and save to `/root/output/result.xlsx`

### Preparation
1. Create `/root/output/` directory if it doesn't exist.
2. Open `/root/data/workbook.xlsx` and inspect both sheets (`Task` and `Data`) thoroughly before making any changes.
3. On sheet `Task`, identify:
   - The series codes in column D for rows 12-17, 19-24, 26-31, 35-40.
   - The years in row 10 for columns H through L.
   - The structure of the yellow cells in H12:L17, H19:L24, H26:L31.
4. On sheet `Data`, inspect rows 21-38 to understand the data layout:
   - Determine whether data is organized with series codes in a column and years in a row, or vice versa.
   - Identify the exact column that contains series codes and the exact row that contains years.
   - Note the exact cell references for the data range.

### Step 1: Lookup Formulas in H12:L17, H19:L24, H26:L31

For each yellow cell in these three blocks, write a formula that:
- Takes the series code from column D of that row on sheet `Task`
- Takes the year from row 10 of that column on sheet `Task`
- Looks up the corresponding value from sheet `Data` rows 21:38

Use one of the allowed patterns: `INDEX(MATCH, MATCH)`, `VLOOKUP+MATCH`, `HLOOKUP+MATCH`, or `XLOOKUP+MATCH`.

**Important**: Before writing formulas, carefully determine:
- Whether the Data sheet has years across columns (horizontal) or down rows (vertical) in the range rows 21:38.
- Which column/row contains the series codes and which contains the years.
- Use appropriate absolute/mixed references so formulas can be placed across the H:L columns and down the rows correctly. Lock the lookup column (series code in column D) row-relatively and the year row (row 10) column-relatively.

For example, if using INDEX-MATCH-MATCH and the Data sheet has series codes in a column and years in a header row within rows 21:38, the formula pattern would be something like:
`=INDEX(Data!<data_range>, MATCH($D12, Data!<series_code_column>, 0), MATCH(H$10, Data!<year_row>, 0))`

Adjust the exact ranges based on what you observe in the Data sheet. The key is that $D12 locks column D (series code) and H$10 locks row 10 (year), allowing the formula to be correctly placed in all cells of each block.

### Step 2: Net Budget Buffer in H35:L40 and Summary Statistics in H42:L47

**H35:L40 - Net Budget Buffer formula:**
For each cell, compute:
`= (Committed Funding - Operating Spend) / Approved Budget Base * 100`

where:
- Committed Funding values are in H12:L17 (the first block)
- Operating Spend values are in H19:L24 (the second block)  
- Approved Budget Base values are in H26:L31 (the third block)

So for cell H35: `=(H12-H19)/H26*100`
For cell H36: `=(H13-H20)/H27*100`
...and so on for all 6 rows × 5 columns.

**IMPORTANT**: Verify which block corresponds to which metric by reading the labels on the Task sheet. The mapping above (first block = Committed Funding, second = Operating Spend, third = Approved Budget Base) is an assumption - confirm it by checking the labels in the Task sheet before writing formulas. Adjust if the actual labels differ.

**H42:L47 - Summary Statistics (column-wise over H35:L40):**
- Row 42: `=MIN(H35:H40)` (minimum) — but check the label in column D/E/F/G to confirm which statistic goes in which row
- Row 43: `=MAX(H35:H40)` (maximum)
- Row 44: `=MEDIAN(H35:H40)` (median)
- Row 45: `=AVERAGE(H35:H40)` (simple mean)
- Row 46: `=PERCENTILE(H35:H40, 0.25)` (25th percentile)
- Row 47: `=PERCENTILE(H35:H40, 0.75)` (75th percentile)

**IMPORTANT**: Read the actual labels in the Task sheet for rows 42-47 to determine the correct order of min, max, median, mean, 25th percentile, and 75th percentile. Place each formula in the row matching its label.

### Step 3: Weighted Mean in H50:L50

For each column (H through L), calculate the weighted mean using SUMPRODUCT:
`=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)`

This uses the Net Budget Buffer percentages (H35:H40) as values and the Approved Budget Base (H26:H31) as weights.

### Final Steps
1. After writing all formulas, save the workbook to `/root/output/result.xlsx`.
2. Re-open the saved file and verify:
   - Sheets `Task` and `Data` exist (no extra sheets).
   - Spot-check a few formula cells to confirm they contain formulas (not just values).
   - Verify that the formulas reference the correct cells and sheets.
3. Do NOT add any new sheets, macros, VBA, external links, or helper tabs.
4. Do NOT change any existing formatting.

### Technical Notes
- Use `openpyxl` to read and write the workbook.
- When writing formulas, assign them as strings starting with `=` to cell values.
- Make sure to preserve existing content and formatting by loading with `openpyxl` and only modifying the target cells.
- Use `keep_vba=False` (default) and do not strip formatting when saving.

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