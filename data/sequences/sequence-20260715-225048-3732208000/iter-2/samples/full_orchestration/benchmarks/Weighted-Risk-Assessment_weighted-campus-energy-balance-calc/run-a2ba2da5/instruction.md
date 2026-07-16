# Task Instruction

## Task: Populate Excel formulas in /root/data/workbook.xlsx and save to /root/output/result.xlsx

### Step 0: Inspect the workbook structure
1. Open `/root/data/workbook.xlsx` using openpyxl (with data_only=False to see formulas).
2. Inspect sheet `Task`:
   - Read row 10 to find the years in columns H through L.
   - Read column D for rows 12–17, 19–24, 26–31 to find the series codes.
   - Read rows 35–40 column D to find campus names and the structure of the Net renewable balance block.
   - Read rows 42–47 column D/G to find the stat labels (min, max, median, mean, 25th, 75th percentile).
   - Read row 50 to understand the MCEC weighted mean row.
   - Note the exact row/column layout of the yellow cells and any existing content.
3. Inspect sheet `Data`:
   - Read rows 21–38 to understand the data layout: which column holds series codes, which row holds years, and where the numeric data lives.
   - Identify the exact column of series codes and the row of year headers in the Data sheet.
   - Note the range boundaries precisely.

Print all findings before proceeding.

### Step 1: Populate lookup formulas in H12:L17, H19:L24, H26:L31
For each cell in these three blocks, write an Excel formula using INDEX/MATCH (preferred) that:
- Uses the series code from column D of that row on sheet `Task`.
- Uses the year from row 10 of the corresponding column on sheet `Task`.
- Looks up data from sheet `Data` rows 21:38.

The exact formula pattern should be:
```
=INDEX(Data!<data_range>, MATCH($D<row>, Data!<series_code_column>, 0), MATCH(H$10, Data!<year_header_row>, 0))
```
Adjust the ranges based on what you found in Step 0. The `$D<row>` should use a mixed reference (column absolute, row relative to the current row). `H$10` should use column-relative, row-absolute reference.

IMPORTANT: Use the actual column letters and row numbers from the Data sheet. The data range for INDEX should cover the numeric data area only. The MATCH for series codes should reference the series code column. The MATCH for years should reference the year header row.

### Step 2: Net renewable balance formulas in H35:L40
Based on the block structure:
- Rows 12–17 likely correspond to one metric (e.g., Renewable Generation)
- Rows 19–24 likely correspond to another metric (e.g., Grid Consumption)
- Rows 26–31 likely correspond to Baseline Energy Demand

Verify which block is which by reading the labels. Then for each cell in H35:L40:
```
=(H12 - H19) / H26 * 100
```
(Adjust row references to match the correct campus row within each block. Row 35 uses the first campus from each block, row 36 uses the second, etc.)

For rows 42–47 (summary statistics), use these formulas across H42:L47:
- Minimum: `=MIN(H35:H40)` (or the equivalent column range)
- Maximum: `=MAX(H35:H40)`
- Median: `=MEDIAN(H35:H40)`
- Mean: `=AVERAGE(H35:H40)`
- 25th percentile: `=PERCENTILE(H35:H40, 0.25)`
- 75th percentile: `=PERCENTILE(H35:H40, 0.75)`

Match each stat to the correct row based on the labels found in Step 0.

### Step 3: Weighted mean in H50:L50
For each column H through L:
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```
This computes the weighted mean of Net renewable balance percentages, weighted by Baseline Energy Demand.

### Step 4: Save and validate
1. Create `/root/output/` directory if it doesn't exist.
2. Save the workbook to `/root/output/result.xlsx`.
3. Reopen the saved file and verify:
   - Sheets `Task` and `Data` exist (no extra sheets).
   - Cells H12, L17, H19, L24, H26, L31 contain formulas (strings starting with '=').
   - Cells H35, L40 contain formulas.
   - Cells H42, L47 contain formulas.
   - Cell H50 and L50 contain formulas.
   - No macros or VBA present.

### Critical Notes
- Do NOT use data_only=True when loading; you must preserve and write formulas.
- Do NOT add any new sheets.
- Do NOT change any existing formatting, just write formulas into the yellow cells.
- Use openpyxl to read and write the workbook.
- Print the Data sheet structure carefully before writing any formulas to ensure correct range references.

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