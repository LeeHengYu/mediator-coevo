# Task Instruction

## Task: Populate formulas in /root/data/workbook.xlsx and save to /root/output/result.xlsx

Follow these steps precisely:

### Step 0: Inspect the workbook
1. `mkdir -p /root/output`
2. Use openpyxl to open `/root/data/workbook.xlsx` and inspect:
   - Sheet `Task`: Read row 10 (the year headers in H10:L10). Read column D rows 12-17, 19-24, 26-31 (series codes). Read rows 35-40 column D (service names). Read H42:L47 labels (min/max/median/mean/p25/p75). Read row 50 label. Note any existing content, formatting, or fill colors.
   - Sheet `Data`: Read rows 21-38 completely. Understand the layout: which column has series codes, which row has years, and where the data values are. Print the first few columns and rows so you understand the exact structure (column letters and row numbers).
3. Print all findings before writing any formulas.

### Step 1: Lookup formulas in H12:L17, H19:L24, H26:L31

For each yellow cell in these three blocks, write an Excel formula that:
- Takes two inputs: the series code from column D of that row, and the year from row 10 of that column.
- Looks up data from sheet `Data` rows 21:38.
- Uses one of these patterns: INDEX+MATCH, VLOOKUP+MATCH, HLOOKUP+MATCH, or XLOOKUP+MATCH.

IMPORTANT: Determine the exact data layout on the Data sheet first. If data is arranged with series codes in a column and years in a row header, INDEX(MATCH,MATCH) is the most natural two-dimensional lookup. Use absolute references for the data range and mixed references where appropriate so formulas can be filled across rows and columns.

When writing formulas with openpyxl, set the cell's `.value` to the formula string (e.g., `'=INDEX(Data!$B$21:$F$38,MATCH($D12,Data!$A$21:$A$38,0),MATCH(H$10,Data!$B$20:$F$20,0))'`). Adjust column/row references based on actual inspection.

### Step 2: Net SLA Buffer in H35:L40 and summary stats in H42:L47

For H35:L40, the formula is:
`(Latency Budget Preserved - Latency Budget Consumed) / Covered Request Capacity * 100`

Determine which of the three blocks (H12:L17, H19:L24, H26:L31) corresponds to each metric by checking the labels in the Task sheet (likely around rows 11, 18, 25). Then write formulas referencing the appropriate cells. For example, if block 1 is Latency Budget Preserved (rows 12-17), block 2 is Latency Budget Consumed (rows 19-24), and block 3 is Covered Request Capacity (rows 26-31), then:
`H35 = (H12 - H19) / H26 * 100`

Adjust based on actual labels.

For H42:L47, write column-wise summary formulas over H35:L40:
- MIN: `=MIN(H35:H40)`
- MAX: `=MAX(H35:H40)`
- MEDIAN: `=MEDIAN(H35:H40)`
- MEAN (simple): `=AVERAGE(H35:H40)`
- 25th percentile: `=PERCENTILE(H35:H40,0.25)` or `=PERCENTILE.INC(H35:H40,0.25)`
- 75th percentile: `=PERCENTILE(H35:H40,0.75)` or `=PERCENTILE.INC(H35:H40,0.75)`

Match each row (42-47) to its label as shown on the sheet.

### Step 3: Weighted mean in H50:L50

For each column (H through L):
`=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)`

This computes the weighted mean of Net SLA Buffer percentages weighted by Covered Request Capacity.

### Step 4: Save and verify
1. Do NOT change any formatting, sheet names, or add sheets/macros.
2. Save to `/root/output/result.xlsx`.
3. Re-open the saved file and verify:
   - All formula cells contain formula strings (start with '=').
   - Spot-check a few formulas for correctness.
   - No extra sheets were added.
   - The file opens without errors.

### Critical Notes
- Read actual cell contents and layout BEFORE writing any formulas. Do not assume column/row positions.
- Use openpyxl with `data_only=False` when reading to see existing formulas.
- When saving, do not use `data_only` mode.
- Preserve all existing formatting by loading with openpyxl and not touching styles.
- Double-check that your Data sheet range references match the actual data layout exactly.

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