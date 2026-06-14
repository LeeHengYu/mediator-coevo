# Task Instruction

Execute the following steps in a single Python script using openpyxl to produce /root/output/result.xlsx.

## Preliminary
1. `mkdir -p /root/output`
2. Load `/root/data/workbook.xlsx` with `openpyxl.load_workbook('/root/data/workbook.xlsx')`. Do NOT use `data_only=True` — we want to write formulas.
3. Inspect the workbook to understand the layout:
   - On sheet `Task`: read row 10 to find the year headers in columns H–L (columns 8–12). Read column D (rows 12–17, 19–24, 26–31) to find the series codes for each block. Read any existing labels/structure in rows 35–50.
   - On sheet `Data`: read rows 21–38 to understand the data table structure (header row, series code column, year columns).
   - Print all of this to stdout so you can verify your understanding before writing formulas.

## Step 1: Lookup formulas in H12:L17, H19:L24, H26:L31
For each yellow cell in these three blocks, write an INDEX/MATCH formula that:
- Looks up the series code from column D of that row against the series-code column in `Data!` rows 21–38
- Looks up the year from row 10 of the current column against the year header row in `Data!`
- Pattern: `=INDEX(Data!<data_range>,MATCH(<series_code_cell>,Data!<series_code_column>,0),MATCH(<year_cell>,Data!<year_header_row>,0))`

IMPORTANT: Before writing formulas, determine the exact layout of the Data sheet:
- Which row is the header row with years?
- Which column has the series codes?
- What is the full rectangular data range?
Use absolute references (with $) for the data range, series-code column, and year header row. Use mixed references so the formula can be written per-cell correctly (or just write each cell individually with correct references).

## Step 2: Net budget buffer in H35:L40
The three blocks from Step 1 correspond to three metrics. Based on the task description and typical layout:
- Block 1 (rows 12–17): Committed Funding
- Block 2 (rows 19–24): Operating Spend  
- Block 3 (rows 26–31): Approved Budget Base

Verify this by reading the labels in the Task sheet (likely in column A or nearby). The formula for each cell in H35:L40 is:
`=(H12-H19)/H26*100` (adjusted for the corresponding row offsets for each department)

So for row 35: `=(<row12_cell>-<row19_cell>)/<row26_cell>*100`
For row 36: `=(<row13_cell>-<row20_cell>)/<row27_cell>*100`
...and so on for all 6 departments across columns H–L.

## Step 2 continued: Summary statistics in H42:L47
For each column (H through L):
- Row 42: `=MIN(H35:H40)` (minimum)
- Row 43: `=MAX(H35:H40)` (maximum)
- Row 44: `=MEDIAN(H35:H40)` (median)
- Row 45: `=AVERAGE(H35:H40)` (simple mean)
- Row 46: `=PERCENTILE(H35:H40,0.25)` (25th percentile)
- Row 47: `=PERCENTILE(H35:H40,0.75)` (75th percentile)

Verify the order of these statistics by reading any labels in column A/B/C/D for rows 42–47. Adjust the order to match the labels.

## Step 3: Weighted mean in H50:L50
For each column (H through L):
`=SUMPRODUCT(H35:H40,H26:H31)/SUM(H26:H31)`

This uses the Step 2 percentages as values and the Approved Budget Base block as weights.

## Final
- Save the workbook to `/root/output/result.xlsx` using `wb.save('/root/output/result.xlsx')`.
- After saving, reload the file and print the formula strings in a few sample cells (e.g., H12, H35, H42, H50) to verify they were written correctly.
- Do NOT add any new sheets, macros, VBA, or external links.
- Do NOT modify any existing formatting.

## Key Pitfalls to Avoid (from failed sibling task)
- The hospital-bedflow sibling task failed because cells were left empty (None). Make absolutely sure every cell in the specified ranges gets a formula written to it.
- Double-check that your INDEX/MATCH references point to the correct Data sheet ranges.
- Verify that the series codes in column D are non-empty for all rows you're referencing.
- After writing, re-read every target cell to confirm it contains a formula string (not None).

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