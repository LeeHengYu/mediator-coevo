# Task Instruction

## Task: Populate formulas in /root/data/workbook.xlsx and save to /root/output/result.xlsx

### Setup
1. Create the output directory: `mkdir -p /root/output`
2. Install openpyxl if not already available: `pip install openpyxl`
3. Inspect the workbook structure thoroughly before making any changes:
   - Open `/root/data/workbook.xlsx` with openpyxl
   - Print the sheet names to confirm `Task` and `Data` exist
   - Print the contents of sheet `Task` rows 1-55, columns A-M, to understand the layout (especially column D for series codes, row 10 for years, and the yellow cell regions)
   - Print sheet `Data` rows 1-40, all populated columns, to understand the data source structure (especially rows 21-38)
   - Pay close attention to: exact column letters where years appear in row 10 on Task sheet, exact series codes in column D for rows 12-17, 19-24, 26-31, and the structure of the Data sheet (whether data is arranged for VLOOKUP/HLOOKUP/INDEX-MATCH)

### Step 1: Populate lookup formulas in H12:L17, H19:L24, H26:L31

For each cell in the ranges H12:L17, H19:L24, and H26:L31 on sheet `Task`:
- The formula must look up a value using TWO inputs: (1) the series code from column D of that row, and (2) the year from row 10 of that column (H10, I10, J10, K10, or L10).
- The source data is on sheet `Data` in rows 21:38.
- Use one of the allowed patterns: `INDEX(MATCH,MATCH)`, `VLOOKUP+MATCH`, `HLOOKUP+MATCH`, or `XLOOKUP+MATCH`.
- Before writing formulas, carefully determine:
  - Where on the Data sheet the series codes are (which column)
  - Where on the Data sheet the years are (which row)
  - What the data range is (rows 21:38, but identify exact columns)
  - Whether VLOOKUP or INDEX/MATCH is most natural given the layout
- Use absolute references for the lookup array/range on the Data sheet so formulas are robust.
- Write the formulas as strings (not computed values) so Excel will recalculate them.

### Step 2: Net budget buffer in H35:L40 and summary statistics in H42:L47

For H35:L40 (6 departments × 5 years):
- Formula: `(Committed Funding - Operating Spend) / Approved Budget Base * 100`
- Committed Funding values are in H12:L17
- Operating Spend values are in H19:L24  
- Approved Budget Base values are in H26:L31
- So for cell H35: `=(H12-H19)/H26*100`, and similarly for the rest of the 6×5 grid.
- Verify by checking that rows 35-40 correspond to the same 6 departments as rows 12-17, 19-24, 26-31.

For H42:L47 (summary statistics, column-wise over H35:L40):
- Row 42: MIN — e.g., `=MIN(H35:H40)` for column H
- Row 43: MAX — e.g., `=MAX(H35:H40)`
- Row 44: MEDIAN — e.g., `=MEDIAN(H35:H40)`
- Row 45: AVERAGE (simple mean) — e.g., `=AVERAGE(H35:H40)`
- Row 46: 25th percentile — e.g., `=PERCENTILE(H35:H40,0.25)`
- Row 47: 75th percentile — e.g., `=PERCENTILE(H35:H40,0.75)`
- **IMPORTANT**: Check the labels in column A/B/C/D for rows 42-47 to determine the exact order of these statistics. Do NOT assume the order above — match the labels.

### Step 3: Weighted mean in H50:L50

For each column H through L in row 50:
- Use SUMPRODUCT to calculate weighted mean
- Values = the Step 2 percentages (H35:H40 for column H, etc.)
- Weights = Approved Budget Base (H26:H31 for column H, etc.)
- Formula: `=SUMPRODUCT(H35:H40,H26:H31)/SUM(H26:H31)`

### Final Steps
1. Do NOT modify any existing formatting, do NOT add sheets, macros, VBA, external links, or helper tabs.
2. Save the workbook to `/root/output/result.xlsx`.
3. Verify the saved file by reopening it and printing the formula content of representative cells (e.g., H12, L17, H35, H42, H50) to confirm formulas were written correctly.
4. Also verify no extra sheets were added and the file opens without errors.

### Critical Notes
- Use openpyxl and open the workbook with `data_only=False` to preserve and write formulas.
- When writing formulas, they must be Excel formula strings starting with `=`.
- Carefully inspect the actual workbook layout BEFORE writing any formulas. Print row/column contents to understand the exact structure. The mapping of rows to statistics, the exact location of series codes, and the year row positions are all critical.
- If the Data sheet has a different layout than expected, adapt the lookup formula accordingly while staying within the allowed patterns.

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