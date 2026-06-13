# Task Instruction

Execute the following steps precisely to complete the task.

## 0. Setup
```bash
mkdir -p /root/output
cp /root/data/workbook.xlsx /root/output/result.xlsx
```

## 1. Inspect the workbook structure
Open `/root/output/result.xlsx` with openpyxl and inspect:
- Sheet names (should include `Task` and `Data`).
- On sheet `Task`: read row 10 to see the year headers in columns H–L. Read column D rows 12–17, 19–24, 26–31 to see the series codes. Read rows 35–40 to see hospital names or labels. Read row 50 for the MHN label. Read any existing content/formulas in H42:L47 labels (min, max, median, mean, 25th, 75th percentile labels in column D or G).
- On sheet `Data`: read rows 21–38 to understand the data layout — identify which column holds the series code, which row holds years, and the orientation of the data (whether years are in columns or rows).

Print all of this information so you understand the exact layout before writing any formulas.

## 2. Populate lookup formulas in H12:L17, H19:L24, H26:L31

Using openpyxl, write Excel formulas (as strings) into each cell in the three blocks. For each cell at row `r`, column `c` (where H=8, I=9, J=10, K=11, L=12):

- The series code is in column D of the same row: reference `$D{r}` (dollar-sign on column to lock it).
- The year is in row 10 of the same column: reference `{col_letter}$10` (dollar-sign on row to lock it).
- The data source is on sheet `Data` in rows 21:38.

Based on the data layout you discovered in step 1, choose the appropriate pattern. The most likely pattern if Data has series codes in one column and years across columns:
```
=INDEX(Data!$B$21:$Z$38, MATCH($D{r}, Data!$A$21:$A$38, 0), MATCH({col}$10, Data!$B$20:$Z$20, 0))
```
Adjust the exact ranges based on what you find in step 1. The key requirements:
- Use one of: VLOOKUP+MATCH, HLOOKUP+MATCH, XLOOKUP+MATCH, or INDEX+MATCH.
- Two inputs: series code from column D of current row, year from row 10.
- Source is Data rows 21:38.

IMPORTANT: Verify the exact column where series codes live on the Data sheet and the exact row where years are headers. Adjust all range references accordingly.

## 3. Net patient flow formulas in H35:L40

For each cell in H35:L40, the formula computes:
`(Patient Admissions - Patient Discharges) / Effective Bed Capacity * 100`

Based on the block layout:
- H12:L17 is likely one metric (e.g., Patient Admissions)
- H19:L24 is likely another metric (e.g., Patient Discharges)
- H26:L31 is likely Effective Bed Capacity

Verify which block corresponds to which metric by reading the labels. Then for each cell at row offset `i` (0–5) in H35:L40:
```
=({admissions_cell} - {discharges_cell}) / {capacity_cell} * 100
```
For example if admissions are in rows 12–17, discharges in 19–24, capacity in 26–31:
```
=(H12-H19)/H26*100
```
for cell H35, and similarly offset for each row and column.

## 4. Summary statistics in H42:L47

For each column (H through L), write formulas for the six statistics over the Net patient flow range (e.g., H35:H40):
- Row 42: `=MIN(H35:H40)`
- Row 43: `=MAX(H35:H40)`
- Row 44: `=MEDIAN(H35:H40)`
- Row 45: `=AVERAGE(H35:H40)`
- Row 46: `=PERCENTILE(H35:H40, 0.25)` or `=PERCENTILE.INC(H35:H40, 0.25)`
- Row 47: `=PERCENTILE(H35:H40, 0.75)` or `=PERCENTILE.INC(H35:H40, 0.75)`

IMPORTANT: Check the labels in column D/G for rows 42–47 to determine the exact order of min, max, median, mean, 25th percentile, 75th percentile. Match the formula to the label in each row.

## 5. Weighted mean in H50:L50

For each column (H through L):
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```
This uses SUMPRODUCT with the Net patient flow percentages as values and Effective Bed Capacity as weights.

## 6. Save and verify
- Save the workbook to `/root/output/result.xlsx`.
- Re-open it and print all formula cells to verify they are correctly written.
- Ensure no new sheets were added, no macros, no external links.
- Verify the formulas reference the correct cells by spot-checking a few.

## Critical Notes
- Use openpyxl to write formulas as strings (e.g., cell.value = '=INDEX(...)').
- Do NOT use data_only mode when writing.
- Do NOT alter formatting, column widths, colors, or any existing content outside the specified cells.
- Do NOT add sheets or helper columns.
- Read the actual workbook structure carefully before writing ANY formula. The exact row/column references depend on the actual layout.

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
Task metadata: author_email=catpaw@meituan.com, author_name=CatPaw Task Engineer, category=spreadsheet-formula-reuse, difficulty=easy, tags=[excel, formulas, lookup, statistics, weighted-mean].
Verifier config: timeout_sec=600.0.