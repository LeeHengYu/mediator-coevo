# Task Instruction

Execute the following steps exactly, in order.

## 0 – Inspect the workbook
```bash
cp /root/data/workbook.xlsx /root/data/workbook_backup.xlsx
```
Open `/root/data/workbook.xlsx` with openpyxl (data_only=False) and print:
- Sheet names.
- Sheet `Task`: values/formulas in D12:D17, D19:D24, D26:D31 (series codes), and H10:L10 (year headers).
- Sheet `Task`: any existing content in H12:L17, H19:L24, H26:L31, H35:L40, H42:L47, H50:L50.
- Sheet `Data`: rows 21–38 – print the first row (row 21, headers) fully, and a sample of data rows to understand the layout (columns, series codes, year mapping).
- Identify which column in `Data` rows 21:38 holds the series code and which columns hold year values.

## 1 – Write a Python script that fills the workbook

Create `/root/solve.py` that does the following using **openpyxl** (preserve formatting with `load_workbook(filename, data_only=False)`):

### Step 1 – Lookup formulas in H12:L17, H19:L24, H26:L31

For every cell in these three blocks, write an INDEX/MATCH formula that:
- Looks up the series code from column D of the same row on sheet `Task`.
- Looks up the year from row 10 of the same column on sheet `Task`.
- Searches in `Data!$A$21:$A$38` for the series code (row match).
- Searches in `Data!$A$21:$<lastcol>$21` (or the actual header row) for the year (column match).
- Returns the intersection from the data block `Data!$A$21:$<lastcol>$38`.

Use the exact column letters discovered in step 0. The formula pattern per cell (e.g., H12) should be:
```
=INDEX(Data!$A$21:$<lastcol>$38, MATCH(D12, Data!$A$21:$A$38, 0), MATCH(H$10, Data!$A$21:$<lastcol>$21, 0))
```
Adjust the `Data` range references to match the actual layout you discovered. Make sure `D12` and `H$10` use appropriate relative/absolute references so that the series code reference is row-relative and the year reference is column-relative with a fixed row.

### Step 2 – Net budget buffer in H35:L40

The three blocks from Step 1 correspond to three data series per department (likely Committed Funding, Operating Spend, Approved Budget Base – verify from the workbook which block is which by reading labels near rows 11, 18, 25). The formula for each cell in H35:L40 is:
```
= (Committed_Funding_cell - Operating_Spend_cell) / Approved_Budget_Base_cell * 100
```
Map each department row in 35–40 to the corresponding rows in the three blocks (12–17, 19–24, 26–31). For example, H35 might be `=(H12-H19)/H26*100` – but verify which block is which from the labels.

### Step 2b – Summary statistics in H42:L47

For each column H through L:
- Row 42: `=MIN(H35:H40)`
- Row 43: `=MAX(H35:H40)`
- Row 44: `=MEDIAN(H35:H40)`
- Row 45: `=AVERAGE(H35:H40)`
- Row 46: `=PERCENTILE(H35:H40,0.25)` ← use legacy PERCENTILE, NOT PERCENTILE.INC
- Row 47: `=PERCENTILE(H35:H40,0.75)` ← use legacy PERCENTILE, NOT PERCENTILE.INC

Verify the row-to-statistic mapping by reading any labels in column A-G for rows 42-47. Assign formulas to match the label (min, max, median, mean, 25th pctl, 75th pctl) regardless of the row order I assumed above.

### Step 3 – Weighted mean in H50:L50

For each column H through L:
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```
(Approved Budget Base block is H26:L31; Net budget buffer percentages are H35:L40.)

### Save
Save to `/root/output/result.xlsx`. Create `/root/output/` directory if needed. Do NOT create new sheets. Do NOT alter formatting, fonts, fills, or column widths.

## 2 – Run and verify
```bash
mkdir -p /root/output
python3 /root/solve.py
```
Then re-open `/root/output/result.xlsx` with openpyxl and print the formulas in all modified cells to confirm they are present and syntactically correct. Also verify no cells contain `#NAME?`, `#REF!`, or `#VALUE!` literal strings.

## Critical constraints
- Use `PERCENTILE` not `PERCENTILE.INC` or `PERCENTILE.EXC`.
- Use `INDEX`/`MATCH` pattern for lookups.
- Do not add sheets, macros, VBA, external links, or helper tabs.
- Preserve all existing formatting.
- Read the actual workbook structure before writing any formulas – adapt column letters and row numbers to what you find, not to assumptions.

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