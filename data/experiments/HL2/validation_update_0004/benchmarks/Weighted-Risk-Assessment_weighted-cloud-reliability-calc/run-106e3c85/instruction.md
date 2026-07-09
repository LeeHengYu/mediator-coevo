# Task Instruction

Execute the following steps to populate and save the workbook.

## 0 – Inspect the workbook
```python
import openpyxl, pprint
wb = openpyxl.load_workbook('/root/data/workbook.xlsx')
for s in wb.sheetnames:
    print(f'--- {s} ---')
    ws = wb[s]
    print(f'  dims: {ws.dimensions}')
# Inspect Task sheet layout
ws = wb['Task']
print('\nTask sheet key rows/cols:')
for r in range(10, 52):
    vals = [ws.cell(r, c).value for c in range(1, 14)]
    print(f'  Row {r}: {vals}')
# Inspect Data sheet layout
ws2 = wb['Data']
print('\nData sheet rows 19-40:')
for r in range(19, 41):
    vals = [ws2.cell(r, c).value for c in range(1, 14)]
    print(f'  Row {r}: {vals}')
```
Print everything and study it before writing any formulas.

## 1 – Write lookup formulas (Step 1)

Based on prior successful runs, the Data sheet has:
- Row 21 as the header row with year values starting in some column (likely B or C onward).
- Column A containing series codes.
- Data rows in 21:38.

For each yellow cell in the three blocks H12:L17, H19:L24, H26:L31 on sheet Task:
- Column D of the current row holds the series code.
- Row 10 holds the year for each column (H–L).
- Use INDEX/MATCH/MATCH:
  `=INDEX(Data!$B$22:$<lastcol>$38, MATCH($D<row>,Data!$A$22:$A$38,0), MATCH(H$10,Data!$B$21:$<lastcol>$21,0))`

Adjust the exact column letters and row numbers after inspecting the actual Data sheet layout in step 0. Use mixed references: anchor the series-code column with `$D<row>` and the year row with `<col>$10` so the formula replicates correctly across the 5 columns × 6 rows of each block.

Repeat for all three blocks (rows 12-17, 19-24, 26-31), columns H-L.

## 2 – Net reliability gap (Step 2)

In H35:L40 (6 regions × 5 years), write formulas referencing the lookup blocks:
- Successful API Requests block: rows 12-17
- Failed API Requests block: rows 19-24
- Compute Capacity block: rows 26-31

Formula for cell H35:
`=(H12-H19)/H26*100`

Replicate across H35:L40 using the same row offsets (region 1 = row 12/19/26, region 2 = row 13/20/27, etc.).

## 3 – Summary statistics (rows 42-47)

For each column H–L:
- Row 42 (Min):    `=MIN(H35:H40)`
- Row 43 (Max):    `=MAX(H35:H40)`
- Row 44 (Median): `=MEDIAN(H35:H40)`
- Row 45 (Mean):   `=AVERAGE(H35:H40)`
- Row 46 (25th):   `=PERCENTILE(H35:H40,0.25)`
- Row 47 (75th):   `=PERCENTILE(H35:H40,0.75)`

Check the actual labels in column D/E of rows 42-47 to confirm the order (min, max, median, mean, 25th, 75th) and adjust if needed.

## 4 – Weighted mean (row 50)

For each column H–L:
`=SUMPRODUCT(H35:H40,H26:H31)/SUM(H26:H31)`

## 5 – Save

```python
import os
os.makedirs('/root/output', exist_ok=True)
wb.save('/root/output/result.xlsx')
```

## 6 – Verify

Reload `/root/output/result.xlsx` and print all formula cells in the Task sheet to confirm:
- H12:L31 contain INDEX/MATCH formulas
- H35:L40 contain reliability gap formulas
- H42:L47 contain stat formulas
- H50:L50 contain SUMPRODUCT formulas
- No extra sheets were added
- No macros or VBA

## Important constraints
- All values written to cells must be formula strings starting with '='. Do NOT evaluate them to numbers.
- Do not modify any existing formatting, sheet names, or other cell content.
- Do not add sheets, macros, VBA, external links, or helper tabs.
- Use openpyxl throughout.
- Adapt column letters and row numbers based on what you actually see in step 0. The numbers above are from prior runs but MUST be verified against the current workbook before writing.

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