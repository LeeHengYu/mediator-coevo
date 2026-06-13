# Task Instruction

Execute the following steps in a single Python script using openpyxl.

## 0. Inspect the workbook
```python
import openpyxl, os, shutil

src = '/root/data/workbook.xlsx'
dst = '/root/output/result.xlsx'
os.makedirs('/root/output', exist_ok=True)
shutil.copy2(src, dst)

wb = openpyxl.load_workbook(dst)
ws_task = wb['Task']
ws_data = wb['Data']
```

Before writing any formulas, inspect and print:
- `ws_task` rows 10-50, columns D-L (to see series codes in col D, years in row 10, row labels for rows 12-17, 19-24, 26-31, 35-40, 42-47, 50).
- `ws_data` rows 19-40, columns A-Z (to find the data table header row and structure — identify which row has years, which column has series codes, and the data range).

Print all these values so you can confirm the exact layout before writing formulas.

## 1. Step 1 — Lookup formulas in H12:L17, H19:L24, H26:L31

For each cell in these three blocks, write an INDEX/MATCH/MATCH formula that:
- Looks up the series code from column D of the current row against the series codes column on sheet `Data` (within rows 21:38).
- Looks up the year from row 10 of the current column against the year header row on sheet `Data`.
- Returns the intersection value.

Use the inspected layout to determine:
- The exact column on `Data` that holds series codes (likely column A or B).
- The exact row on `Data` that holds years (likely row 20 or the row just above row 21).
- The data range on `Data` (the rectangular block of values).

Write the formula as a string, e.g.:
```
=INDEX(Data!C21:G38, MATCH(D12, Data!B21:B38, 0), MATCH(H10, Data!C20:G20, 0))
```
Adjust column letters and row numbers based on what you actually see in the inspection.

Use absolute references for the data range and lookup arrays so they don't shift, but keep the D-column reference (series code) and row-10 reference (year) as relative/mixed references that vary correctly across the block.

Loop over all three blocks (rows 12-17, 19-24, 26-31) and columns H-L (columns 8-12).

## 2. Step 2a — Net reliability gap in H35:L40

From the inspection, identify which row blocks correspond to:
- Successful API Requests (likely rows 12-17)
- Failed API Requests (likely rows 19-24)
- Compute Capacity (likely rows 26-31)

Confirm by reading the labels in the worksheet. Then for each cell in H35:L40, write a formula:
```
=(H12-H19)/H26*100
```
adjusted so that each row in 35-40 maps to the corresponding region row in the three blocks above (row 35↔row 12/19/26, row 36↔row 13/20/27, etc.), and each column H-L stays aligned.

## 2b — Summary statistics in H42:L47

Read the labels in cells around rows 42-47 (column D or nearby) to confirm which statistic goes in which row. Expected mapping (verify against actual labels):
- Row 42: MIN
- Row 43: MAX
- Row 44: MEDIAN
- Row 45: AVERAGE (simple mean)
- Row 46: 25th percentile (PERCENTILE)
- Row 47: 75th percentile (PERCENTILE)

For each column H-L, write the corresponding formula referencing H35:H40 (or the appropriate column's 35:40 range). Examples:
```
=MIN(H35:H40)
=MAX(H35:H40)
=MEDIAN(H35:H40)
=AVERAGE(H35:H40)
=PERCENTILE(H35:H40,0.25)
=PERCENTILE(H35:H40,0.75)
```

## 3. Step 3 — Weighted mean in H50:L50

For each column H-L, write a SUMPRODUCT formula:
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```
This computes the weighted mean of the net reliability gap values (H35:H40) weighted by Compute Capacity (H26:H31).

## 4. Save
```python
wb.save(dst)
```

## Critical reminders
- Do NOT skip the inspection step. Print the actual cell values before writing any formulas.
- Adjust all row/column references based on the actual inspected layout.
- Ensure `wb.save(dst)` is called at the very end.
- Do not add sheets, macros, VBA, external links, or helper tabs.
- Do not modify existing formatting.
- After saving, re-open the file and spot-check a few cells (e.g., H12, H35, H42, H50) to confirm they contain formula strings (not None).

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