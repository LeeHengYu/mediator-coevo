# Task Instruction

Execute the following steps to produce /root/output/result.xlsx from /root/data/workbook.xlsx.

## 0 – Inspect the workbook
```python
import openpyxl, pprint
wb = openpyxl.load_workbook('/root/data/workbook.xlsx')
for s in wb.sheetnames:
    print(f'=== {s} ===')
    ws = wb[s]
    for row in ws.iter_rows(min_row=1, max_row=min(ws.max_row, 50), values_only=False):
        for c in row:
            if c.value is not None:
                print(f'  {c.coordinate}: {repr(c.value)}')
```
Run this first and read the output carefully. Note:
- The series codes in column D of the Task sheet (rows 12-17, 19-24, 26-31).
- The years in row 10 (columns H-L).
- The layout of the Data sheet rows 21-38 (which column holds the series code, which columns hold year-indexed values, and what row 20 or the header row looks like).
- The labels in rows 35-40 (Net capacity headroom), 42-47 (statistics), and 50 (weighted mean).
- Any existing formulas or values already present.

## 1 – Write the lookup formulas (H12:L17, H19:L24, H26:L31)

Use the INDEX/MATCH pattern referencing the Data sheet. The exact formula depends on the Data sheet layout you discover in step 0. The general pattern is:

```
=INDEX(Data!<value_columns>, MATCH($D12, Data!<series_code_column>, 0), MATCH(H$10, Data!<year_header_row>, 0))
```

Adjust the ranges to cover Data rows 21:38 and the appropriate columns. Use absolute row/column references ($) so the formula can be filled across H-L and down each block.

Write these formulas as strings into the cells using openpyxl (e.g., `ws['H12'] = '=INDEX(...)'`).

## 2 – Net capacity headroom (H35:L40)

These six rows correspond to the six hospital clusters. The three input blocks are:
- Available Care Slots: rows 12-17
- Occupied Care Slots: rows 19-24
- Staffed Bed Capacity: rows 26-31

For each cell in H35:L40, write:
```
=(H12-H19)/H26*100
```
(adjusting row numbers for each cluster row).

Specifically:
- Row 35 uses rows 12, 19, 26
- Row 36 uses rows 13, 20, 27
- Row 37 uses rows 14, 21, 28
- Row 38 uses rows 15, 22, 29
- Row 39 uses rows 16, 23, 30
- Row 40 uses rows 17, 24, 31

## 3 – Statistics (H42:L47)

For each column H through L:
- Row 42 (MIN):    `=MIN(H35:H40)`
- Row 43 (MAX):    `=MAX(H35:H40)`
- Row 44 (MEDIAN): `=MEDIAN(H35:H40)`
- Row 45 (MEAN):   `=AVERAGE(H35:H40)`
- Row 46 (25th percentile): `=PERCENTILE(H35:H40,0.25)`
- Row 47 (75th percentile): `=PERCENTILE(H35:H40,0.75)`

**CRITICAL**: Use `PERCENTILE` (legacy), NOT `PERCENTILE.INC`. The latter requires an `_xlfn.` prefix in openpyxl and has caused #NAME? errors in past runs. Similarly use `MEDIAN` not `_xlfn.MEDIAN` etc.

Verify the row-to-statistic mapping by checking the labels in column D/E/F/G of those rows. If the order differs from what's listed above, adjust accordingly.

## 4 – Weighted mean (H50:L50)

For each column H through L:
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```
This computes the weighted mean of Net capacity headroom using Staffed Bed Capacity as weights.

## 5 – Save
```python
import os
os.makedirs('/root/output', exist_ok=True)
wb.save('/root/output/result.xlsx')
```

## 6 – Validate
Reload the saved workbook (data_only=False) and print all formulas in the Task sheet for cells H12:L50 to confirm they are present and correctly structured. Check that no cell is None or empty where a formula is expected.

## Important constraints
- Do NOT add sheets, macros, VBA, external links, or helper tabs.
- Do NOT change existing formatting.
- Do NOT use `_xlfn.` prefixed functions. Use legacy Excel function names: PERCENTILE, MEDIAN, AVERAGE, MIN, MAX, INDEX, MATCH, SUMPRODUCT, SUM.
- Adjust all cell references and ranges based on what you actually observe in step 0. Do not assume the layout — verify it first.

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