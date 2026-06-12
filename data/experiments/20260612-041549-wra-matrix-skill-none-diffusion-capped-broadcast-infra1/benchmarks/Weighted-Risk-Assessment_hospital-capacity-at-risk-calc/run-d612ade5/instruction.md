# Task Instruction

Execute the following steps to produce /root/output/result.xlsx from /root/data/workbook.xlsx.

## 0 – Inspect the workbook
```bash
pip install openpyxl --quiet
```
```python
import openpyxl, pprint
wb = openpyxl.load_workbook('/root/data/workbook.xlsx')
for s in wb.sheetnames:
    print(f'--- {s} ---')
    ws = wb[s]
    print(f'  dims: {ws.dimensions}')
    # Print first 50 rows to understand layout
    for row in ws.iter_rows(min_row=1, max_row=50, values_only=False):
        vals = [(c.coordinate, c.value) for c in row if c.value is not None]
        if vals:
            print(' ', vals)
```
Read the output carefully. Identify:
- The series codes in column D of the Task sheet (rows 12-17, 19-24, 26-31).
- The year headers in row 10 (columns H-L).
- The Data sheet layout, especially rows 21-38 and their column structure (which column holds the series code, which holds the year, which holds the value, or if it's a matrix with years as columns).
- The existing content of H35:L40, H42:L47, H50:L50 to see if anything is already there.

## 1 – Populate lookup formulas in H12:L17, H19:L24, H26:L31

Based on the Data sheet layout (rows 21:38), construct lookup formulas. Use INDEX/MATCH or VLOOKUP/MATCH pattern.

IMPORTANT: The Data sheet likely has a table where one column contains series codes and columns contain year-based data. Determine the exact structure before writing formulas.

For each cell in the yellow ranges, the formula should look up:
- The series code from column D of the same row on the Task sheet
- The year from row 10 of the same column on the Task sheet
- Against the data in Data!rows 21:38

Typical pattern (adjust column/row references after inspection):
```
=INDEX(Data!$B$21:$Z$38, MATCH($D12, Data!$A$21:$A$38, 0), MATCH(H$10, Data!$B$20:$Z$20, 0))
```
Adjust the exact ranges based on what you find in the Data sheet.

Use openpyxl to write these formulas. Make sure:
- Row references use $ for absolute refs on the Data ranges.
- Column D ref is $D (absolute column, relative row).
- Row 10 ref is absolute row ($10), relative column.

## 2 – Net capacity headroom in H35:L40

The formula is: (Available Care Slots - Occupied Care Slots) / Staffed Bed Capacity * 100

Identify which of the three blocks (H12:L17, H19:L24, H26:L31) corresponds to:
- Available Care Slots
- Occupied Care Slots  
- Staffed Bed Capacity

Look at the labels in column D or nearby cells. Then for each cell in H35:L40, write a formula like:
```
=(H12-H19)/H26*100
```
(adjusting row offsets so each of the 6 hospital clusters maps correctly)

## 3 – Statistics in H42:L47

For each column (H through L), calculate column-wise stats over H35:L40:
- Row 42: MIN
- Row 43: MAX  
- Row 44: MEDIAN
- Row 45: AVERAGE
- Row 46: PERCENTILE (25th) – use legacy PERCENTILE function, NOT PERCENTILE.INC
- Row 47: PERCENTILE (75th) – use legacy PERCENTILE function

Check the Task sheet labels in column D/E for rows 42-47 to confirm the order. Adjust accordingly.

Example formulas for column H:
```
=MIN(H35:H40)
=MAX(H35:H40)
=MEDIAN(H35:H40)
=AVERAGE(H35:H40)
=PERCENTILE(H35:H40,0.25)
=PERCENTILE(H35:H40,0.75)
```

## 4 – Weighted mean in H50:L50

Use SUMPRODUCT with the headroom percentages (H35:H40) as values and Staffed Bed Capacity (H26:H31) as weights:
```
=SUMPRODUCT(H35:H40,H26:H31)/SUM(H26:H31)
```

## 5 – Save

```python
import os
os.makedirs('/root/output', exist_ok=True)
wb.save('/root/output/result.xlsx')
```

## 6 – Validate

Re-open the saved file and verify:
- Cells H12, L17, H19, L24, H26, L31 contain formula strings (start with '=').
- Cells H35, L40 contain formula strings.
- Cells H42:L47 contain formula strings with correct function names.
- Cell H50 contains a SUMPRODUCT formula.
- No new sheets were added.
- Print all formula strings for manual review.

Also run the verifier if available:
```bash
cd /root && python -m pytest test_output.py -v 2>&1 | head -80
```

## Critical Notes
- Use legacy `PERCENTILE` not `PERCENTILE.INC` or `PERCENTILE.EXC`.
- Do NOT add sheets, macros, VBA, external links, or helper tabs.
- Do NOT change existing formatting.
- Inspect the Data sheet thoroughly before writing any formulas – get the exact row/column layout right.
- If the Data sheet has years as row headers (horizontal layout), you may need HLOOKUP+MATCH or a transposed INDEX/MATCH.

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