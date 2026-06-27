# Task Instruction

Execute the following steps exactly:

## 1. Inspect the workbook
```python
import openpyxl, shutil, os
os.makedirs('/root/output', exist_ok=True)
wb = openpyxl.load_workbook('/root/data/workbook.xlsx')
print('Sheet names:', wb.sheetnames)

task = wb['Task']
data = wb['Data']

# Print Task sheet layout: rows 1-55, columns A-M
for r in range(1, 56):
    vals = []
    for c in range(1, 14):
        v = task.cell(row=r, column=c).value
        vals.append(str(v) if v is not None else '')
    print(f'Row {r:2d}: {" | ".join(vals)}')

print('\n--- Data sheet rows 18-40, cols A-M ---')
for r in range(18, 41):
    vals = []
    for c in range(1, 14):
        v = data.cell(row=r, column=c).value
        vals.append(str(v) if v is not None else '')
    print(f'Row {r:2d}: {" | ".join(vals)}')

# Also print Data sheet header rows 1-5 and column structure
print('\n--- Data sheet rows 1-20, cols A-M ---')
for r in range(1, 21):
    vals = []
    for c in range(1, 14):
        v = data.cell(row=r, column=c).value
        vals.append(str(v) if v is not None else '')
    print(f'Row {r:2d}: {" | ".join(vals)}')

wb.close()
```

## 2. Understand the structure
After inspecting, identify:
- The series codes in column D of the Task sheet (rows 12-17, 19-24, 26-31)
- The years in row 10 for columns H through L
- The layout of Data sheet rows 21-38 (what column has the series code, what row/column has years, and where the values are)
- What the three blocks represent (Latency Budget Preserved, Latency Budget Consumed, Covered Request Capacity)
- The six services and their row positions

## 3. Write formulas
Open the workbook with openpyxl (data_only=False to preserve formulas), and write formulas into the cells.

For the lookup formulas in H12:L17, H19:L24, H26:L31:
- Use INDEX/MATCH pattern. The exact formula depends on the Data sheet layout discovered in step 1.
- The formula should look up the series code from column D and the year from row 10 in the Data sheet rows 21:38.
- Example pattern (adjust column letters and row numbers based on actual layout):
  `=INDEX(Data!$B$21:$B$38,MATCH(1,($D12=Data!$A$21:$A$38)*(H$10=Data!$C$21:$C$38),0))` — but this is a CSE array formula. Prefer a simpler non-array approach:
  `=INDEX(Data!<value_column>,MATCH($D12,Data!<series_column>,0),MATCH(H$10,Data!<year_header_row>,0))`
  Adjust based on actual Data layout.

For H35:L40 (Net SLA buffer):
- Formula: `=(H12-H19)/H26*100` (adjusting row references for each of the 6 services, where row 12 is Latency Budget Preserved, row 19 is Latency Budget Consumed, row 26 is Covered Request Capacity)

For H42:L47 (statistics):
- H42: `=MIN(H35:H40)`
- H43: `=MAX(H35:H40)`
- H44: `=MEDIAN(H35:H40)`
- H45: `=AVERAGE(H35:H40)`
- H46: `=_xlfn.PERCENTILE.INC(H35:H40,0.25)`  — USE THE `_xlfn.` PREFIX
- H47: `=_xlfn.PERCENTILE.INC(H35:H40,0.75)`  — USE THE `_xlfn.` PREFIX

For H50:L50 (weighted mean):
- `=SUMPRODUCT(H35:H40,H26:H31)/SUM(H26:H31)`

## 4. Critical: _xlfn. prefix for PERCENTILE
The previous run failed because PERCENTILE.INC was not prefixed with `_xlfn.`. You MUST use `_xlfn.PERCENTILE.INC` in the formula strings. This is essential.

## 5. Save
Save to `/root/output/result.xlsx`. Do NOT change formatting, do NOT add sheets.

## 6. Validate
```python
import subprocess
result = subprocess.run(['python', '-m', 'pytest', '/root/test_output.py', '-v'], capture_output=True, text=True, cwd='/root')
print(result.stdout)
print(result.stderr)
```
If tests fail, read the error output carefully, re-inspect the workbook cells, and fix.

## Important Notes
- When writing formulas with openpyxl, just assign the formula string to cell.value (e.g., `cell.value = '=INDEX(...)'`)
- Make sure dollar signs are correct for mixed references: lock the series code column with $D and lock the year row with $10 where needed.
- After writing formulas, re-read a few cells to confirm they contain formula strings (not None).
- The Data sheet layout must be inspected first — do NOT guess column positions.
- If the Data sheet has a horizontal layout (years as columns), use a 2D INDEX/MATCH. If vertical, adjust accordingly.
- Preserve all existing cell values and formatting. Only write into the specified empty/yellow cells.

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