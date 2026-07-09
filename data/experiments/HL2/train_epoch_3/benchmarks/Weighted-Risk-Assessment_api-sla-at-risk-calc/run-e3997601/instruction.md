# Task Instruction

Execute the following steps in order.

## 1 – Inspect the workbook
```
cd /root/data
python3 - <<'PY'
import openpyxl, json
wb = openpyxl.load_workbook('workbook.xlsx', data_only=False)
for s in wb.sheetnames:
    print(f'--- Sheet: {s} ---')
    ws = wb[s]
    print(f'  Dimensions: {ws.dimensions}')

# Task sheet: print rows 1-55, cols A-M
ws = wb['Task']
for r in range(1, 56):
    vals = []
    for c in range(1, 14):
        cell = ws.cell(r, c)
        v = cell.value
        vals.append(str(v) if v is not None else '')
    print(f'  Row {r:2d}: {vals}')

# Data sheet: print rows 1-45, cols A-Z (enough to see structure)
ws2 = wb['Data']
for r in range(1, 45):
    vals = []
    for c in range(1, 27):
        cell = ws2.cell(r, c)
        v = cell.value
        vals.append(str(v) if v is not None else '')
    print(f'  Row {r:2d}: {vals}')
PY
```
Read the output carefully. Identify:
- The series codes in column D of the Task sheet for rows 12-17, 19-24, 26-31.
- The years in row 10 (columns H-L).
- The layout of the Data sheet rows 21-38 (which column holds the series code, which holds the year, which holds the value).
- What is already in H35:L40, H42:L47, H50:L50.
- The labels for rows 35-40 (the six services) and rows 42-47 (min, max, median, mean, 25th, 75th).

## 2 – Build and write formulas with openpyxl
Create a Python script that:

a) Opens `/root/data/workbook.xlsx` preserving formatting (do NOT use data_only).

b) **Step 1 – Lookup formulas (H12:L17, H19:L24, H26:L31)**
   For each cell in these ranges, write an INDEX/MATCH formula that looks up the value from the Data sheet rows 21:38. The formula should match on two criteria:
   - The series code from column D of the current row on the Task sheet.
   - The year from row 10 on the Task sheet.
   Determine from the Data sheet inspection which column contains the series codes and which row/column contains the years, then construct the formula accordingly.
   
   Use the pattern: `=INDEX(Data!<value_column_range>,MATCH(<series_code_cell>,Data!<series_code_range>,0))` if data is arranged with years in separate columns, or a two-dimensional INDEX/MATCH if needed. Adapt based on actual layout.

c) **Step 2 – Net SLA buffer (H35:L40)**
   The formula is: `(Latency Budget Preserved - Latency Budget Consumed) / Covered Request Capacity * 100`
   - Identify which of the three blocks (H12:L17, H19:L24, H26:L31) corresponds to "Latency Budget Preserved", "Latency Budget Consumed", and "Covered Request Capacity" from the labels on the Task sheet.
   - For each cell, write the formula referencing the corresponding cells from those blocks.

d) **Step 2 – Statistics (H42:L47)**
   For each column H through L:
   - MIN of H35:H40 (or the appropriate column)
   - MAX
   - MEDIAN
   - AVERAGE
   - Use `PERCENTILE` (NOT `PERCENTILE.INC` or `PERCENTILE.EXC`) for 25th percentile with k=0.25
   - Use `PERCENTILE` for 75th percentile with k=0.75
   Map these to the correct rows based on the labels found in step 1.

e) **Step 3 – Weighted mean (H50:L50)**
   For each column, write: `=SUMPRODUCT(<Net_SLA_buffer_column>,<Covered_Request_Capacity_column>)/SUM(<Covered_Request_Capacity_column>)`
   where the Net SLA buffer values are from H35:H40 (etc.) and Covered Request Capacity from H26:H31 (etc.).

f) Save to `/root/output/result.xlsx`. Create `/root/output/` if it doesn't exist.

## 3 – Validate
Re-open `/root/output/result.xlsx` (not data_only) and print all formula cells in the ranges H12:L17, H19:L24, H26:L31, H35:L40, H42:L47, H50:L50 to confirm they are non-empty and contain formulas (strings starting with '='). Also open with data_only=True and print those cells to check for #NAME?, #REF!, #VALUE!, or None errors.

## Important constraints
- Do NOT add sheets, macros, VBA, external links, or helper tabs.
- Do NOT change existing formatting.
- Use `PERCENTILE` not `PERCENTILE.INC` or `PERCENTILE.EXC`.
- Use `AVERAGE` not `MEAN`.
- All formulas must be Excel-compatible strings written via openpyxl.
- Preserve all existing content outside the target cells.

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