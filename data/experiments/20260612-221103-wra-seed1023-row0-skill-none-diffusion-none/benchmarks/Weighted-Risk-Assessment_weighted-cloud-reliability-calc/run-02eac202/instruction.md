# Task Instruction

Execute the following steps in order. Each step is a separate code block.

## Step 0 – Inspect the workbook

```python
import openpyxl, os, shutil

wb = openpyxl.load_workbook('/root/data/workbook.xlsx')

# ---- Task sheet layout ----
ws = wb['Task']
print('=== Task sheet ===')
for r in range(1, 55):
    vals = []
    for c in range(1, 15):  # A-N
        v = ws.cell(r, c).value
        if v is not None:
            vals.append(f"{ws.cell(r, c).coordinate}={v!r}")
    if vals:
        print(f"Row {r}: {', '.join(vals)}")

# ---- Data sheet layout ----
ws2 = wb['Data']
print('\n=== Data sheet (rows 1-5 and 18-42) ===')
for r in list(range(1, 6)) + list(range(18, 43)):
    vals = []
    for c in range(1, 30):
        v = ws2.cell(r, c).value
        if v is not None:
            vals.append(f"{ws2.cell(r, c).coordinate}={v!r}")
    if vals:
        print(f"Row {r}: {', '.join(vals)}")

wb.close()
```

Read the output carefully. Identify:
- Row 10 on Task: which columns hold years (H–L).
- Column D on Task: which rows hold series codes for each block (H12:L17, H19:L24, H26:L31).
- Data sheet rows 21:38: the layout of the lookup table (which row holds headers/series codes, which columns hold years, where the data values are).
- Rows 35-40 on Task: labels for the six regions used in Net reliability gap.
- Row 42-47 labels: min, max, median, mean, 25th percentile, 75th percentile.
- Row 50: label for Global Cloud Mesh (GCM) weighted mean.

## Step 1 – Write lookup formulas into H12:L17, H19:L24, H26:L31

Based on the inspection output, write a Python script using openpyxl that:
1. Opens `/root/data/workbook.xlsx`.
2. For each yellow cell in H12:L17, H19:L24, H26:L31, writes an INDEX/MATCH formula that:
   - Uses the series code from column D of the same row on the Task sheet.
   - Uses the year from row 10 of the same column on the Task sheet.
   - Looks up the value in the Data sheet rows 21:38.
   - Uses the exact column and row ranges you found in Step 0.
   - Example pattern: `=INDEX(Data!$<first_data_col>$21:$<last_data_col>$38, MATCH(D12, Data!$<series_col>$21:$<series_col>$38, 0), MATCH(H10, Data!$<first_data_col>$20:$<last_data_col>$20, 0))`
   - Adjust the column letters, row numbers, and header row based on what you actually found in Step 0.
3. Does NOT save yet.

## Step 2 – Net reliability gap (H35:L40) and summary statistics (H42:L47)

Continuing in the same script (or a new block that re-opens the workbook):
1. For each cell in H35:L40, write a formula:
   `=(H12 - H19) / H26 * 100`
   adjusting row references so that:
   - "Successful API Requests" comes from the first block (rows 12-17)
   - "Failed API Requests" comes from the second block (rows 19-24)
   - "Compute Capacity" comes from the third block (rows 26-31)
   - The six regions in rows 35-40 correspond to the six regions in rows 12-17 / 19-24 / 26-31 in the same order.
2. For H42:L47, write column-wise formulas:
   - Row 42 (MIN): `=MIN(H35:H40)`
   - Row 43 (MAX): `=MAX(H35:H40)`
   - Row 44 (MEDIAN): `=MEDIAN(H35:H40)`
   - Row 45 (AVERAGE): `=AVERAGE(H35:H40)`
   - Row 46 (25th percentile): `=PERCENTILE(H35:H40, 0.25)`
   - Row 47 (75th percentile): `=PERCENTILE(H35:H40, 0.75)`
   Adjust the exact row labels (min/max/median/mean/25th/75th) to match what you see in Step 0 output.

## Step 3 – Weighted mean for GCM (H50:L50)

For each column H through L in row 50:
`=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)`

This uses the Net reliability gap percentages as values and Compute Capacity as weights.

## Step 4 – Save

```python
os.makedirs('/root/output', exist_ok=True)
wb.save('/root/output/result.xlsx')
wb.close()
```

## Step 5 – Verify

Re-open `/root/output/result.xlsx` and print the formula (not value) in every target cell to confirm they are non-None strings starting with '=':

```python
wb2 = openpyxl.load_workbook('/root/output/result.xlsx')
ws = wb2['Task']
for row_range in [(12,17), (19,24), (26,31), (35,40), (42,47), (50,50)]:
    for r in range(row_range[0], row_range[1]+1):
        for c_letter in ['H','I','J','K','L']:
            cell = ws[f'{c_letter}{r}']
            print(f'{c_letter}{r} = {cell.value!r}')
wb2.close()
```

Confirm every printed value is a formula string (starts with '='). If any is None, diagnose and fix before finishing.

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