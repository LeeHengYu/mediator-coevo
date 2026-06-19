# Task Instruction

Execute the following steps exactly, in order.

## 0 — Inspect the workbook
```bash
cp /root/data/workbook.xlsx /root/data/workbook_backup.xlsx
```
Open `/root/data/workbook.xlsx` with openpyxl (data_only=False) and print:
- Sheet names
- `Task` sheet: contents of columns D–G for rows 10–50 (to see series codes, labels, row structure)
- `Task` sheet: contents of H10:L10 (year headers)
- `Data` sheet: row 20 headers, then rows 21–38 (all columns) to understand the lookup source
- `Task` sheet: cells H12:L17, H19:L24, H26:L31 (confirm they are empty/yellow)
- `Task` sheet: rows 35–50, columns D–L (to see labels for Net SLA buffer, statistics, and weighted mean rows)

Print everything clearly before making any edits.

## 1 — Write the formulas with openpyxl

Use openpyxl to write Excel formulas (as strings starting with `=`) into the cells. Do NOT use data_only mode for writing. Preserve all existing formatting by loading with `openpyxl.load_workbook(keep_vba=False)` and not touching any cells outside the target ranges.

### Step 1 — Lookup formulas in H12:L17, H19:L24, H26:L31

For each cell in these three 6×5 blocks:
- Row gives a series code in column D of that row (e.g., D12, D13, …)
- Column gives a year in row 10 (e.g., H10, I10, …)
- Data lives on sheet `Data` in rows 21:38

Use `INDEX(MATCH,MATCH)` pattern. Determine the exact data range on `Data` sheet from your inspection. The formula pattern for cell H12 should look like:
```
=INDEX(Data!$B$21:$XX$38,MATCH($D12,Data!$A$21:$A$38,0),MATCH(H$10,Data!$B$20:$XX$20,0))
```
Adjust the column references based on what you find in the Data sheet. The key is:
- Match the series code (column D of current row) against the first column of the Data range (rows 21:38)
- Match the year (row 10) against the header row of the Data range (row 20)
- INDEX into the data body

Apply the same formula pattern (with appropriate $ anchoring for copy) to all 90 cells across the three blocks.

### Step 2 — Net SLA buffer in H35:L40

For each of the 6 services (rows 35–40) and 5 year columns (H–L):
```
=(H12 - H19) / H26 * 100
```
where row 12 maps to row 35, row 13 to row 36, etc. (i.e., row offset is +23 from the first lookup block to the net buffer block). Verify the exact row mapping from your inspection — the first block (H12:L17) is "Latency Budget Preserved", the second (H19:L24) is "Latency Budget Consumed", and the third (H26:L31) is "Covered Request Capacity". Adjust if the labels say otherwise.

### Step 2 continued — Statistics in H42:L47

For each column (H through L):
- Row 42 (MIN): `=MIN(H35:H40)`
- Row 43 (MAX): `=MAX(H35:H40)`
- Row 44 (MEDIAN): `=MEDIAN(H35:H40)`
- Row 45 (AVERAGE/MEAN): `=AVERAGE(H35:H40)`
- Row 46 (25th percentile): `=PERCENTILE(H$35:H$40,0.25)`
- Row 47 (75th percentile): `=PERCENTILE(H$35:H$40,0.75)`

**CRITICAL**: Use `PERCENTILE` (legacy function), NOT `PERCENTILE.INC` or `PERCENTILE.EXC`. The previous run failed with #NAME? errors because the environment doesn't support dotted function names. Similarly use `AVERAGE` not `MEAN`.

Check the labels in column D/G for rows 42–47 to confirm which statistic goes in which row. Map accordingly — do not assume the order above is correct without verifying from the sheet inspection.

### Step 3 — Weighted mean in H50:L50

For each column (H through L):
```
=SUMPRODUCT(H35:H40,H26:H31)/SUM(H26:H31)
```
This computes the weighted mean of Net SLA buffer percentages weighted by Covered Request Capacity.

## 2 — Save

Save the workbook to `/root/output/result.xlsx`. Create the output directory if needed:
```python
import os
os.makedirs('/root/output', exist_ok=True)
wb.save('/root/output/result.xlsx')
```

## 3 — Validate

After saving, reload the file and print all formula strings in the target cells to confirm:
- H12:L17, H19:L24, H26:L31 contain INDEX/MATCH formulas
- H35:L40 contain the net buffer formula
- H42:L47 contain MIN, MAX, MEDIAN, AVERAGE, PERCENTILE (no dots!) formulas
- H50:L50 contain SUMPRODUCT formulas
- No cells are empty/None
- No use of PERCENTILE.INC, PERCENTILE.EXC, or MEAN anywhere

Also verify that no other cells were modified by checking a few known non-target cells still have their original values.

If the verifier test file exists at `/root/test_output.py` or similar, run it:
```bash
cd /root && python -m pytest test_output.py -v 2>&1 | head -80
```

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