# Task Instruction

Execute the following steps to produce /root/output/result.xlsx.

## 0 — Preparation
```bash
mkdir -p /root/output
pip install openpyxl
```

## 1 — Inspect the workbook
Open `/root/data/workbook.xlsx` with openpyxl (data_only=False) and inspect:
- Sheet `Task`: print cells D12:D17, D19:D24, D26:D31 (series codes), row 10 columns H–L (years), cells H35:H40 labels/formulas if any, H42:H47 labels, H50 label. Also print the current content of H12, H19, H26 to see if they already have formulas or are blank.
- Sheet `Data`: print rows 21–38, focusing on the header row (row 21 presumably) and a few data rows to understand the layout — which column holds the series code, which columns/rows hold years, and where the numeric data lives.
- Print the exact column letters and row numbers so the VLOOKUP/INDEX/MATCH references are correct.

## 2 — Write the Python script
After inspecting, write a Python script `/root/solve.py` that:

### 2a — Open workbook
```python
import openpyxl
wb = openpyxl.load_workbook('/root/data/workbook.xlsx')
ws = wb['Task']
```

### 2b — Step 1: Populate H12:L17, H19:L24, H26:L31 with lookup formulas
For each cell in these ranges, build an INDEX/MATCH/MATCH formula that:
- Looks up the series code from column D of the same row
- Looks up the year from row 10 of the same column
- Searches in the Data sheet rows 21:38

Use this pattern (adjust exact Data-sheet references after inspection):
```
=INDEX(Data!$B$22:$<lastcol>$38, MATCH($D<row>,Data!$A$22:$A$38,0), MATCH(<colref>$10,Data!$B$21:$<lastcol>$21,0))
```
Adjust the exact column letters and row numbers based on what you find in the inspection step. The key is that the series code column in Data and the year header row in Data must be identified precisely.

### 2c — Step 2: Net container flow in H35:L40
For each cell (row r, col c) in H35:L40, write a formula:
```
=(<loaded_inbound_cell> - <loaded_outbound_cell>) / <throughput_capacity_cell> * 100
```
where:
- loaded_inbound_cell = corresponding cell in H12:L17 block (rows 12–17)
- loaded_outbound_cell = corresponding cell in H19:L24 block (rows 19–24)
- throughput_capacity_cell = corresponding cell in H26:L31 block (rows 26–31)

So for H35: `=(H12-H19)/H26*100`, for I35: `=(I12-I19)/I26*100`, etc.
For H36: `=(H13-H20)/H27*100`, etc.

### 2d — Step 2: Statistics in H42:L47
For each column c in H–L:
- Row 42 (Min): `=MIN(<c>35:<c>40)`
- Row 43 (Max): `=MAX(<c>35:<c>40)`
- Row 44 (Median): `=MEDIAN(<c>35:<c>40)` — but write as `=_xlfn.MEDIAN(<c>35:<c>40)` if needed. First try without prefix; the feedback only mentions PERCENTILE having issues.
- Row 45 (Mean): `=AVERAGE(<c>35:<c>40)`
- Row 46 (25th percentile): `=_xlfn.PERCENTILE.INC(<c>35:<c>40,0.25)`
- Row 47 (75th percentile): `=_xlfn.PERCENTILE.INC(<c>35:<c>40,0.75)`

**CRITICAL**: For rows 46 and 47, you MUST use the `_xlfn.PERCENTILE.INC` prefix. This was the cause of the previous failure (#NAME? error). Do NOT use bare `PERCENTILE` or `PERCENTILE.INC` — always prefix with `_xlfn.`.

Also check whether MEDIAN needs the prefix. To be safe, use `_xlfn.MEDIAN` as well since it could also be a future function.

### 2e — Step 3: Weighted mean in H50:L50
For each column c in H–L:
```
=SUMPRODUCT(<c>35:<c>40, <c>26:<c>31) / SUM(<c>26:<c>31)
```
This computes the weighted mean of net container flow percentages weighted by terminal throughput capacity.

### 2f — Save
```python
wb.save('/root/output/result.xlsx')
```

## 3 — Run and verify
```bash
python /root/solve.py
```
Then re-open the saved file with openpyxl (data_only=False) and print:
- A sample of cells from each block (H12, L17, H19, L24, H26, L31) to confirm formulas are present
- H35, L40 to confirm net flow formulas
- H42, H46, H47 to confirm stats formulas (especially check the _xlfn prefix is in the percentile formulas)
- H50 to confirm SUMPRODUCT formula
- Verify no cells show None (which would mean the formula wasn't written)

## 4 — Run the verifier if available
```bash
cd /root && python -m pytest test_output.py -v 2>&1 | head -80
```
If tests fail, read the error messages carefully and fix.

## Key Reminders
- Do NOT add new sheets, macros, VBA, or external links.
- Do NOT change existing formatting.
- The `_xlfn.` prefix is essential for PERCENTILE.INC (and possibly MEDIAN) to avoid #NAME? errors.
- Inspect the Data sheet layout carefully before writing formulas — get the exact cell references right.
- The inspection step is critical: do it first, then write the script based on actual cell contents.

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