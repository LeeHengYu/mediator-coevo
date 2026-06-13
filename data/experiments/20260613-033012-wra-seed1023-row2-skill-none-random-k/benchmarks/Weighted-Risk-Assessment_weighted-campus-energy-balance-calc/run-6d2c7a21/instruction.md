# Task Instruction

Execute the following steps exactly, in order.

## 0 – Inspect the source workbook
```bash
cp /root/data/workbook.xlsx /root/output/result.xlsx
```
Open `/root/output/result.xlsx` with openpyxl (data_only=False). Print:
- Sheet names.
- `Task` sheet: contents of D12:D17, D19:D24, D26:D31 (series codes), row 10 columns H–L (years), H35:L40 labels/current content, H42:L47 labels, H50:L50 label.
- `Data` sheet: rows 21–38, all populated columns (at least A–Z). Identify the layout: which column holds the series code, which row holds years, and where numeric data starts.
- Check what is already in the yellow target cells (H12:L17 etc.) – empty or pre-filled?

This inspection is critical; do NOT skip it.

## 1 – Write lookup formulas in H12:L17, H19:L24, H26:L31

Using openpyxl, write formulas into every cell in these three blocks.

Each formula must combine INDEX + MATCH (safest cross-engine compatibility):
```
=INDEX(Data!<data_range>, MATCH($D12, Data!<series_code_column>, 0), MATCH(H$10, Data!<year_row>, 0))
```
Adjust `<data_range>`, `<series_code_column>`, and `<year_row>` based on what you found in step 0. Use absolute references for the Data ranges and mixed references ($D for column, $10 for row) so the formula copies correctly across the 5 columns × 6 rows in each block.

Make sure:
- The series-code column reference is a single column (e.g., `Data!$A$21:$A$38`).
- The year row reference is a single row (e.g., `Data!$B$20:$Z$20` — adjust to actual).
- The data range spans all series rows and year columns on Data sheet.

## 2 – Net renewable balance formulas in H35:L40

For each campus row i (i = 0..5) and each year column j (H..L):
```
= (H12_cell - H19_cell) / H26_cell * 100
```
where H12_cell is the corresponding Renewable Generation cell (rows 12–17), H19_cell is Grid Consumption (rows 19–24), H26_cell is Baseline Energy Demand (rows 26–31).

Write actual cell-reference formulas, e.g. for H35:
```
=(H12-H19)/H26*100
```

## 3 – Summary statistics in H42:L47

For each year column (H through L), write these formulas:
- Row 42 (MIN):    `=MIN(H35:H40)`
- Row 43 (MAX):    `=MAX(H35:H40)`
- Row 44 (MEDIAN): `=MEDIAN(H35:H40)`
- Row 45 (MEAN):   `=AVERAGE(H35:H40)`
- Row 46 (25th percentile): `=PERCENTILE(H35:H40,0.25)`
- Row 47 (75th percentile): `=PERCENTILE(H35:H40,0.75)`

**IMPORTANT**: Use exactly `PERCENTILE` (not `PERCENTILE.INC`, not `_xlfn.PERCENTILE.INC`). Verify by reading the cell value back from openpyxl that the string is exactly as intended — no stray characters, no `_xlfn.` prefix.

If the verifier still rejects `PERCENTILE`, try `_xlfn.PERCENTILE.INC` as a fallback — but try plain `PERCENTILE` first.

## 4 – Weighted mean in H50:L50

For each year column (H through L):
```
=SUMPRODUCT(H35:H40,H26:H31)/SUM(H26:H31)
```
This computes the weighted mean of the net-renewable-balance percentages weighted by Baseline Energy Demand.

## 5 – Save and verify

Save the workbook to `/root/output/result.xlsx`.

Then reopen it (data_only=False) and print:
- A sample formula from each block (H12, H19, H26, H35, H42–H47, H50).
- Confirm no cells contain `#NAME?` or other error literals in the formula text.
- Confirm the yellow-cell ranges are all populated (not None).

Finally, if a test script exists at `/root/tests/test_output.py` or similar, run it:
```bash
cd /root && python -m pytest tests/ -x -v 2>&1 | head -80
```
Report the result. If PERCENTILE causes #NAME?, re-edit those cells to use `_xlfn.PERCENTILE.INC` instead, re-save, and re-run the test.

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