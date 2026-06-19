# Task Instruction

Execute the following steps carefully to produce `/root/output/result.xlsx`.

## Phase 0 – Inspect the workbook

```python
import openpyxl, os, shutil

wb = openpyxl.load_workbook('/root/data/workbook.xlsx')
for name in wb.sheetnames:
    ws = wb[name]
    print(f'=== Sheet: {name} ===')
    for row in ws.iter_rows(min_row=1, max_row=55, max_col=13, values_only=False):
        vals = [(c.coordinate, c.value) for c in row if c.value is not None]
        if vals:
            print(vals)
```

Run this first. Read the output carefully to understand:
- The series codes in column D for rows 12-17, 19-24, 26-31 on sheet `Task`.
- The years in row 10 (columns H-L) on sheet `Task`.
- The layout of sheet `Data` rows 21-38 (which column holds the series code, which row holds the year headers, and where the data values are).
- The labels in rows 35-40 (Net SLA buffer services), rows 42-47 (statistics labels: min, max, median, mean, 25th pctl, 75th pctl), and row 50.
- The exact row/column structure of Data sheet so INDEX/MATCH references are correct.

## Phase 1 – Write formulas with openpyxl

After inspecting, write a Python script that:

1. Opens `/root/data/workbook.xlsx` with `openpyxl.load_workbook`.
2. Gets the `Task` sheet: `ws = wb['Task']`.
3. For each yellow lookup block (H12:L17, H19:L24, H26:L31), writes an `INDEX(MATCH,MATCH)` formula into each cell. The formula should:
   - Use the series code from column D of the same row (e.g., `$D12`).
   - Use the year from row 10 of the same column (e.g., `H$10`).
   - Reference the Data sheet's data range and its row/column headers appropriately.
   - Use exact match (0) for both MATCH functions.
   - Example pattern: `=INDEX(Data!$B$22:$S$38,MATCH($D12,Data!$A$22:$A$38,0),MATCH(H$10,Data!$B$21:$S$21,0))` — but adjust the exact ranges based on what you see in Phase 0.

4. For H35:L40 (Net SLA buffer), write a formula that computes:
   `(Latency Budget Preserved - Latency Budget Consumed) / Covered Request Capacity * 100`
   This means for each cell, reference the corresponding cells from the three lookup blocks. For example, if row 35 corresponds to the first service:
   `=(H12-H19)/H26*100`
   Adjust row references based on the actual layout discovered in Phase 0.

5. For H42:L47 (statistics), write column-wise formulas over H35:L40 (6 cells):
   - Row 42: `=MIN(H35:H40)` (adjust column)
   - Row 43: `=MAX(H35:H40)`
   - Row 44: `=MEDIAN(H35:H40)`
   - Row 45: `=AVERAGE(H35:H40)`
   - Row 46: **`=PERCENTILE(H35:H40,0.25)`** — use `PERCENTILE` not `PERCENTILE.INC` or `PERCENTILE.EXC`. This is critical based on prior failures with `#NAME?` errors.
   - Row 47: `=PERCENTILE(H35:H40,0.75)`

   **IMPORTANT**: Before writing, test which function name works. Try writing `PERCENTILE` first. Do NOT use `PERCENTILE.INC` or `PERCENTILE.EXC` as these caused `#NAME?` errors in previous runs. The plain `PERCENTILE` function is universally recognized.

6. For H50:L50 (weighted mean), write a SUMPRODUCT formula:
   `=SUMPRODUCT(H35:H40,H26:H31)/SUM(H26:H31)`
   Adjust column letters for each of H through L.

7. Save to `/root/output/result.xlsx`:
   ```python
   os.makedirs('/root/output', exist_ok=True)
   wb.save('/root/output/result.xlsx')
   ```

## Phase 2 – Verify

After saving, re-open `/root/output/result.xlsx` and print the values/formulas in all modified cells to confirm:
- Every cell in H12:L17, H19:L24, H26:L31 contains a formula string starting with `=`.
- Every cell in H35:L40 contains a formula.
- Every cell in H42:L47 contains a formula (especially check rows 46-47 use `PERCENTILE`).
- H50:L50 contains SUMPRODUCT formulas.
- No cells are None or empty.

Also check that no extra sheets were added and the sheet names are unchanged.

## Key Warnings
- Use `PERCENTILE(range,0.25)` and `PERCENTILE(range,0.75)` — NOT `PERCENTILE.INC` or `PERCENTILE.EXC`.
- All formulas must start with `=`.
- Do not modify any existing formatting, values, or structure outside the specified cells.
- Do not add sheets, macros, VBA, or external links.
- Make sure to write to the `Task` sheet, not `Data` or any other sheet.
- Ensure the INDEX/MATCH references on the Data sheet are correct by carefully examining the Phase 0 output.

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