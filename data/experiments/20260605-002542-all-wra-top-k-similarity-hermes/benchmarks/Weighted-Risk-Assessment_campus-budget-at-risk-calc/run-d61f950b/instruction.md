# Task Instruction

Execute the following steps to complete the task.

## 0. Inspect the workbook and test expectations

```bash
cd /root
find . -name '*.py' -path '*/tests/*' | head -20
cat tests/test_outputs.py 2>/dev/null || true
```

Open the workbook with openpyxl (data_only=False) and inspect:
- Sheet `Task`: read the layout of rows 10-50, especially columns D and H-L. Note the series codes in column D for rows 12-17, 19-24, 26-31, 35-40. Note the year headers in H10:L10. Note any existing formulas or labels in rows 42-47 and row 50.
- Sheet `Data`: read rows 21-38 to understand the data layout (which row holds headers, which column holds series codes, where the year columns start, etc.).

Print all of this so you understand the exact structure before writing any formulas.

## 1. Populate lookup formulas in H12:L17, H19:L24, H26:L31

For each cell in these ranges, write a formula that looks up the value from `Data!$21:$38` using:
- The series code from column D of the current row on `Task`
- The year from row 10 of the current column on `Task`

Use `INDEX(MATCH, MATCH)` pattern, e.g.:
```
=INDEX(Data!$A$21:$Z$38, MATCH($D12, Data!$A$21:$A$38, 0), MATCH(H$10, Data!$A$21:$Z$21, 0))
```
Adjust the exact range boundaries based on what you see in the Data sheet (the column and row extents of the data block). Make sure:
- The row lookup array is the series-code column of Data rows 21:38
- The column lookup array is the header row of Data rows 21:38
- Use appropriate absolute references ($) so the formula can be applied across the block

## 2. Populate Net budget buffer in H35:L40

The formula for each cell is:
```
=(CommittedFunding - OperatingSpend) / ApprovedBudgetBase * 100
```
where:
- CommittedFunding = corresponding cell in H12:L17 block (rows 12-17)
- OperatingSpend = corresponding cell in H19:L24 block (rows 19-24)  
- ApprovedBudgetBase = corresponding cell in H26:L31 block (rows 26-31)

So for H35: `=(H12-H19)/H26*100`, for H36: `=(H13-H20)/H27*100`, etc.

Verify by checking the row-to-row mapping: row 35↔(12,19,26), row 36↔(13,20,27), ..., row 40↔(17,24,31).

## 3. Populate statistics in H42:L47

These are column-wise statistics over H35:L40 (the 6 Net budget buffer values per column):
- Row 42: `=MIN(H35:H40)` (minimum)
- Row 43: `=MAX(H35:H40)` (maximum)
- Row 44: `=MEDIAN(H35:H40)` (median)
- Row 45: `=AVERAGE(H35:H40)` (mean)
- Row 46: `=PERCENTILE(H35:H40,0.25)` (25th percentile)
- Row 47: `=PERCENTILE(H35:H40,0.75)` (75th percentile)

**CRITICAL**: Check the labels in column D/E/F/G for rows 42-47 to determine the correct order (min, max, median, mean, 25th pct, 75th pct). Map them accordingly.

**CRITICAL**: Use `PERCENTILE` (not `PERCENTILE.INC` or `PERCENTILE.EXC`). The cross-task failure signatures show #NAME? errors from using function names that openpyxl/Excel engines don't recognize. `PERCENTILE` is the safe, universally recognized function name. Similarly use `MEDIAN`, `MIN`, `MAX`, `AVERAGE` — these are all safe.

## 4. Populate weighted mean in H50:L50

For each column, use SUMPRODUCT with the Net budget buffer percentages (H35:H40) as values and the Approved Budget Base (H26:H31) as weights:
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```

## 5. Save the workbook

```python
import shutil, os
os.makedirs('/root/output', exist_ok=True)
shutil.copy('/root/data/workbook.xlsx', '/root/output/result.xlsx')
# Then open /root/output/result.xlsx with openpyxl, write all formulas, save
```

Make sure to preserve existing formatting: open with openpyxl without data_only, write only to the specified cells, and save.

## 6. Validate

Run the test suite:
```bash
cd /root && python -m pytest tests/ -v 2>&1 | tail -60
```

If any cells show #NAME? errors or wrong values, inspect and fix. Pay special attention to rows 46-47 (percentile functions).

If tests reference specific expected values, compare your formula outputs (you can open with data_only=True after saving and re-opening, or check the test expectations directly).

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