# Task Instruction

## Task: Populate formulas in /root/data/workbook.xlsx and save to /root/output/result.xlsx

### Pre-work: Inspect the workbook and test infrastructure
1. `mkdir -p /root/output`
2. Open and inspect `/root/data/workbook.xlsx` using openpyxl (data_only=False) to understand:
   - Sheet `Task`: read row 10 (year headers in H10:L10), column D for rows 12-17, 19-24, 26-31 (series codes), the yellow cell ranges, row 35-40 labels, row 42-47 labels, and row 50 label.
   - Sheet `Data`: read rows 21-38 to understand the data layout (column headers, row labels, structure).
   - Note exact cell contents, merged cells, existing formulas, and formatting.
3. Read `/tests/test_outputs.py` (and any other test files) to understand exactly what the verifier checks — expected values, tolerances, cell ranges, how it evaluates formulas (does it use openpyxl data_only=True, or xlcalc, or subprocess Excel?).
4. Print all findings before writing any formulas.

### Step 1: Lookup formulas in H12:L17, H19:L24, H26:L31
For each cell in these ranges, write a formula that looks up a value from `Data!$21:$38` using:
- The series code from column D of the current row on `Task` sheet
- The year from row 10 of the `Task` sheet

Use `INDEX(MATCH, MATCH)` pattern — this is the safest and most universally recognized:
```
=INDEX(Data!<data_range>, MATCH(<series_code_cell>, Data!<series_code_column>, 0), MATCH(<year_cell>, Data!<year_header_row>, 0))
```
Adjust the exact ranges after inspecting the Data sheet layout. Make sure:
- Row references cover rows 21:38 on Data sheet
- The MATCH for series codes searches the correct column in Data (likely column A or B)
- The MATCH for years searches the correct header row in Data
- Use absolute references where appropriate ($) to allow the formula pattern to work across the range

### Step 2: Net production slack in H35:L40 and statistics in H42:L47
For H35:L40, calculate:
```
=(H12 - H19) / H26 * 100
```
(Adjust row references: row 12-17 = Finished Output, row 19-24 = Scrap And Rework, row 26-31 = Rated Production Capacity. Map each plant row in 35-40 to corresponding rows in 12-17, 19-24, 26-31.)

For H42:L47 (column-wise statistics over H35:L40):
- H42: `=MIN(H35:H40)` (or `=MIN(H$35:H$40)`)
- H43: `=MAX(H35:H40)`
- H44: `=MEDIAN(H35:H40)`
- H45: `=AVERAGE(H35:H40)`
- H46: `=PERCENTILE(H35:H40, 0.25)` — **IMPORTANT: use `PERCENTILE` not `PERCENTILE.INC` or `PERCENTILE.EXC`**. Check the test file to see which function name the evaluator recognizes. The cross-task artifacts show #NAME? errors in rows 46-47, likely from using `.INC`/`.EXC` suffixed function names that openpyxl/xlcalc don't recognize.
- H47: `=PERCENTILE(H35:H40, 0.75)`

**Critical**: Verify by reading the test expectations which row is which statistic (min/max/median/mean/25th/75th). The order in rows 42-47 must match what the verifier expects. Check the row labels in column D or the test file.

### Step 3: Weighted mean in H50:L50
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```
This computes the weighted mean of the Net production slack percentages (H35:H40) weighted by Rated Production Capacity (H26:H31).

### Save and Validate
1. Save the workbook to `/root/output/result.xlsx` using openpyxl, preserving all existing formatting.
2. Re-open the saved file and print all formula cells to verify they look correct.
3. Run the test suite: `cd /root && python -m pytest tests/ -v` (or however the tests are structured).
4. If tests fail, read the error output carefully, diagnose, fix, and re-run.
5. If you see #NAME? errors, check that you used base function names (PERCENTILE, not PERCENTILE.INC; AVERAGE not AVERAGEIF, etc.).
6. If the SUMPRODUCT weighted mean formula doesn't match expectations, check whether the verifier expects `SUMPRODUCT(values, weights)/SUM(weights)` or a different formulation.

### Key Warnings from Cross-Task Evidence
- Do NOT use `PERCENTILE.INC` or `PERCENTILE.EXC` — use `PERCENTILE`.
- Do NOT use any `.INC`/`.EXC` suffixed statistical functions.
- Verify the exact row-to-statistic mapping before writing formulas.
- Inspect the Data sheet thoroughly — the exact range boundaries matter for INDEX/MATCH.

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