# Task Instruction

Execute the following steps to complete the hospital-capacity-at-risk workbook task.

## Phase 0 – Inspect the workbook
1. `pip install openpyxl` if needed.
2. Open `/root/data/workbook.xlsx` with openpyxl (data_only=False).
3. Print the sheet names.
4. For the `Task` sheet, print:
   - Row 10 (headers / years) columns A–L.
   - Rows 11–50, columns A–L (values). Focus on column D (series codes) and column G–L.
5. For the `Data` sheet, print:
   - Row 1 (or whichever row has headers).
   - Rows 21–38 fully (all columns with data).
6. Record:
   - The exact series codes in column D for rows 12–17, 19–24, 26–31.
   - The exact years in H10:L10.
   - The layout of the Data sheet rows 21–38: which column holds the series code, which columns hold year-indexed values, and whether data is arranged by rows or columns.
   - The labels in rows 35–40 (cluster names), rows 42–47 (statistic names), and row 50.

## Phase 1 – Write lookup formulas in H12:L17, H19:L24, H26:L31
Using openpyxl, write INDEX/MATCH formulas into each cell. The pattern for cell (r, c) should be:

```
=INDEX(Data!<value_range>, MATCH(D{r}, Data!<series_code_column>, 0), MATCH(<Task>!{col}10, Data!<year_header_range>, 0))
```

Make sure:
- The Data sheet range references match what you found in Phase 0. Rows 21:38 of the Data sheet should be the lookup area.
- The series code column in Data and the year header row in Data are correctly identified.
- All formulas start with `=`.
- Use absolute references where appropriate so formulas are robust.

Write formulas for all 3 blocks (H12:L17, H19:L24, H26:L31) — that's 6 rows × 5 columns × 3 blocks = 90 cells.

## Phase 2 – Net capacity headroom formulas in H35:L40
For each of the 6 hospital clusters (rows 35–40) and each year column (H–L), write:

```
=(H12 - H19) / H26 * 100
```

where H12 is the Available Care Slots cell, H19 is the Occupied Care Slots cell, and H26 is the Staffed Bed Capacity cell for the same cluster and year. Adjust row references per cluster:
- Row 35 uses rows 12, 19, 26
- Row 36 uses rows 13, 20, 27
- Row 37 uses rows 14, 21, 28
- Row 38 uses rows 15, 22, 29
- Row 39 uses rows 16, 23, 30
- Row 40 uses rows 17, 24, 31

## Phase 3 – Summary statistics in H42:L47
For each year column (H–L), write formulas in rows 42–47. Check the labels in column D/E for rows 42–47 to determine which statistic goes where. Expected statistics: MIN, MAX, MEDIAN, AVERAGE, 25th percentile, 75th percentile.

Use these Excel functions:
- MIN: `=MIN(H35:H40)`
- MAX: `=MAX(H35:H40)`
- MEDIAN: `=MEDIAN(H35:H40)`
- AVERAGE: `=AVERAGE(H35:H40)`
- 25th percentile: `=PERCENTILE(H35:H40, 0.25)`
- 75th percentile: `=PERCENTILE(H35:H40, 0.75)`

**Important**: Use `PERCENTILE` (not `PERCENTILE.INC` or `PERCENTILE.EXC`) as the cross-task feedback indicates `PERCENTILE.INC` caused #NAME? errors.

Match each formula to the correct row based on the label you see in the workbook.

## Phase 4 – Weighted mean in H50:L50
For each year column (H–L):
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```
This computes the weighted mean of Net capacity headroom using Staffed Bed Capacity as weights.

## Phase 5 – Save and verify
1. `mkdir -p /root/output`
2. Save the workbook to `/root/output/result.xlsx`.
3. Re-open the saved file and print cells H12, H19, H26, H35, H42, H50 (column H) to confirm they contain formula strings (starting with `=`), not None.
4. Confirm no new sheets were added.

## Critical checks
- Every formula cell must have a string value starting with '='.
- Do NOT use data_only=True when writing.
- Work only on the 'Task' sheet. Do not modify 'Data'.
- Do not add sheets, macros, VBA, external links, or helper tabs.
- Preserve all existing formatting.

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