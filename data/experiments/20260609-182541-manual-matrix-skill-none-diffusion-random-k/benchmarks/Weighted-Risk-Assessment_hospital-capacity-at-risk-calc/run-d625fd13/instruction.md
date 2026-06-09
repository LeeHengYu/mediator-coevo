# Task Instruction

Execute the following steps precisely to complete the task.

## 0. Setup
```bash
mkdir -p /root/output
cp /root/data/workbook.xlsx /root/output/result.xlsx
```

## 1. Inspect the workbook structure
Open `/root/output/result.xlsx` with openpyxl and inspect:
- Sheet names (should include `Task` and `Data`).
- On sheet `Task`: read row 10 to find the year headers in columns H–L. Read column D rows 12–17, 19–24, 26–31 to find the series codes. Read any labels in rows 35–40 (column D or nearby) and rows 42–47 (stat labels). Read row 50 label.
- On sheet `Data`: read rows 21–38 to understand the data layout — identify which row holds headers, which column holds series codes, and how years are arranged (row-wise or column-wise). Print out enough to understand the orientation.

Print all of this information before proceeding.

## 2. Populate H12:L17, H19:L24, H26:L31 with lookup formulas

For each cell in these three blocks, write a formula that:
- Takes the series code from column D of the same row on sheet `Task`
- Takes the year from row 10 of the same column on sheet `Task`
- Looks up the value on sheet `Data` rows 21:38

Use INDEX/MATCH pattern. The exact formula depends on the Data sheet layout discovered in step 1. Two likely patterns:

**If Data has series codes in a column and years across a row:**
```
=INDEX(Data!<data_range>, MATCH($D12, Data!<series_code_column>, 0), MATCH(H$10, Data!<year_row>, 0))
```

**If Data has years in a column and series codes across a row (transposed):**
Adjust accordingly.

IMPORTANT: Use absolute references for the lookup arrays ($D12 with $ on column, H$10 with $ on row) so formulas can be written consistently across the block. The Data range references should also be absolute.

Write formulas using openpyxl by setting each cell's `.value` to the formula string. Do NOT use `data_only=True` when loading. Make sure to preserve existing formatting by NOT changing fonts, fills, borders, number formats, etc.

## 3. Populate H35:L40 — Net capacity headroom

For each of the 6 hospital clusters (rows 35–40) and each year column (H–L), write a formula:
```
=(H12 - H19) / H26 * 100
```
where row 12 corresponds to Available Care Slots, row 19 to Occupied Care Slots, and row 26 to Staffed Bed Capacity for the same cluster. Adjust row references based on the actual row mapping:
- Row 35 uses rows 12, 19, 26
- Row 36 uses rows 13, 20, 27
- Row 37 uses rows 14, 21, 28
- Row 38 uses rows 15, 22, 29
- Row 39 uses rows 16, 23, 30
- Row 40 uses rows 17, 24, 31

## 4. Populate H42:L47 — Summary statistics

For each year column (H–L), in rows 42–47, write these formulas over the range H35:H40 (same column):
- Row 42 (MIN): `=MIN(H35:H40)`
- Row 43 (MAX): `=MAX(H35:H40)`
- Row 44 (MEDIAN): `=MEDIAN(H35:H40)`
- Row 45 (MEAN): `=AVERAGE(H35:H40)`
- Row 46 (25th percentile): `=PERCENTILE(H35:H40, 0.25)` or `=PERCENTILE.INC(H35:H40, 0.25)`
- Row 47 (75th percentile): `=PERCENTILE(H35:H40, 0.75)` or `=PERCENTILE.INC(H35:H40, 0.75)`

IMPORTANT: Check the actual labels in rows 42–47 column D (or nearby) to confirm which row is which statistic. Assign formulas according to the actual labels, not my assumed ordering.

## 5. Populate H50:L50 — Weighted mean

For each year column (H–L):
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```
This computes the weighted mean of the Net capacity headroom percentages weighted by Staffed Bed Capacity.

## 6. Save and verify
- Save the workbook.
- Reopen it and verify that all target cells contain formula strings (not None or numeric values).
- Print a sample of formulas from each block to confirm correctness.
- Confirm no new sheets were added.
- Confirm the file is saved at `/root/output/result.xlsx`.

## Critical notes
- Load the workbook with `openpyxl.load_workbook(path)` — do NOT pass `data_only=True`.
- When writing formulas, set cell.value to a string starting with `=`.
- Do NOT modify any cell formatting (font, fill, border, number_format, alignment, etc.).
- Do NOT add sheets, delete sheets, or rename sheets.
- Adapt all row/column references based on what you actually observe in step 1. My row numbers above are best guesses — the actual labels in the workbook are authoritative.
- For PERCENTILE, use `PERCENTILE.INC` if you want to match typical Excel behavior, but `PERCENTILE` also works.

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