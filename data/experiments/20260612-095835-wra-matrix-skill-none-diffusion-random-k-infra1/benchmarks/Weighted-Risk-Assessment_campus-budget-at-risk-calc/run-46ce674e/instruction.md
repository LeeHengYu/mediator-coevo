# Task Instruction

Execute the following steps in order to produce `/root/output/result.xlsx`.

## 0 — Environment setup
```bash
pip install openpyxl
mkdir -p /root/output
```

## 1 — Inspect the workbook structure
Open `/root/data/workbook.xlsx` with openpyxl and print:
- Sheet names.
- On sheet `Task`: the contents of rows 10-11 (to see headers/years in H-L), column D for rows 12-31 (series codes), rows 35-50 column D (labels), and any existing content in H12:L50.
- On sheet `Data`: rows 19-40 to understand the layout — especially which row holds headers, where series codes live, and where year-columns start.

Print everything clearly before writing any formulas.

## 2 — Understand the data layout on `Data` sheet
Identify:
- The row range 21:38 that contains source records.
- Which column holds the series code (the lookup key).
- Which row holds the year headers that correspond to the years in `Task!H10:L10`.
- Whether the data is arranged so that VLOOKUP or INDEX/MATCH is more natural.

## 3 — Write lookup formulas in H12:L17, H19:L24, H26:L31
For each yellow cell (e.g., H12), write a formula using INDEX/MATCH (preferred) or VLOOKUP/MATCH:
- The lookup value for the series code comes from column D of the same row on `Task`.
- The lookup value for the year comes from row 10 of the same column on `Task`.
- The lookup range is `Data!$A$21:$Z$38` (adjust column range based on inspection).
- Use MATCH to find the row by series code and MATCH to find the column by year.

Example pattern (adjust references after inspection):
```
=INDEX(Data!$B$21:$Z$38, MATCH($D12, Data!$A$21:$A$38, 0), MATCH(H$10, Data!$B$20:$Z$20, 0))
```
Adjust the exact ranges based on what you found in step 1-2. Make sure:
- The series-code column reference in MATCH uses an absolute column (`$D12`).
- The year row reference in MATCH uses an absolute row (`H$10`).
- Both MATCH functions use exact match (0).

Write formulas as strings using openpyxl (do NOT compute values in Python — write actual Excel formulas).

## 4 — Write Net budget buffer formulas in H35:L40
For each of the six departments (rows 35-40), calculate:
```
=(committed_funding_cell - operating_spend_cell) / approved_budget_base_cell * 100
```
where:
- Committed Funding block is H12:L17
- Operating Spend block is H19:L24
- Approved Budget Base block is H26:L31

So for H35: `=(H12-H19)/H26*100`, for H36: `=(H13-H20)/H27*100`, etc.
Confirm the row mapping by checking that the department order in rows 35-40 matches the order in the three lookup blocks.

## 5 — Write summary statistics in H42:L47
For each column (H through L):
- H42: `=MIN(H35:H40)`
- H43: `=MAX(H35:H40)`
- H44: `=MEDIAN(H35:H40)`
- H45: `=AVERAGE(H35:H40)`
- H46: `=PERCENTILE(H35:H40, 0.25)`
- H47: `=PERCENTILE(H35:H40, 0.75)`

Check the labels in column D (or wherever) for rows 42-47 to confirm the order is min, max, median, mean, 25th, 75th. Adjust row assignments if the labels differ.

## 6 — Write weighted mean in H50:L50
For each column (H through L):
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```
This uses the Net budget buffer percentages as values and Approved Budget Base as weights.

## 7 — Save
Save the workbook to `/root/output/result.xlsx`. Do NOT change any formatting, do NOT add sheets.

## 8 — Validate
Reopen `/root/output/result.xlsx` with openpyxl and print:
- The formula (not value) in cells H12, L17, H19, L24, H26, L31, H35, L40, H42, H47, H50, L50.
- Confirm none of these cells are None or empty.
- Confirm no new sheets were added.

## Critical Notes
- Write Excel formula strings, not Python-computed values. openpyxl stores them as strings starting with `=`.
- Do not use `data_only=True` when reading.
- Preserve all existing formatting — do not touch fonts, fills, borders, column widths, etc.
- The avoid-artifact warns about cells returning None: this happens when formulas are not written or cell coordinates are wrong. Double-check every row/column index.

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