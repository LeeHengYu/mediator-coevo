# Task Instruction

You must update an Excel workbook and save the result. Follow these steps exactly.

## 0. Inspect the workbook
```bash
pip install openpyxl
```
Then run a Python script that:
- Opens `/root/data/workbook.xlsx` with `openpyxl` (with `data_only=False` so formulas are preserved).
- Prints the contents of sheet `Task` rows 1–55, columns A–M (both `.value` and cell formatting/fill color if easy).
- Prints sheet `Data` rows 18–40, columns A–Z (to understand the data layout).
- Prints the series codes in column D of sheet `Task` rows 12–31.
- Prints the years in row 10 of sheet `Task` columns H–L.
- Prints any existing formulas or values in the target ranges.

Study the output carefully before proceeding.

## 1. Understand the data layout
On sheet `Data`, rows 21–38 contain source records. Determine:
- Which column holds the series codes (should match column D of `Task`).
- Which row/column holds the year headers.
- How the data is organized (rows vs columns for series and years).

## 2. Populate H12:L17, H19:L24, H26:L31 with lookup formulas
For each cell in these ranges, write a formula that looks up the value from `Data!` rows 21–38 using:
- The series code from column D of the same row on `Task`
- The year from row 10 of the same column on `Task`

Use `INDEX(MATCH, MATCH)` pattern since the data is a 2D table. The formula pattern should be something like:
```
=INDEX(Data!$B$21:$Z$38, MATCH($D12, Data!$A$21:$A$38, 0), MATCH(H$10, Data!$B$20:$Z$20, 0))
```
Adjust the exact ranges based on what you discover in step 0. The key points:
- Row lookup: MATCH the series code in column D against the series code column in Data
- Column lookup: MATCH the year in row 10 against the year header row in Data
- The INDEX range should cover the data area (excluding headers)

Make sure to use mixed references correctly: `$D12` (column absolute, row relative) and `H$10` (column relative, row absolute).

## 3. Populate H35:L40 with Net SLA buffer formula
The formula is: `(Latency Budget Preserved - Latency Budget Consumed) / Covered Request Capacity * 100`

Based on the layout:
- H12:L17 likely corresponds to one metric block (e.g., Latency Budget Preserved)
- H19:L24 likely corresponds to another metric block (e.g., Latency Budget Consumed)
- H26:L31 likely corresponds to Covered Request Capacity

Verify which block is which by checking labels in the Task sheet. Then for cell H35:
```
=(H12-H19)/H26*100
```
(Adjust row references based on actual block assignments. The six services in rows 35–40 should correspond to the six services in rows 12–17, 19–24, 26–31.)

## 4. Populate H42:L47 with summary statistics
For each column (H through L), calculate over the H35:L40 range:
- Row 42: Minimum → `=MIN(H35:H40)`
- Row 43: Maximum → `=MAX(H35:H40)`
- Row 44: Median → `=MEDIAN(H35:H40)`
- Row 45: Mean → `=AVERAGE(H35:H40)`
- Row 46: 25th percentile → `=PERCENTILE(H35:H40, 0.25)`
- Row 47: 75th percentile → `=PERCENTILE(H35:H40, 0.75)`

Check the labels in column D or G of rows 42–47 to confirm the correct order.

## 5. Populate H50:L50 with weighted mean
For each column:
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```
This computes the weighted mean of Net SLA buffer percentages weighted by Covered Request Capacity.

## 6. Save
- Save the workbook to `/root/output/result.xlsx` (create `/root/output/` if needed).
- Do NOT change formatting, do NOT add sheets, macros, VBA, external links, or helper tabs.

## 7. Verify
After saving, reopen the file and print the formulas in all target cells to confirm they are correctly written. Also evaluate a few cells manually (using the Data sheet values) to cross-check.

IMPORTANT: Do the inspection step FIRST and study the actual layout before writing any formulas. The exact row/column references depend on the actual workbook structure.

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