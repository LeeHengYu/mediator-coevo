# Task Instruction

Execute the following steps to produce /root/output/result.xlsx from /root/data/workbook.xlsx.

## 0 – Preparation
```bash
mkdir -p /root/output
pip install openpyxl
```

## 1 – Inspect the workbook
Open /root/data/workbook.xlsx with openpyxl (data_only=False) and print:
- Sheet names.
- Task!D12:D17, D19:D24, D26:D31 (series codes for the three blocks).
- Task!H10:L10 (year headers).
- Task!H35:H40 row labels or D35:D40 if present.
- Data!A21:A38 and Data!row 20 or 21 column headers (to understand the data layout).
- Task!H42:H47 row labels (to identify which stat goes in which row).
- Task!H50 row label and any weight references.
- Task!D35:D40 or G35:G40 for plant names / series codes used in Step 2.
- Task!D42:D47 or G42:G47 for stat labels.
- Task!D50 or G50 for the weighted-mean label.

This inspection determines the exact layout before writing any formulas.

## 2 – Write formulas with openpyxl
Use a Python script with openpyxl to write formulas into the workbook. Do NOT use data_only mode for writing.

### Step 1 – Lookup formulas in H12:L17, H19:L24, H26:L31
For each cell (row r, column c where c = 8..12 for H..L):
- Let `series_ref` = the cell in column D of the same row (e.g., $D12).
- Let `year_ref` = the cell in row 10 of the same column (e.g., H$10).
- Use INDEX-MATCH-MATCH against Data sheet rows 21:38.
  Formula pattern (adjust column range based on inspection):
  ```
  =INDEX(Data!$B$21:$XX$38,MATCH($D12,Data!$A$21:$A$38,0),MATCH(H$10,Data!$B$20:$XX$20,0))
  ```
  Adjust the column extent ($XX) to match the actual data range found during inspection. Use absolute row references for the Data range and mixed references for the series code ($D) and year (row $10) so the formula copies correctly.

### Step 2 – Net production slack in H35:L40
Based on the instruction:
```
Net production slack = (Finished Output - Scrap And Rework) / Rated Production Capacity * 100
```
The three blocks are:
- H12:L17 = one metric (e.g., Finished Output)
- H19:L24 = another metric (e.g., Scrap And Rework)
- H26:L31 = another metric (e.g., Rated Production Capacity)

Determine which block is which from the row labels visible during inspection. Then for each cell in H35:L40:
```
=(H12-H19)/H26*100
```
(Adjust row offsets so each of the 6 plants maps correctly across the three blocks.)

### Step 2 continued – Statistics in H42:L47
For each column c in H..L, write the six statistics. Based on the row labels found during inspection, assign:
- Minimum: `=MIN(H35:H40)`
- Maximum: `=MAX(H35:H40)`
- Median: `=MEDIAN(H35:H40)`
- Simple mean: `=AVERAGE(H35:H40)`
- 25th percentile: `=PERCENTILE(H35:H40,0.25)`
- 75th percentile: `=PERCENTILE(H35:H40,0.75)`

**IMPORTANT**: Use `PERCENTILE` (not `PERCENTILE.INC` or `_xlfn.PERCENTILE.INC`) to avoid #NAME? errors. Match the stat to the correct row based on the labels found during inspection.

### Step 3 – Weighted mean in H50:L50
For each column c in H..L:
```
=SUMPRODUCT(H35:H40,H26:H31)/SUM(H26:H31)
```
This computes the weighted mean of the net production slack percentages using Rated Production Capacity as weights.

## 3 – Save
Save the workbook to `/root/output/result.xlsx`. Do not modify formatting, do not add sheets.

## 4 – Validate
Reopen the saved file with openpyxl (data_only=False) and print the formulas in a sample of cells (e.g., H12, L17, H35, L40, H42, H47, H50, L50) to confirm they were written correctly and contain no Python string artifacts.

## Key Constraints
- Do NOT use `_xlfn.` prefixed function names.
- Use plain `PERCENTILE`, `MIN`, `MAX`, `MEDIAN`, `AVERAGE`, `SUMPRODUCT`, `SUM`, `INDEX`, `MATCH`.
- Do NOT add macros, VBA, external links, helper tabs, or new sheets.
- Preserve all existing formatting.
- The inspection step is critical — do it first and adapt all formulas to the actual layout.

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