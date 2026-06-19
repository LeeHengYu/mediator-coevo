# Task Instruction

Execute the following steps precisely to complete the task.

## 0. Setup
```bash
mkdir -p /root/output
pip install openpyxl
```

## 1. Inspect the workbook structure
Open `/root/data/workbook.xlsx` with openpyxl and inspect:
- Sheet names (confirm `Task` and `Data` exist)
- On sheet `Task`: read row 10 (the year headers in columns H–L), column D rows 12–17, 19–24, 26–31 to understand the series codes and region labels
- Read the yellow cell ranges H12:L17, H19:L24, H26:L31 to confirm they are currently empty
- Read rows 35–40 (Net reliability gap region rows), rows 42–47 (stats rows), row 50 (weighted mean row)
- On sheet `Data`: read rows 21–38 to understand the data layout — identify which column holds the series code, which row holds years, and how the data is arranged (is it a vertical table with series codes in one column and years across columns, or some other layout?)

Print all of this information before proceeding. This is critical to get the formulas right.

## 2. Populate lookup formulas in H12:L17, H19:L24, H26:L31

Based on the inspection, write a Python script using openpyxl to set formula strings in each cell. Each formula should use INDEX/MATCH (or VLOOKUP with MATCH, etc.) referencing:
- The series code from column D of the same row on sheet `Task`
- The year from row 10 of the same column on sheet `Task`
- The data range on sheet `Data` rows 21:38

IMPORTANT: Use absolute references to the Data sheet (e.g., `Data!$A$21:$A$38` for series codes, `Data!$B$20:$Z$20` for years — adjust based on actual layout). Anchor references appropriately so formulas work across the range.

For example, if Data sheet has series codes in column A rows 21:38 and years in row 20 starting column B, a suitable INDEX/MATCH formula for cell H12 would be:
`=INDEX(Data!$B$21:$Z$38,MATCH($D12,Data!$A$21:$A$38,0),MATCH(H$10,Data!$B$20:$Z$20,0))`

Adjust column/row references based on actual inspection results. The key contract:
- Row anchor: `$D12` (column locked, row relative) for the series code
- Column anchor: `H$10` (row locked, column relative) for the year
- Data ranges: fully absolute with sheet prefix

Set these formulas for all 3 blocks (H12:L17, H19:L24, H26:L31) — that's 6 rows × 5 columns = 30 cells per block, 90 cells total.

## 3. Net reliability gap formulas in H35:L40

The formula is: `(Successful API Requests - Failed API Requests) / Compute Capacity * 100`

Based on inspection, identify which block corresponds to which metric:
- If H12:L17 = Successful API Requests, H19:L24 = Failed API Requests, H26:L31 = Compute Capacity (verify from the labels in the Task sheet)
- Then for cell H35: `=(H12-H19)/H26*100` (adjust row references for each of the 6 regions)

Make sure the row mapping is correct — region 1 in row 35 should reference region 1 data from each block. Verify the region order matches across blocks.

## 4. Summary statistics in H42:L47

For each column H through L, in the stats rows:
- Minimum: `=MIN(H35:H40)` (adjust column)
- Maximum: `=MAX(H35:H40)`
- Median: `=MEDIAN(H35:H40)`
- Mean: `=AVERAGE(H35:H40)`
- 25th percentile: `=PERCENTILE(H35:H40,0.25)` or `=PERCENTILE.INC(H35:H40,0.25)`
- 75th percentile: `=PERCENTILE(H35:H40,0.75)` or `=PERCENTILE.INC(H35:H40,0.75)`

Check the labels in column D (or nearby) for rows 42–47 to determine which row gets which statistic. Map them correctly.

## 5. Weighted mean in H50:L50

For each column H through L:
`=SUMPRODUCT(H35:H40,H26:H31)/SUM(H26:H31)`

This computes the weighted mean of Net reliability gap percentages using Compute Capacity as weights.

## 6. Save
Save as `/root/output/result.xlsx`. Do NOT use `data_only=True` when loading. Load with openpyxl preserving formulas (default behavior). After saving, re-open and verify that the formula cells contain formula strings (not None or 0).

## 7. Validation
- Re-open `/root/output/result.xlsx` with openpyxl
- Print the contents of cells H12, L17, H19, L24, H26, L31, H35, L40, H42, L47, H50, L50
- Confirm they all contain formula strings starting with '='
- Confirm no new sheets were added
- Confirm the workbook has exactly the original sheets

## Critical Notes
- Do NOT use `data_only=True` when opening the workbook
- Do NOT add any sheets, macros, or VBA
- Do NOT modify formatting — only set cell values (formulas)
- The inspection in step 1 is essential — do not skip it. The exact row/column layout of the Data sheet determines every formula.
- If the Data sheet layout differs from assumptions, adapt all formulas accordingly before writing them.

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
Task metadata: author_email=catpaw@meituan.com, author_name=CatPaw Task Engineer, category=spreadsheet-formula-reuse, difficulty=easy, tags=[excel, formulas, lookup, statistics, weighted-mean].
Verifier config: timeout_sec=600.0.