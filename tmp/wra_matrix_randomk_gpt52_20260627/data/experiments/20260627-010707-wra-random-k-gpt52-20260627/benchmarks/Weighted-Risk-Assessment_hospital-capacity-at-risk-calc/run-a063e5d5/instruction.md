# Task Instruction

Execute the following steps to produce /root/output/result.xlsx from /root/data/workbook.xlsx.

## 0 – Preparation
```bash
mkdir -p /root/output
```
Open and inspect the workbook with openpyxl to understand the exact layout:
- Sheet names, row/column structure of `Task` and `Data`.
- Confirm yellow-highlighted target ranges: H12:L17, H19:L24, H26:L31, H35:L40, H42:L47, H50:L50.
- Confirm that column D holds series codes, row 10 holds years, and Data!A21:?38 holds the source table.
- Print a few sample cells so you know the exact series codes and year values.

## 1 – Populate lookup formulas (H12:L17, H19:L24, H26:L31)

For every cell in these three 6×5 blocks, write an INDEX/MATCH formula that:
- Looks up the series code from column D of the current row against Data column A (rows 21–38).
- Looks up the year from row 10 of the current column against Data row 20 (or whichever row holds the year headers).
- Returns the intersection from the Data table.

Use this pattern (adjust exact Data ranges after inspection):
```
=INDEX(Data!$B$21:$<lastcol>$38, MATCH($D12,Data!$A$21:$A$38,0), MATCH(H$10,Data!$B$20:$<lastcol>$20,0))
```
Key anchoring rules:
- `$D12` — column absolute, row relative (so it shifts down within each block).
- `H$10` — row absolute, column relative (so it shifts right across years).
- All Data ranges are fully absolute.

Confirm the exact Data header row and last column by inspection before writing formulas.

## 2 – Net capacity headroom (H35:L40)

For each of the 6 hospital clusters (rows 35–40) and 5 year columns (H–L), write:
```
=(H12 - H19) / H26 * 100
```
where:
- Row 12–17 = Available Care Slots (block 1)
- Row 19–24 = Occupied Care Slots (block 2)
- Row 26–31 = Staffed Bed Capacity (block 3)

The row offsets must correspond: row 35 uses rows 12, 19, 26; row 36 uses rows 13, 20, 27; etc.

## 3 – Summary statistics (H42:L47)

For each year column (H–L), compute column-wise stats over H35:L40:
- Row 42: `=MIN(H35:H40)`
- Row 43: `=MAX(H35:H40)`
- Row 44: `=MEDIAN(H35:H40)`
- Row 45: `=AVERAGE(H35:H40)`
- Row 46: `=PERCENTILE(H35:H40,0.25)`  ← use PERCENTILE, not PERCENTILE.INC or PERCENTILE.EXC, to avoid #NAME? errors in older Excel compatibility modes. However, first check the workbook: if existing formulas already use .INC/.EXC style, match that style. If unsure, use PERCENTILE and QUARTILE which are universally recognized.
- Row 47: `=PERCENTILE(H35:H40,0.75)`

IMPORTANT: Verify the exact row assignments (min/max/median/mean/25th/75th) by checking any labels in column D or nearby columns for rows 42–47. Place each formula in the row that matches its label.

## 4 – Weighted mean (H50:L50)

For each year column:
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```
This computes the weighted average of the headroom percentages using Staffed Bed Capacity as weights.

## 5 – Save

Save the workbook to `/root/output/result.xlsx` using openpyxl, preserving all existing formatting. Do NOT recalculate values — write string formulas only (do not use data_only mode).

## 6 – Validation

After saving, reopen the file and:
1. Confirm all target cells contain formula strings (start with '='), not None or literal values.
2. Spot-check that anchoring is correct: e.g., cell I12 should reference I$10 (not H$10) and $D12.
3. Confirm no cells in the target ranges are empty.
4. Confirm no extra sheets were added.
5. Print a summary of checks.

## Critical Notes
- All formulas must be written as Excel formula strings, not Python-computed values.
- Do NOT use PERCENTILE.INC or PERCENTILE.EXC unless the workbook already uses that style — these cause #NAME? errors in some contexts (see failed artifact from api-sla-at-risk-calc).
- Do NOT add sheets, macros, VBA, external links, or helper tabs.
- Do NOT modify any existing formatting.
- Inspect before writing — confirm every range reference against the actual workbook layout.

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