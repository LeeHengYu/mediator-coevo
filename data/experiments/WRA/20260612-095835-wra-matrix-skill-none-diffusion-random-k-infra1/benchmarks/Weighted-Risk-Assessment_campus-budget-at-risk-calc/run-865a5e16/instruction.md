# Task Instruction

Execute the following steps to produce /root/output/result.xlsx from /root/data/workbook.xlsx.

## Preparation
1. `mkdir -p /root/output`
2. `pip install openpyxl` (if not already installed).
3. Open `/root/data/workbook.xlsx` with openpyxl (`data_only=False` so existing formulas are preserved).
4. Inspect the `Data` sheet:
   - Confirm the header row layout (row 20 or 21) and identify which column holds the series codes and which columns hold the year-indexed data.
   - Identify the exact row range 21:38 that contains the source records.
5. Inspect the `Task` sheet:
   - Confirm column D contains series codes in rows 12-17, 19-24, 26-31.
   - Confirm row 10 contains years in columns H-L.
   - Confirm H35:L40 is the Net budget buffer area, H42:L47 is the stats area, and H50:L50 is the weighted mean row.

## Step 1 – Lookup formulas in H12:L17, H19:L24, H26:L31
For each cell in these three 6×5 blocks, write an Excel formula string using INDEX/MATCH that:
- Matches the series code from column D of the current row against the series-code column on the Data sheet (rows 21:38).
- Matches the year from row 10 of the current column against the header row on the Data sheet.
- Returns the intersecting value.

Use this pattern (adjust Data sheet references based on your inspection):
```
=INDEX(Data!$B$21:$<lastcol>$38, MATCH($D12, Data!$A$21:$A$38, 0), MATCH(H$10, Data!$B$20:$<lastcol>$20, 0))
```
Replace `$B`, `$A`, `$<lastcol>`, and `$20` with the actual column letters and header row discovered during inspection. The key requirements:
- `$D12` uses an absolute column but relative row so it shifts down within each block.
- `H$10` uses a relative column but absolute row so it shifts right across columns H-L.
- All Data-sheet references are fully absolute ($).

Write the formula as a string to each cell (e.g., `ws['H12'] = '=INDEX(...)'`). Do NOT compute values in Python.

## Step 2a – Net budget buffer in H35:L40
Based on the task description, the three lookup blocks correspond to:
- H12:L17 → Committed Funding
- H19:L24 → Operating Spend  
- H26:L31 → Approved Budget Base

Verify this by checking the labels on the Task sheet near those rows. Then for each cell in H35:L40 (6 rows × 5 columns), write a formula:
```
=(H12-H19)/H26*100
```
adjusting row references to align each department row (row 35↔rows 12,19,26; row 36↔rows 13,20,27; etc.) and each column H-L.

## Step 2b – Summary statistics in H42:L47
For each column H through L, write these formulas:
- Row 42 (MIN):    `=MIN(H35:H40)`
- Row 43 (MAX):    `=MAX(H35:H40)`
- Row 44 (MEDIAN): `=MEDIAN(H35:H40)`
- Row 45 (MEAN):   `=AVERAGE(H35:H40)`
- Row 46 (25th %): `=PERCENTILE(H35:H40,0.25)`
- Row 47 (75th %): `=PERCENTILE(H35:H40,0.75)`

Check the labels in column D or nearby to confirm the exact order of MIN/MAX/MEDIAN/MEAN/P25/P75 and adjust row assignments accordingly.

## Step 3 – Weighted mean in H50:L50
For each column H through L:
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```
This computes the weighted mean of Net budget buffer percentages weighted by Approved Budget Base.

## Finalization
1. Save the workbook to `/root/output/result.xlsx`.
2. Re-open the saved file and verify:
   - Cells H12, L17, H19, L24, H26, L31 contain formula strings (start with '=').
   - Cells H35, L40 contain formula strings.
   - Cells H42, H47 contain formula strings.
   - Cell H50 contains a formula string.
   - No new sheets were added.
3. Print a summary of spot-checked cells to confirm formulas are present.

## Critical Reminders
- Write FORMULA STRINGS, never computed Python values. Every target cell must start with '='.
- Do not add sheets, macros, VBA, external links, or helper columns.
- Do not alter existing formatting.
- If the row/column layout discovered during inspection differs from assumptions, adapt all references accordingly before writing any formulas.
- The avoid-recheck artifact warns that cells returning None means formulas were never written; double-check every target cell is populated after the write loop.

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