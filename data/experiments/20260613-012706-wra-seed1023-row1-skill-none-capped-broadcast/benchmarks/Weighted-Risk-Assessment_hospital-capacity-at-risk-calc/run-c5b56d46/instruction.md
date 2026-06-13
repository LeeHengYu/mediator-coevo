# Task Instruction

Execute the following steps precisely to produce /root/output/result.xlsx.

## 0 – Environment & Inspection
```bash
mkdir -p /root/output
pip install openpyxl
```
Open `/root/data/workbook.xlsx` with openpyxl and inspect:
- Sheet `Task`: print rows 9-11 (to see year headers in row 10), print column D rows 12-31 (series codes), print rows 33-50 to understand layout.
- Sheet `Data`: print rows 19-40 to see the data table structure (headers, row labels, columns).

Note the exact column letters and row numbers for the Data table. Identify:
- Which column holds the series codes (keys) in the Data sheet.
- Which row holds the year headers in the Data sheet.
- The exact range of the data table (e.g., rows 21:38, and which columns).

## 1 – Populate lookup formulas in H12:L17, H19:L24, H26:L31

For each cell in these three blocks, write a formula that looks up the value from the `Data` sheet using the series code in column D of the same row on `Task` and the year in row 10 of the same column on `Task`.

Use INDEX/MATCH pattern. The formula pattern for cell H12 should be something like:
```
=INDEX(Data!<data_columns>,MATCH($D12,Data!<key_column>,0),MATCH(H$10,Data!<year_header_row>,0))
```
Adjust the exact references after inspecting the Data sheet layout. The `$D12` locks column D; `H$10` locks row 10.

Apply the analogous formula to every cell in H12:L17, H19:L24, H26:L31 (that's 5 columns × 6 rows × 3 blocks = 90 cells).

## 2 – Net capacity headroom in H35:L40

For each of the 6 hospital clusters (rows 35-40) and 5 year columns (H-L), write:
```
=(H12 - H19) / H26 * 100
```
where row 12 corresponds to Available Care Slots, row 19 to Occupied Care Slots, row 26 to Staffed Bed Capacity — adjusted for the cluster offset within each block. Specifically:
- H35 = (H12 - H19) / H26 * 100
- H36 = (H13 - H20) / H27 * 100
- H37 = (H14 - H21) / H28 * 100
- H38 = (H15 - H22) / H29 * 100
- H39 = (H16 - H23) / H30 * 100
- H40 = (H17 - H24) / H31 * 100
And similarly for columns I through L.

## 3 – Summary statistics in H42:L47

For each column (H through L), compute over the 6-cell range in the corresponding column of rows 35:40:
- Row 42: MIN, e.g., `=MIN(H35:H40)`
- Row 43: MAX, e.g., `=MAX(H35:H40)`
- Row 44: MEDIAN, e.g., `=MEDIAN(H35:H40)`
- Row 45: AVERAGE (simple mean), e.g., `=AVERAGE(H35:H40)`
- Row 46: 25th percentile — use `=PERCENTILE(H35:H40,0.25)` 
- Row 47: 75th percentile — use `=PERCENTILE(H35:H40,0.75)`

**CRITICAL**: The previous run failed with `#NAME?` on percentile formulas. To handle this:
1. First try writing the formulas as `=PERCENTILE(H35:H40,0.25)` (legacy name).
2. After saving, open the file with openpyxl in data_only mode or use a test evaluation to check if the formulas are recognized.
3. If the verifier environment uses LibreOffice or an engine that doesn't recognize `PERCENTILE`, also try `PERCENTILE.INC`. However, note that openpyxl may have issues with dot-containing function names.
4. **Important openpyxl detail**: When writing formulas with dots in function names (like `PERCENTILE.INC`), openpyxl may prefix them with `_xlfn.` internally. If you write `=PERCENTILE.INC(...)`, openpyxl might not handle it correctly. Instead, try writing it as `=_xlfn.PERCENTILE.INC(H35:H40,0.25)` if plain `PERCENTILE` doesn't work.
5. **Safest approach**: First try plain `PERCENTILE`. If after saving and re-reading the file the formulas show `#NAME?`, switch to `=_xlfn.PERCENTILE.INC(...)` for rows 46-47.

Actually, based on the failure feedback, let me be more specific: Use `=PERCENTILE(H35:H40,0.25)` first. The `#NAME?` error in the previous run may have been caused by using `PERCENTILE.INC` without the `_xlfn.` prefix. Plain `PERCENTILE` is the safest legacy function name that works across most engines.

## 4 – Weighted mean in H50:L50

For each column H through L:
```
=SUMPRODUCT(H35:H40,H26:H31)/SUM(H26:H31)
```
This computes the weighted mean of the net capacity headroom percentages using Staffed Bed Capacity as weights.

## 5 – Save and Validate

Save to `/root/output/result.xlsx`. Then:
1. Re-open the file with openpyxl.
2. Print all formulas in the key cells (H12, H19, H26, H35, H42-H47, H50) to confirm they are correctly written.
3. Check that no cells in the required ranges are None or empty.
4. Verify sheet names are exactly `Task` and `Data` (no extra sheets).
5. Confirm formatting was not altered (do not change any cell styles, number formats, fills, fonts, borders, or column widths).

## Important Constraints
- Do NOT add sheets, macros, VBA, external links, or helper tabs.
- Do NOT modify existing formatting.
- Only write formulas into the specified yellow cells.
- Use openpyxl for all operations.
- Make sure to preserve all existing content in both sheets.

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