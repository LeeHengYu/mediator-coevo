# Task Instruction

Execute the following steps precisely to complete the task.

## 0. Setup
```bash
mkdir -p /root/output
pip install openpyxl
```

## 1. Inspect the workbook structure
- Open `/root/data/workbook.xlsx` with openpyxl (keep formulas via `data_only=False`).
- Print the sheet names to confirm `Task` and `Data` exist.
- Print the contents of the `Task` sheet rows 1–55, columns A–M, to understand the layout: what is in column D (series codes), what is in row 10 (years), what the yellow cell regions look like, and what labels exist for rows 35–50.
- Print the `Data` sheet rows 1–40, columns A–Z (or however wide it goes), to understand the data layout, especially rows 21–38. Identify how series codes and years are arranged (which is the lookup key column/row).

## 2. Understand the data orientation on `Data` sheet
Determine:
- Are series codes in a column and years across a row (suggesting HLOOKUP or INDEX/MATCH)?
- Or are years in a column (suggesting VLOOKUP)?
- What exact row/column range holds the data in rows 21:38?
- What are the exact series codes in `Task!D12:D17`, `Task!D19:D24`, `Task!D26:D31`?
- What are the exact years in `Task!H10:L10`?

## 3. Populate Step 1 formulas in H12:L17, H19:L24, H26:L31
Using `INDEX/MATCH` pattern (most flexible), write formulas into each cell. The formula pattern for a cell at row `r`, column `c` should be:

```
=INDEX(Data!<data_range>, MATCH($D{r}, Data!<series_code_column>, 0), MATCH(H$10, Data!<year_row>, 0))
```

Adjust the references based on what you found in step 2:
- `$D{r}` — the series code in column D of the current row (absolute column, relative row)
- `H$10` (or I$10, J$10, etc.) — the year in row 10 (relative column, absolute row)
- `Data!<data_range>` — the rectangular data block in rows 21:38
- `Data!<series_code_column>` — the column containing series codes
- `Data!<year_row>` — the row containing years

Make sure:
- Column D reference uses `$D` (absolute column) so it doesn't shift when going across columns.
- Row 10 reference uses `$10` (absolute row) so it doesn't shift when going down rows.
- The data range, series code range, and year range are all anchored with `$` signs appropriately.

Write these formulas using openpyxl by assigning formula strings to each cell. Do NOT use `data_only=True`. Use the Translator or manual string construction to place the correct formula in each of the 90 cells (3 blocks × 6 rows × 5 columns).

## 4. Populate Step 2: Net patient flow (H35:L40)
Based on the task description:
- Patient Admissions are in H12:L17
- Patient Discharges are in H19:L24  
- Effective Bed Capacity is in H26:L31

Verify this by checking the labels in the Task sheet. The formula for cell H35 should be:
```
=(H12-H19)/H26*100
```
And similarly for each of the 6 hospitals × 5 years (H35:L40). Make sure row references correspond correctly (row 35 uses rows 12, 19, 26; row 36 uses rows 13, 20, 27; etc.).

## 5. Populate Step 2: Summary statistics (H42:L47)
For each column (H through L), calculate over the 6 hospital net-flow values (rows 35:40):
- H42: `=MIN(H35:H40)` (minimum)
- H43: `=MAX(H35:H40)` (maximum)
- H44: `=MEDIAN(H35:H40)` (median)
- H45: `=AVERAGE(H35:H40)` (simple mean)
- H46: `=PERCENTILE(H35:H40,0.25)` (25th percentile)
- H47: `=PERCENTILE(H35:H40,0.75)` (75th percentile)

Check the labels in column D/E/F/G of rows 42–47 to confirm the correct order of these statistics. Adjust the row assignments if the labels indicate a different order.

## 6. Populate Step 3: Weighted mean (H50:L50)
For each column, e.g. H50:
```
=SUMPRODUCT(H35:H40,H26:H31)/SUM(H26:H31)
```
This uses the net patient flow percentages as values and effective bed capacity as weights.

## 7. Save
Save the workbook to `/root/output/result.xlsx`. Do NOT change any existing formatting, do not add sheets, macros, VBA, external links, or helper tabs.

## 8. Verify
- Reopen `/root/output/result.xlsx` with openpyxl.
- Print the formulas in a sample of cells (e.g., H12, L17, H35, L40, H42, H47, H50, L50) to confirm they are correct formula strings.
- Confirm no extra sheets were added.
- Confirm the file exists and is non-empty.

## IMPORTANT NOTES
- Use openpyxl throughout. Do NOT use `data_only=True` when loading — you need to preserve and write formulas.
- When writing formulas, they must start with `=`.
- All cell references in formulas must use Excel-style notation (e.g., `Data!$A$21:$F$38`).
- Double-check the exact data layout before writing any formulas. The specific ranges depend on what you observe in the spreadsheet.
- If the Data sheet has years in a row and series codes in a column, INDEX(MATCH,MATCH) is the cleanest approach.
- Preserve all existing cell values, styles, and formatting. Only write into the specified target cells.

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