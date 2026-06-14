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
- On sheet `Task`: read cells D12:D17, D19:D24, D26:D31 to see the series codes; read H10:L10 to see the years; read row 35-40 labels; read row 42-47 labels (min/max/median/mean/25th/75th); read row 50 label.
- On sheet `Data`: read rows 21-38 to understand the data layout — specifically which row has headers, which column has series codes, and how years are arranged (row-wise or column-wise). Print enough to understand the exact structure.
- Check which cells in H12:L17, H19:L24, H26:L31 are currently empty (the yellow cells to populate).
- Print all findings before proceeding.

## 2. Populate lookup formulas in H12:L17, H19:L24, H26:L31

Based on the inspection, write a Python script using openpyxl to insert spreadsheet formulas (not computed values) into every cell in these three blocks.

Each formula must use two inputs:
- The series code from column D of the same row (e.g., $D12 for row 12)
- The year from row 10 of the same column (e.g., H$10 for column H)

Use one of these allowed lookup patterns: INDEX/MATCH, VLOOKUP/MATCH, HLOOKUP/MATCH, or XLOOKUP/MATCH.

The data source range is `Data!` rows 21:38. Determine from inspection whether the data is arranged with series codes in a column and years in a row, or vice versa, and choose the appropriate lookup pattern.

IMPORTANT: When writing formulas with openpyxl, if a formula starts with `=`, openpyxl treats it as a formula. Use absolute/mixed references appropriately ($D12 for the series code column, H$10 for the year row) so formulas can be written in a loop.

Example pattern if data has series codes in column A and years across a header row on Data sheet:
```
=INDEX(Data!$B$21:$Z$38, MATCH($D12, Data!$A$21:$A$38, 0), MATCH(H$10, Data!$B$20:$Z$20, 0))
```
Adjust the exact ranges based on what you find in the inspection step. The key is that the range references on the Data sheet must be correct.

## 3. Calculate Net capacity headroom in H35:L40

Insert spreadsheet formulas in H35:L40. The formula for each cell is:
```
=(AvailableCareSlots - OccupiedCareSlots) / StaffedBedCapacity * 100
```
where:
- Available Care Slots are in H12:L17 (rows 12-17 correspond to rows 35-40)
- Occupied Care Slots are in H19:L24 (rows 19-24 correspond to rows 35-40)
- Staffed Bed Capacity are in H26:L31 (rows 26-31 correspond to rows 35-40)

So for cell H35: `=(H12-H19)/H26*100`
For cell H36: `=(H13-H20)/H27*100`
...and so on for all 6 rows × 5 columns.

## 4. Calculate summary statistics in H42:L47

Insert spreadsheet formulas for column-wise statistics over H35:L40:
- Row 42 (Minimum): `=MIN(H35:H40)` for each column H-L
- Row 43 (Maximum): `=MAX(H35:H40)` for each column H-L
- Row 44 (Median): `=MEDIAN(H35:H40)` for each column H-L
- Row 45 (Mean): `=AVERAGE(H35:H40)` for each column H-L
- Row 46 (25th percentile): `=PERCENTILE(H35:H40,0.25)` for each column H-L
- Row 47 (75th percentile): `=PERCENTILE(H35:H40,0.75)` for each column H-L

IMPORTANT: Verify the labels in rows 42-47 during inspection to confirm which row is which statistic. Adjust the row assignments if the labels differ from the assumed order above.

## 5. Weighted mean in H50:L50

Insert a SUMPRODUCT-based weighted mean formula in each cell H50:L50:
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```
This uses the Net capacity headroom percentages (H35:H40) as values and the Staffed Bed Capacity (H26:H31) as weights.

## 6. Save

Save the workbook to `/root/output/result.xlsx` preserving all existing formatting. When loading with openpyxl, do NOT use `data_only=True`. Load with `keep_vba=False` (default). Do not modify any other cells or sheets.

## 7. Verify

After saving, reload `/root/output/result.xlsx` and print:
- A sample of formulas from each block (H12, L17, H19, L24, H26, L31, H35, L40, H42, L47, H50, L50)
- Confirm they are formula strings (start with '=')
- Confirm no cells in the target ranges are None/empty

## Critical Notes
- Do NOT compute values in Python; insert Excel formula strings.
- Use openpyxl to load the workbook without data_only so existing formulas are preserved.
- Do not add sheets, macros, VBA, external links, or helper tabs.
- The inspection in step 1 is essential — do not skip it. Adjust all subsequent formula ranges based on what you actually find in the data.

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