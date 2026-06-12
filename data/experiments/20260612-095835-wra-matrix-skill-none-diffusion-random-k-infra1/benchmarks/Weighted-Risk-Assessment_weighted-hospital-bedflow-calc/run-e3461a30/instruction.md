# Task Instruction

## Task: Populate formulas in /root/data/workbook.xlsx and save to /root/output/result.xlsx

### Step 0: Inspect the workbook
1. `mkdir -p /root/output`
2. Open and inspect `/root/data/workbook.xlsx` using openpyxl (or similar). Print:
   - Sheet names
   - The contents of sheet `Task` rows 1-55, columns A-L (values AND formulas if any). Pay special attention to:
     - Column D rows 12-17, 19-24, 26-31 (series codes)
     - Row 10 columns H-L (years)
     - Row 35-40 labels (hospital names)
     - Rows 42-47 labels (min, max, median, mean, 25th, 75th percentile)
     - Row 50 label
   - The contents of sheet `Data` rows 1-40, focusing on rows 21-38. Print all column headers and the structure so you understand how data is laid out (which column has series codes, which columns/rows have years and values).
3. Identify the exact structure of the Data sheet: Is data arranged with series codes in a column and years across columns (suitable for VLOOKUP/HLOOKUP), or some other layout? Note the exact column letters and row numbers.

### Step 1: Populate lookup formulas in yellow cells

For each cell in the three blocks `H12:L17`, `H19:L24`, `H26:L31` on sheet `Task`:
- The formula must use TWO inputs: the series code from column D of that row, and the year from row 10 of that column.
- The lookup source is sheet `Data` rows 21:38.
- Use one of the allowed patterns: `INDEX(MATCH,MATCH)`, `VLOOKUP+MATCH`, `HLOOKUP+MATCH`, or `XLOOKUP+MATCH`.
- Use `INDEX/MATCH` as the preferred approach since it's most flexible.
- Make sure references to the Data sheet are absolute where needed so formulas can be filled across the block.
- Example pattern (adjust based on actual Data sheet structure):
  - If Data has series codes in column A and years in a header row, use something like:
    `=INDEX(Data!$B$21:$F$38, MATCH($D12, Data!$A$21:$A$38, 0), MATCH(H$10, Data!$B$20:$F$20, 0))`
  - Adjust column/row references based on what you actually find in the Data sheet.

### Step 2: Net patient flow and summary statistics

In `H35:L40`, calculate Net Patient Flow for each of 6 hospitals:
- Formula: `(Patient Admissions - Patient Discharges) / Effective Bed Capacity * 100`
- Patient Admissions should be from the H12:L17 block, Patient Discharges from H19:L24, and Effective Bed Capacity from H26:L31.
- For H35: `=(H12-H19)/H26*100` (adjust row numbers based on which hospital maps to which row)
- Make sure the hospital ordering in rows 35-40 matches the ordering in rows 12-17, 19-24, 26-31. Verify by checking labels in column D or nearby columns.

In `H42:L47`, calculate column-wise summary statistics over H35:L40:
- Row 42 (Min): `=MIN(H35:H40)` etc.
- Row 43 (Max): `=MAX(H35:H40)` etc.
- Row 44 (Median): `=MEDIAN(H35:H40)` etc.
- Row 45 (Mean): `=AVERAGE(H35:H40)` etc.
- Row 46 (25th percentile): `=PERCENTILE(H35:H40, 0.25)` etc.
- Row 47 (75th percentile): `=PERCENTILE(H35:H40, 0.75)` etc.
- **IMPORTANT**: Check the actual labels in column D (or nearby) for rows 42-47 to determine the correct order of min/max/median/mean/25th/75th. Match the formula to the label, not my assumed ordering.

### Step 3: Weighted mean in H50:L50
- Use SUMPRODUCT to calculate weighted mean of Net Patient Flow percentages (H35:H40) weighted by Effective Bed Capacity (H26:H31):
  `=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)`
- Apply this for each column H through L.

### Step 4: Save and verify
1. Save the workbook to `/root/output/result.xlsx`. Preserve all existing formatting. Do NOT create new sheets or remove any sheets.
2. Re-open `/root/output/result.xlsx` and print the formula content of all modified cells to verify correctness.
3. Also print the computed values (using data_only or evaluating) if possible, to sanity-check that lookups return numbers and calculations look reasonable.

### Critical constraints
- Do NOT add sheets, macros, VBA, external links, or helper tabs.
- Do NOT modify existing formatting.
- Work ONLY inside the existing `Task` and `Data` sheets.
- All formulas must be Excel spreadsheet formulas (not Python calculations).
- Use openpyxl to write formulas as strings (e.g., cell.value = '=INDEX(...)') so they are preserved as formulas in the .xlsx file.

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