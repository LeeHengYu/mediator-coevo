# Task Instruction

Execute the following steps precisely to complete the hospital capacity workbook task.

## Step 0: Inspect the workbook
1. Copy `/root/data/workbook.xlsx` to `/root/output/result.xlsx`.
2. Open `/root/output/result.xlsx` using openpyxl (with `data_only=False` so formulas are preserved).
3. Print the contents of sheet `Task` rows 1–55 and columns A–M. Pay special attention to:
   - Column D rows 12–17, 19–24, 26–31 (series codes)
   - Row 10 columns H–L (years)
   - Rows 35–40 (Net capacity headroom area)
   - Rows 42–47 (summary statistics area)
   - Row 50 (weighted mean area)
4. Print sheet `Data` rows 1–40, focusing on rows 21–38 to understand the data layout: which row contains which series code, which columns contain which years, and the exact structure.
5. Print all cell values so you understand the exact row/column mapping before writing any formulas.

## Step 1: Populate lookup formulas in H12:L17, H19:L24, H26:L31

For each cell in the three blocks (rows 12–17, 19–24, 26–31) and columns H–L:
- The formula must look up the value from sheet `Data` rows 21:38 using:
  - The series code from column D of the current row on sheet `Task`
  - The year from row 10 of the current column on sheet `Task`
- Use an INDEX/MATCH pattern (or VLOOKUP with MATCH, etc.) that references `Data!$A$21:$A$38` (or wherever the series codes live) and the year header row on `Data`.
- IMPORTANT: Before writing formulas, confirm exactly which column on `Data` holds the series codes and which row holds the year headers. Adjust all references accordingly.
- Example pattern (adjust references after inspection):
  `=INDEX(Data!$B$21:$Z$38, MATCH($D12, Data!$A$21:$A$38, 0), MATCH(H$10, Data!$B$20:$Z$20, 0))`
  Adapt the exact ranges based on what you find in Step 0.

## Step 2: Net capacity headroom (H35:L40) and summary statistics (H42:L47)

For H35:L40 (6 rows corresponding to 6 hospital clusters):
- Formula: `=(AvailableCareSlots - OccupiedCareSlots) / StaffedBedCapacity * 100`
- Identify which of the three blocks (H12:L17, H19:L24, H26:L31) corresponds to Available Care Slots, Occupied Care Slots, and Staffed Bed Capacity by reading the labels in the Task sheet (likely in column A or nearby). Map them correctly.
- For example, if rows 12–17 are Available Care Slots, rows 19–24 are Occupied Care Slots, and rows 26–31 are Staffed Bed Capacity, then for cell H35: `=(H12-H19)/H26*100`, H36: `=(H13-H20)/H27*100`, etc.
- Adjust row references based on actual labels found in Step 0.

For H42:L47 (column-wise summary statistics of H35:L40):
- Row 42: `=MIN(H35:H40)` (or whichever row is MIN per the labels)
- Row 43: `=MAX(H35:H40)`
- Row 44: `=MEDIAN(H35:H40)`
- Row 45: `=AVERAGE(H35:H40)` (simple mean)
- Row 46: `=PERCENTILE(H35:H40, 0.25)` (25th percentile)
- Row 47: `=PERCENTILE(H35:H40, 0.75)` (75th percentile)
- Match the exact row to the exact label (MIN, MAX, MEDIAN, MEAN, 25th, 75th) as shown on the Task sheet. Read the labels in column A/B/C/D for rows 42–47 before assigning.

## Step 3: Weighted mean in H50:L50

For each column H–L in row 50:
- `=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)`
- This computes the weighted mean of the Net capacity headroom percentages (Step 2) weighted by Staffed Bed Capacity.
- Alternatively, if the task says use SUMPRODUCT: `=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)` is the standard weighted-mean formula using SUMPRODUCT.

## Step 4: Save and validate
1. Save the workbook to `/root/output/result.xlsx`.
2. Re-open the file and print all formula cells to confirm they are correctly placed.
3. Verify no new sheets were added, no macros, no external links.
4. Verify the yellow-cell ranges all contain formulas (not hardcoded values).

## Critical constraints
- Do NOT add sheets, macros, VBA, external links, or helper tabs.
- Do NOT change existing formatting.
- All formulas must be Excel-compatible spreadsheet formulas written as strings (e.g., `ws['H12'] = '=INDEX(...)'`).
- Use openpyxl to write formulas. Do NOT use xlsxwriter (it cannot preserve existing workbooks).
- After every formula-writing step, re-read a sample cell to confirm the formula string was stored correctly.

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