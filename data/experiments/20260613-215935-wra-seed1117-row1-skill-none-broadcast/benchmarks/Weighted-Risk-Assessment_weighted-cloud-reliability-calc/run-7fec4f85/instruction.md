# Task Instruction

## Task: Update /root/data/workbook.xlsx with formulas and save to /root/output/result.xlsx

### Phase 0: Inspect the workbook
1. Create `/root/output/` directory if it doesn't exist.
2. Use `openpyxl` to open `/root/data/workbook.xlsx` and inspect:
   - Sheet `Task`: Print the contents of rows 10-50, columns D through L (values AND any existing formulas). Pay special attention to:
     - Row 10 (years)
     - Column D rows 12-17, 19-24, 26-31 (series codes)
     - The yellow cell ranges H12:L17, H19:L24, H26:L31 (currently empty or with placeholders)
     - Rows 35-40 (region labels for Net reliability gap)
     - Rows 42-47 (min, max, median, mean, 25th, 75th percentile labels)
     - Row 50 (GCM weighted mean)
   - Sheet `Data`: Print rows 21-38 to understand the data layout (column headers, row structure, where series codes and years appear).
3. Print the exact cell values for row 10 columns H-L on Task sheet (the year headers).
4. Print column D values for rows 12-17, 19-24, 26-31 on Task sheet (the series codes).
5. Print the Data sheet structure: row 1 headers, and specifically rows 21-38 with their column layout. Identify which row contains headers and which column contains series codes, and where year-based data starts.

### Phase 1: Populate lookup formulas in H12:L17, H19:L24, H26:L31

For each cell in these ranges, write a spreadsheet formula (not a Python-computed value) using INDEX/MATCH (or VLOOKUP with MATCH, etc.). The formula must:
- Use the series code from column D of the SAME row on the Task sheet
- Use the year from row 10 of the SAME column on the Task sheet  
- Look up the value from sheet `Data` rows 21:38

Based on your inspection of the Data sheet layout, construct the correct formula pattern. For example, if Data has series codes in column A and years across columns as headers in row 20 (or wherever), a typical INDEX/MATCH/MATCH formula would be:
`=INDEX(Data!$B$21:$XX$38, MATCH($D12, Data!$A$21:$A$38, 0), MATCH(H$10, Data!$B$20:$XX$20, 0))`

Adjust the exact ranges based on what you find in the Data sheet. The key contract: use the actual column/row ranges from Data rows 21:38.

Write these as string formulas into the cells using openpyxl (e.g., `ws['H12'] = '=INDEX(...)'`).

### Phase 2: Net reliability gap in H35:L40

For each cell in H35:L40, write a formula:
`=(H12 - H19) / H26 * 100`
(adjusting row references for each of the 6 regions)

Specifically:
- H35 = (H12 - H19) / H26 * 100  (region 1)
- H36 = (H13 - H20) / H27 * 100  (region 2)
- H37 = (H14 - H21) / H28 * 100  (region 3)
- H38 = (H15 - H22) / H29 * 100  (region 4)
- H39 = (H16 - H23) / H30 * 100  (region 5)
- H40 = (H17 - H24) / H31 * 100  (region 6)

And similarly for columns I through L.

IMPORTANT: Verify that rows 12-17 correspond to "Successful API Requests", rows 19-24 to "Failed API Requests", and rows 26-31 to "Compute Capacity" by checking the Task sheet labels. Adjust if the mapping is different.

### Phase 3: Summary statistics in H42:L47

For each column H through L, write formulas:
- Row 42 (MIN): `=MIN(H35:H40)`
- Row 43 (MAX): `=MAX(H35:H40)`
- Row 44 (MEDIAN): `=MEDIAN(H35:H40)`
- Row 45 (MEAN): `=AVERAGE(H35:H40)`
- Row 46 (25th percentile): `=PERCENTILE(H35:H40, 0.25)`
- Row 47 (75th percentile): `=PERCENTILE(H35:H40, 0.75)`

Verify the label column to confirm which row is which statistic. Adjust row assignments accordingly.

### Phase 4: Weighted mean in H50:L50

For each column H through L:
`=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)`

This uses the Net reliability gap values (H35:H40) as values and Compute Capacity (H26:H31) as weights.

### Phase 5: Save and validate
1. Save the workbook to `/root/output/result.xlsx`.
2. Reopen the saved file and verify:
   - Cells H12:L17, H19:L24, H26:L31 contain formula strings (not None/empty).
   - Cells H35:L40 contain formula strings.
   - Cells H42:L47 contain formula strings.
   - Cells H50:L50 contain formula strings.
   - No new sheets were added.
   - Print a sample of the formulas to confirm correctness.
3. Do NOT add any macros, VBA, external links, or helper sheets.
4. Do NOT change any existing formatting, values, or structure outside the specified cells.

### Critical Notes
- All formulas must be Excel formulas written as strings starting with '=' in openpyxl.
- Do NOT compute values in Python and write numbers. Write FORMULAS.
- Use `openpyxl` to read and write. Open with `data_only=False` to preserve existing formulas.
- When writing formulas, ensure cell references use the correct absolute/relative references ($ signs) so the lookup ranges stay fixed but the row/column inputs vary appropriately.
- Before writing any formula, confirm the exact layout by printing the relevant cells.

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