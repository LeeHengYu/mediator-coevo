# Task Instruction

You must update an Excel workbook at `/root/data/workbook.xlsx` and save the result to `/root/output/result.xlsx`. Follow these steps precisely:

## Step 0: Inspect the workbook
1. `mkdir -p /root/output`
2. Use `openpyxl` to open `/root/data/workbook.xlsx` and inspect:
   - Sheet `Task`: print rows 1–55 for columns A–L (values AND formulas). Pay special attention to:
     - Column D rows 12–17, 19–24, 26–31 (series codes)
     - Row 10 columns H–L (years)
     - The structure of H12:L17, H19:L24, H26:L31 (these are the yellow cells to fill with formulas)
     - Rows 35–40 (Net container flow), 42–47 (statistics), 50 (weighted mean)
   - Sheet `Data`: print rows 1–40 for all columns to understand the data layout (especially rows 21–38). Identify:
     - Where series codes appear (likely in a column)
     - Where years appear (likely in a row)
     - The exact row/column structure so you can build correct lookup formulas
3. Print cell fills/colors for the yellow cells to confirm the target ranges.

## Step 1: Populate lookup formulas in H12:L17, H19:L24, H26:L31

For each cell in these ranges, write a spreadsheet formula (not a Python-computed value) that looks up data from sheet `Data` rows 21:38. The formula must use TWO inputs:
- The series code from column D of the SAME row on sheet `Task`
- The year from row 10 of the SAME column on sheet `Task`

Use one of these patterns: INDEX/MATCH, VLOOKUP/MATCH, HLOOKUP/MATCH, or XLOOKUP/MATCH.

Based on the Data sheet layout:
- If Data has series codes in a column and years in a row header, INDEX(MATCH, MATCH) is likely best.
- Make sure to use appropriate absolute/relative references so formulas can span the range correctly (lock the lookup column for the series code, lock the lookup row for the year, etc.).
- Use `$` anchoring carefully: the column D reference should have the column locked (`$D12`), the row 10 reference should have the row locked (`H$10`), and the data range references should be fully locked.

Write the formulas as strings into the cells using openpyxl. For example:
```python
ws['H12'] = '=INDEX(Data!$B$21:$S$38,MATCH($D12,Data!$A$21:$A$38,0),MATCH(H$10,Data!$B$20:$S$20,0))'
```
(Adjust the exact ranges after inspecting the Data sheet layout.)

## Step 2: Net container flow and statistics in H35:L40 and H42:L47

For H35:L40, write formulas that compute:
`(Loaded Containers Inbound - Loaded Containers Outbound) / Terminal Throughput Capacity * 100`

The three blocks from Step 1 correspond to three different series groups. Determine which block is Inbound, which is Outbound, and which is Throughput Capacity by examining the series codes and any labels. For example, if H12:L17 is Inbound, H19:L24 is Outbound, H26:L31 is Capacity, then:
```
H35 = =(H12-H19)/H26*100
```
Make sure the six ports align row-by-row across all blocks.

For H42:L47, write column-wise statistical formulas over H35:L40:
- H42: `=MIN(H35:H40)` (minimum)
- H43: `=MAX(H35:H40)` (maximum)
- H44: `=MEDIAN(H35:H40)` (median)
- H45: `=AVERAGE(H35:H40)` (simple mean)
- H46: `=PERCENTILE(H35:H40,0.25)` (25th percentile)
- H47: `=PERCENTILE(H35:H40,0.75)` (75th percentile)

Check the labels in column A/B/C for rows 42–47 to confirm the correct order of statistics. Adjust if the labels indicate a different ordering.

## Step 3: Weighted mean in H50:L50

Write a SUMPRODUCT-based formula for the CPA weighted mean:
```
H50 = =SUMPRODUCT(H35:H40,H26:H31)/SUM(H26:H31)
```
This uses the Net container flow percentages as values and Terminal Throughput Capacity as weights.

## Step 4: Save and verify
1. Save the workbook to `/root/output/result.xlsx` preserving all formatting. When opening the workbook, do NOT use `data_only=True` — open it in formula mode so formulas are preserved.
2. Re-open the saved file and print all formula cells in the target ranges to verify they are correctly written.
3. Spot-check a few formulas for correctness of references and anchoring.

## Critical constraints
- Do NOT add new sheets, macros, VBA, external links, or helper tabs.
- Do NOT change existing formatting (fonts, colors, borders, etc.).
- All values in the yellow cells and calculation cells MUST be Excel formulas, not hardcoded Python-computed values.
- Use `openpyxl` for all Excel operations.
- The inspection in Step 0 is essential — do not skip it. The exact column/row references for the Data sheet lookup ranges depend on the actual layout.

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
Task metadata: author_email=catpaw@meituan.com, author_name=CatPaw Task Engineer, category=spreadsheet-formula-reuse, difficulty=medium, tags=[excel, formulas, lookup, statistics, weighted-mean].
Verifier config: timeout_sec=600.0.