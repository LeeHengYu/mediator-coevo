# Task Instruction

## Task: Populate formulas in /root/data/workbook.xlsx and save to /root/output/result.xlsx

### Preparation
1. Create `/root/output/` directory if it doesn't exist.
2. Install `openpyxl` if not already available (`pip install openpyxl`).
3. Open `/root/data/workbook.xlsx` and inspect both sheets (`Task` and `Data`) to understand the layout before writing any code.
   - On sheet `Task`: print rows 10-50 (columns A through M) to see headers, series codes in column D, years in row 10, and the yellow cell regions.
   - On sheet `Data`: print rows 21-38 (all columns) to understand the lookup source structure.
   - Pay special attention to: exact column letters used, the series codes in column D of Task sheet, the years in row 10, and how Data rows 21-38 are structured (which column has the key, which columns have years).

### Step 1: Populate H12:L17, H19:L24, H26:L31 with lookup formulas
For each cell in the three blocks (rows 12-17, 19-24, 26-31; columns H-L):
- The formula must reference two inputs: the series code from column D of the same row, and the year from row 10 of the same column.
- The lookup source is sheet `Data` rows 21:38.
- Use INDEX/MATCH or one of the other allowed patterns (VLOOKUP+MATCH, HLOOKUP+MATCH, XLOOKUP+MATCH).
- Before writing formulas, determine:
  - Which column in Data contains the series codes (the match key).
  - Whether the years are in a row header or column header in Data.
  - The exact range references needed.

IMPORTANT: When writing formulas with openpyxl, set the cell value to a string starting with `=`. Use absolute references where appropriate to ensure correctness. Use the exact sheet name in cross-sheet references (e.g., `Data!$A$21:$A$38`).

Example pattern (adapt based on actual layout discovered during inspection):
- If Data has series codes in column A and years in row 20 across columns B onward:
  `=INDEX(Data!$B$21:$Z$38, MATCH($D12, Data!$A$21:$A$38, 0), MATCH(H$10, Data!$B$20:$Z$20, 0))`
- Adjust ranges based on actual inspection.

### Step 2: Net budget buffer in H35:L40, then summary stats in H42:L47

For H35:L40 (6 department rows):
- Formula: `(Committed Funding - Operating Spend) / Approved Budget Base * 100`
- The three blocks from Step 1 correspond to three data series. Determine which block is Committed Funding (rows 12-17), Operating Spend (rows 19-24), and Approved Budget Base (rows 26-31) by reading the labels/headers on the Task sheet.
- For each cell, e.g., H35: `=(H12-H19)/H26*100` (adjust row references based on which block is which — the department order in rows 35-40 must match the department order in the lookup blocks).
- IMPORTANT: Verify that the department order in rows 35-40 matches rows 12-17/19-24/26-31. If they differ, adjust references accordingly.

For H42:L47 (summary statistics), column-wise over H35:L40:
- Row 42 (Minimum): `=MIN(H35:H40)` etc.
- Row 43 (Maximum): `=MAX(H35:H40)` etc.
- Row 44 (Median): `=MEDIAN(H35:H40)` etc.
- Row 45 (Mean): `=AVERAGE(H35:H40)` etc.
- Row 46 (25th percentile): `=PERCENTILE(H35:H40, 0.25)` etc.
- Row 47 (75th percentile): `=PERCENTILE(H35:H40, 0.75)` etc.
- Check the labels in column A/B/C/D of rows 42-47 to confirm the correct order of min/max/median/mean/25th/75th.

### Step 3: Weighted mean in H50:L50
For each column H through L:
`=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)`
This uses the Net budget buffer percentages as values and Approved Budget Base as weights.

### Final Steps
1. Save the workbook to `/root/output/result.xlsx` using openpyxl. Do NOT use `data_only=True` when loading — preserve formulas.
2. Reopen the saved file and verify:
   - Cells H12, L17, H19, L24, H26, L31 contain formula strings (start with `=`).
   - Cells H35, L40 contain formula strings.
   - Cells H42, L47 contain formula strings.
   - Cells H50, L50 contain formula strings.
   - The workbook still has exactly 2 sheets: `Task` and `Data`.
3. Print a sample of formulas from each block to confirm correctness.

### Critical Constraints
- Do NOT add any new sheets, macros, VBA, external links, or helper tabs.
- Do NOT change existing formatting.
- Do NOT use `data_only=True` when loading the workbook.
- All formulas must be Excel-compatible spreadsheet formulas, not Python computations.
- Inspect the actual workbook layout FIRST before writing any formulas. The exact row/column positions and series code values must come from the actual file, not assumptions.

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