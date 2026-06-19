# Task Instruction

Execute the following steps precisely to complete the hospital capacity workbook task.

## Step 0: Inspect the workbook
1. Copy `/root/data/workbook.xlsx` to `/root/output/result.xlsx`.
2. Open `/root/output/result.xlsx` with openpyxl (with `data_only=False` so you can read and write formulas).
3. Read sheet `Task`: print rows 1–55, columns A–M, to understand the layout. Pay special attention to:
   - Column D rows 12–17, 19–24, 26–31 (series codes for each row)
   - Row 10 columns H–L (years)
   - Rows 35–40 (Net capacity headroom cluster rows)
   - Rows 42–47 (summary statistics labels)
   - Row 50 (weighted mean row)
4. Read sheet `Data`: print rows 1–40, all populated columns, to understand the data layout. Identify:
   - What is in row 1 (headers?), what columns hold series codes, what rows 21–38 contain, and how years appear (as column headers in some row?).
5. Print your findings before proceeding.

## Step 1: Populate lookup formulas in H12:L17, H19:L24, H26:L31

For each cell in these three blocks, write a formula that looks up the value from sheet `Data` rows 21:38. The formula must use two inputs:
- The series code from column D of the current row on sheet `Task`
- The year from row 10 of the current column on sheet `Task`

Use INDEX/MATCH (most reliable in openpyxl). The exact formula pattern depends on the Data sheet layout you discovered in Step 0. Typical pattern if Data has series codes in column A and years across a header row (e.g., row 20):

```
=INDEX(Data!$B$21:$XX$38, MATCH($D12, Data!$A$21:$A$38, 0), MATCH(H$10, Data!$B$20:$XX$20, 0))
```

Adjust the column/row references based on actual Data sheet structure. Key rules:
- Lock the data range and lookup arrays with `$` appropriately.
- The series code reference should lock the column (`$D12`) so it stays in column D when copied across.
- The year reference should lock the row (`H$10`) so it stays in row 10 when copied down.
- Verify the formula works for at least one cell by checking it manually against the Data sheet values.

Write all 54 formulas (18 rows × 5 columns = 90 cells total across three blocks: rows 12-17, 19-24, 26-31, columns H-L).

## Step 2: Net capacity headroom and summary statistics

### H35:L40 — Net capacity headroom
For each of the 6 hospital clusters (rows 35–40) and each year (columns H–L), write a formula:
```
=(H12 - H19) / H26 * 100
```
where:
- H12 corresponds to Available Care Slots (rows 12–17)
- H19 corresponds to Occupied Care Slots (rows 19–24)  
- H26 corresponds to Staffed Bed Capacity (rows 26–31)

So row 35 uses rows 12, 19, 26; row 36 uses rows 13, 20, 27; etc. Adjust references for each row and column.

### H42:L47 — Summary statistics
For each column H through L:
- Row 42 (minimum): `=MIN(H35:H40)`
- Row 43 (maximum): `=MAX(H35:H40)`
- Row 44 (median): `=MEDIAN(H35:H40)`
- Row 45 (mean): `=AVERAGE(H35:H40)`
- Row 46 (25th percentile): `=PERCENTILE(H35:H40, 0.25)`  (or `PERCENTILE.INC`)
- Row 47 (75th percentile): `=PERCENTILE(H35:H40, 0.75)`  (or `PERCENTILE.INC`)

Check the labels in column D/E/F/G of rows 42–47 to confirm which row gets which statistic, and adjust accordingly.

## Step 3: Weighted mean in H50:L50
For each column H through L:
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```
This computes the weighted mean of Net capacity headroom percentages weighted by Staffed Bed Capacity.

## Step 4: Save and validate
1. Save the workbook to `/root/output/result.xlsx` preserving all formatting.
2. Re-open the file and verify:
   - Cells H12:L31 contain formulas (not hardcoded values)
   - Cells H35:L40 contain formulas
   - Cells H42:L47 contain formulas
   - Cells H50:L50 contain formulas
   - No new sheets were added
   - Print a sample of formulas to confirm correctness
3. Optionally open with data_only=True to check computed values make sense.

## Critical constraints
- Do NOT add new sheets, macros, VBA, external links, or helper tabs.
- Do NOT alter existing formatting.
- Use openpyxl to write formulas as strings (they must start with `=`).
- The lookup formulas MUST use one of: VLOOKUP+MATCH, HLOOKUP+MATCH, XLOOKUP+MATCH, or INDEX+MATCH.
- Adapt all cell references to the ACTUAL layout you observe in Step 0. Do not assume — inspect first.

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