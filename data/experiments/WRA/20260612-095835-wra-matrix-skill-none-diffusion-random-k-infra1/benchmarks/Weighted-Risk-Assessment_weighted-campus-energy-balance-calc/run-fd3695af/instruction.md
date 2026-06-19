# Task Instruction

Execute the following steps to produce `/root/output/result.xlsx`.

### 0 – Setup
```bash
mkdir -p /root/output
pip install openpyxl
```

### 1 – Inspect the workbook layout
Open `/root/data/workbook.xlsx` with openpyxl (data_only=False so we see formulas, not cached values). Print:
- Sheet names.
- `Task` sheet: rows 10-12 (to see year headers in row 10 and column D series codes), rows 17-20, rows 24-27, rows 31-36, rows 40-48, rows 49-51.
- `Data` sheet: rows 19-40, columns A-Z (to see the layout of the source data, especially rows 21:38).

Pay special attention to:
- Which columns on `Task` hold the series codes (column D).
- Which row on `Task` holds the years (row 10), and in which columns (H through L).
- The exact layout of `Data` rows 21:38 – which column holds the series code key, and whether years run across columns (for HLOOKUP) or down rows (for VLOOKUP).
- The campus names in `Task` and how the three lookup blocks (H12:L17, H19:L24, H26:L31) map to different metrics.
- The labels/structure around rows 35-40 (Net renewable balance), 42-47 (statistics), and 50 (weighted mean).

### 2 – Write formulas with openpyxl
Using `openpyxl` (load workbook, modify, save), populate the cells as follows. Use `ws['cell'].value = '=FORMULA'` syntax. **Do not use data_only mode when loading for writing.**

#### Step 1 – Lookup blocks (H12:L17, H19:L24, H26:L31)
For each cell in these ranges, write an INDEX/MATCH formula that:
- Looks up the series code from column D of the same row on `Task`.
- Looks up the year from row 10 of the same column on `Task`.
- Searches in the `Data` sheet rows 21:38.

The exact formula pattern depends on the Data layout you discover in step 1. A typical pattern if Data has series codes in column A and years across row 20 (header row) would be:
```
=INDEX(Data!$B$21:$XX$38, MATCH($D12, Data!$A$21:$A$38, 0), MATCH(H$10, Data!$B$20:$XX$20, 0))
```
Adjust column/row references to match the actual layout. Key points:
- Anchor the series-code column reference with `$D` and the row with the actual row number (no $ on row so it changes per row within a block).
- Anchor the year row reference with `$10` so it stays fixed as you go down rows.
- Anchor the Data ranges with `$` so they don't shift.

#### Step 2a – Net renewable balance (H35:L40)
Based on the task description, the formula for each campus (6 campuses, 6 rows) and each year (5 columns) is:
```
=(H12 - H19) / H26 * 100
```
where H12 is from the Renewable Generation block, H19 from Grid Consumption, H26 from Baseline Energy Demand. Adjust row references per campus row. Verify which block corresponds to which metric by checking the labels in the Task sheet.

#### Step 2b – Summary statistics (H42:L47)
For each column (H through L), in the 6 rows 42-47, write:
- Row 42: `=MIN(H35:H40)` (or whichever row is MIN based on labels)
- Row 43: `=MAX(H35:H40)`
- Row 44: `=MEDIAN(H35:H40)`
- Row 45: `=AVERAGE(H35:H40)`
- Row 46: `=PERCENTILE(H35:H40, 0.25)`
- Row 47: `=PERCENTILE(H35:H40, 0.75)`

Check the actual labels in column D/E/F/G of rows 42-47 to confirm the order (min, max, median, mean, 25th, 75th).

#### Step 3 – Weighted mean (H50:L50)
For each column:
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```
This uses the Net renewable balance percentages as values and Baseline Energy Demand as weights.

### 3 – Save
Save the workbook to `/root/output/result.xlsx`. Do NOT change formatting, do NOT add sheets.

### 4 – Validate
Reopen the saved file and print the `.value` of every cell in the formula ranges (H12:L17, H19:L24, H26:L31, H35:L40, H42:L47, H50:L50) to confirm they contain formula strings (starting with `=`), not None.

### Critical checks
- If any cell shows `None` after writing, re-inspect and fix immediately. The avoid-artifact from a sibling task shows that `None` values in these cells cause test failure.
- Make sure all `$` anchoring is correct so formulas don't drift when Excel recalculates.
- Ensure the Data sheet range references match exactly what you observed in step 1.
- Do not modify any existing formatting, sheets, or non-yellow cells.

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