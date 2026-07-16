# Task Instruction

## Task: Populate formulas in /root/data/workbook.xlsx and save to /root/output/result.xlsx

### Pre-work: Inspect the workbook structure

1. Create the output directory: `mkdir -p /root/output`
2. Use `openpyxl` (Python) to open `/root/data/workbook.xlsx` and inspect:
   - **Sheet `Task`**:
     - Read row 10 (especially columns H through L) to see the year headers. Print their values AND types (string vs int).
     - Read column D for rows 12–17, 19–24, 26–31 to see the series codes. Print their values AND types.
     - Read rows 35–40 column D (or B/C) to see the hospital names for Net patient flow.
     - Read rows 42–47 column D (or nearby) to see what statistics are expected (min, max, median, mean, 25th, 75th percentile).
     - Read row 50 to see the weighted mean label.
     - Check what's in H12:L31 currently (are they empty/yellow?).
   - **Sheet `Data`**:
     - Read rows 21–38 fully. Print all cell values for these rows across all populated columns. Pay special attention to:
       - Which column contains the series codes (lookup keys)
       - Which row/column contains year headers
       - The exact format of series codes and years (strings vs numbers, any spaces)
     - Identify the data layout: Is it vertical (series codes in a column, years across columns) or horizontal?

3. Print all findings clearly before proceeding to formula writing.

### Step 1: Write lookup formulas in H12:L17, H19:L24, H26:L31

Based on the inspection, write formulas using INDEX/MATCH (or VLOOKUP with MATCH, etc.) that:
- Use the series code from column D of the current row as one lookup key
- Use the year from row 10 (same column) as the second lookup key
- Look up values from sheet `Data` rows 21:38

**Critical**: Match the exact data types and formats. If years in row 10 of Task are numbers but in Data they are strings (or vice versa), wrap with `TEXT()` or `VALUE()` as needed. If series codes have different formatting, handle that too. Use absolute references (`$`) appropriately so formulas can span the range.

For INDEX/MATCH pattern, a typical formula might be:
`=INDEX(Data!$B$21:$Z$38, MATCH($D12, Data!$A$21:$A$38, 0), MATCH(H$10, Data!$B$20:$Z$20, 0))`
But adjust column/row references based on actual Data sheet layout discovered in inspection.

Write formulas into all cells in:
- H12:L17 (first block)
- H19:L24 (second block)  
- H26:L31 (third block)

### Step 2: Net patient flow formulas in H35:L40 and statistics in H42:L47

For H35:L40, the formula for each hospital/year cell is:
`=(Admissions - Discharges) / Effective_Bed_Capacity * 100`

where:
- Admissions come from H12:L17 (first block)
- Discharges come from H19:L24 (second block)
- Effective Bed Capacity comes from H26:L31 (third block)

So for cell H35: `=(H12-H19)/H26*100` and similarly for the rest of the 6×5 grid.

**Verify**: Check that the row mapping is correct (row 12↔row 19↔row 26↔row 35 for the same hospital, etc.).

For H42:L47, compute column-wise statistics over H35:L40:
- Row 42 (minimum): `=MIN(H35:H40)` across each column
- Row 43 (maximum): `=MAX(H35:H40)`
- Row 44 (median): `=MEDIAN(H35:H40)`
- Row 45 (simple mean): `=AVERAGE(H35:H40)`
- Row 46 (25th percentile): `=PERCENTILE(H35:H40, 0.25)` or `=PERCENTILE.INC(H35:H40, 0.25)`
- Row 47 (75th percentile): `=PERCENTILE(H35:H40, 0.75)` or `=PERCENTILE.INC(H35:H40, 0.75)`

**Check**: Verify which row maps to which statistic by reading the labels in the Task sheet.

### Step 3: Weighted mean in H50:L50

For each column (H through L):
`=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)`

This uses Net patient flow percentages as values and Effective Bed Capacity as weights.

### Saving

- Save the workbook to `/root/output/result.xlsx`
- Do NOT add new sheets, macros, VBA, external links, or helper tabs.
- Preserve all existing formatting.

### Validation

After saving, re-open `/root/output/result.xlsx` and:
1. Verify formulas exist in H12:L17, H19:L24, H26:L31 (should be formula strings, not plain values)
2. Verify formulas exist in H35:L40, H42:L47, H50:L50
3. Spot-check a few cells by evaluating them with openpyxl's data_only or by printing the formula strings to confirm they reference the correct ranges
4. Confirm no new sheets were added
5. Print a summary of what was done

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