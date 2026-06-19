# Task Instruction

Execute the following steps to produce `/root/output/result.xlsx` from `/root/data/workbook.xlsx`.

## Phase 0 – Inspect the workbook

1. Copy the workbook: `cp /root/data/workbook.xlsx /root/output/result.xlsx`
2. Open `/root/output/result.xlsx` with openpyxl (keep `data_only=False` so formulas are preserved).
3. Print the following from sheet `Task`:
   - Row 10 (to see the year headers in columns H–L).
   - Column D rows 12–17, 19–24, 26–31 (to see the series codes for each block).
   - Row 35–40 column D (plant names or codes for the Net-production-slack block).
   - Rows 42–47 column D or E (stat labels: min, max, median, mean, 25th, 75th).
   - Row 50 columns D–G (to confirm "Regional Output Council" label and any weight info).
   - Any existing content/formatting notes in H12, H19, H26, H35, H42, H50.
4. Print from sheet `Data`:
   - Row 20 or 21 headers and a few rows (21–38) to understand the layout: which column holds the series code, which columns/rows hold year-indexed values.
   - Identify the orientation: are years in columns and series codes in rows, or vice versa?

## Phase 1 – Populate lookup formulas in H12:L17, H19:L24, H26:L31

Based on the inspection, write INDEX/MATCH formulas into each cell in those three blocks. Each formula should:
- Use the series code from column D of the same row (e.g., `$D12`).
- Use the year from row 10 of the same column (e.g., `H$10`).
- Look up in `Data!$A$21:$A$38` (or wherever the series codes live) for the row match.
- Look up in `Data!<year header row>` for the column match.
- Use `INDEX(Data!<data range>, MATCH($D12, Data!<series column>, 0), MATCH(H$10, Data!<year row>, 0))`.

Adjust the exact references after inspecting the Data sheet layout. Use absolute references for the data range and lookup arrays, mixed references for the series code (lock column) and year (lock row) so the formula can be written once per block and adjusted per cell.

Write these formulas using openpyxl by assigning formula strings to each cell. Do NOT set values; set formula strings (e.g., `ws['H12'] = '=INDEX(Data!$B$21:$F$38,MATCH($D12,Data!$A$21:$A$38,0),MATCH(H$10,Data!$B$20:$F$20,0))'`).

## Phase 2 – Net production slack in H35:L40

Identify which of the three lookup blocks corresponds to:
- Finished Output (likely H12:L17)
- Scrap And Rework (likely H19:L24)
- Rated Production Capacity (likely H26:L31)

Confirm by checking the labels in column D or nearby cells for rows 12, 19, 26.

For each cell in H35:L40, write a formula:
`=(H12-H19)/H26*100` (adjusting row offsets appropriately for each plant row).

For example, if plant 1 is row 35 and corresponds to row 12 (Finished Output), row 19 (Scrap), row 26 (Capacity):
- `H35 = (H12-H19)/H26*100`
- `H36 = (H13-H20)/H27*100`
- etc.

## Phase 3 – Summary statistics in H42:L47

For each column H through L, write:
- Row 42 (min): `=MIN(H35:H40)`
- Row 43 (max): `=MAX(H35:H40)`
- Row 44 (median): `=MEDIAN(H35:H40)`
- Row 45 (mean): `=AVERAGE(H35:H40)`
- Row 46 (25th percentile): `=PERCENTILE(H35:H40,0.25)`
- Row 47 (75th percentile): `=PERCENTILE(H35:H40,0.75)`

Verify the stat labels in column D/E rows 42–47 to match the correct row to the correct function. Adjust row assignments if the order differs.

## Phase 4 – Weighted mean in H50:L50

For each column H through L:
`=SUMPRODUCT(H35:H40,H26:H31)/SUM(H26:H31)`

This computes the weighted mean of Net Production Slack using Rated Production Capacity as weights.

## Phase 5 – Save and validate

1. Save the workbook with `wb.save('/root/output/result.xlsx')`.
2. Reopen the file and verify:
   - H12 contains a formula string (not None, not a bare value).
   - H35 contains a formula string.
   - H42 contains a formula string.
   - H50 contains a formula string.
3. Print a sample of formula strings from each block to confirm correctness.
4. Run the verifier if available: `cd /root && python -m pytest test_output.py -v` (or whatever test file exists).

## Critical Notes
- Do NOT use `data_only=True` when loading – that strips formulas.
- Do NOT add new sheets, macros, or VBA.
- Preserve all existing formatting (do not clear cells outside the target ranges).
- The failed run from a sibling task showed cells returning None – this happens when formulas aren't written or the file is opened with data_only=True. Avoid this.
- After writing formulas, re-read a few cells to confirm the formula string is stored.

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