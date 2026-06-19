# Task Instruction

Execute the following steps to produce /root/output/result.xlsx from /root/data/workbook.xlsx.

## Phase 0 – Investigation
1. `mkdir -p /root/output`
2. Open /root/data/workbook.xlsx with openpyxl (data_only=False) and inspect:
   - Sheet names (expect 'Task' and 'Data').
   - Task sheet: print cells D12:D17, D19:D24, D26:D31 (series codes), H10:L10 (years), H35:H40 labels or row structure, H42:H47 labels, H50 label.
   - Data sheet: print row 21 (header row) and rows 22–39 (or 21–38 as stated). Identify the column layout: which column holds the series code, which columns hold year data. Print columns A–O of rows 21–39.
   - Note the exact column letters and row numbers so formulas reference the correct ranges.

## Phase 1 – Lookup formulas (H12:L17, H19:L24, H26:L31)
For every cell in these three 6×5 blocks, write an INDEX/MATCH/MATCH formula of the form:

```
=INDEX(Data!$C$22:$O$39, MATCH(Task!$D{row}, Data!$B$22:$B$39, 0), MATCH(Task!{col}$10, Data!$C$21:$O$21, 0))
```

Adjust the exact references based on what you found in Phase 0:
- The first MATCH looks up the series code from column D of the current Task row against the series-code column in Data (likely column B, rows 22–39 or 21–38).
- The second MATCH looks up the year from row 10 of the current Task column against the header row in Data (likely row 21, columns C–O).
- The INDEX range is the rectangular data body on Data (excluding the series-code column and header row).

Make sure to use absolute references for the Data ranges and mixed references ($D for the series code column, $10 for the year row) so the formula can be placed in each cell correctly.

## Phase 2 – Net production slack (H35:L40)
These 6 rows correspond to the six plants. The formula for each cell is:

```
=(H{finished_row} - H{scrap_row}) / H{capacity_row} * 100
```

where:
- finished_row is in the first lookup block (H12:L17, rows 12–17)
- scrap_row is in the second lookup block (H19:L24, rows 19–24)
- capacity_row is in the third lookup block (H26:L31, rows 26–31)

So for plant i (0-based), the row offsets are: finished = 12+i, scrap = 19+i, capacity = 26+i, and the target row = 35+i.

For each cell at (35+i, col) where col is H through L:
```
=({col}{12+i} - {col}{19+i}) / {col}{26+i} * 100
```

## Phase 3 – Summary statistics (H42:L47)
For each column H through L:
- Row 42: `=MIN(H35:H40)` (adjust column letter)
- Row 43: `=MAX(H35:H40)`
- Row 44: `=MEDIAN(H35:H40)`
- Row 45: `=AVERAGE(H35:H40)`
- Row 46: `=PERCENTILE(H35:H40,0.25)`
- Row 47: `=PERCENTILE(H35:H40,0.75)`

## Phase 4 – Weighted mean (H50:L50)
For each column H through L:
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```
(Adjust column letters accordingly.)

## Phase 5 – Save and verify
1. Save the workbook to /root/output/result.xlsx.
2. Reopen the saved file with openpyxl (data_only=False).
3. For each formula block, print a sample of cells to confirm they contain formula strings (starting with '='), not None or raw values.
4. Confirm sheet names are exactly ['Task', 'Data'] (no extra sheets).
5. Print the formulas in H12, L17, H19, L24, H26, L31, H35, L40, H42, H47, H50, L50 to verify correctness.

## Important notes
- Do NOT use data_only=True when writing; open with data_only=False.
- Do NOT add any new sheets, macros, or named ranges.
- Do NOT alter any existing formatting, values, or structure outside the specified cells.
- If the Data sheet header row or series code column differs from the assumed B/C layout, adjust all formulas accordingly based on Phase 0 findings.
- The avoid-recheck artifact warns that cells can end up as None if formulas aren't written correctly. Double-check every formula is a string starting with '='.

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