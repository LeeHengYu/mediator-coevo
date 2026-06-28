# Task Instruction

## Task: Update /root/data/workbook.xlsx with formulas and save to /root/output/result.xlsx

### Phase 0: Inspect the workbook
1. `mkdir -p /root/output`
2. Use Python with openpyxl to open `/root/data/workbook.xlsx` and inspect:
   - Sheet names (confirm `Task` and `Data` exist).
   - On sheet `Task`: read row 10 (the year headers in columns H–L), column D rows 12–17, 19–24, 26–31 to see the series codes. Read row 35–40 column D for plant names/codes. Read rows 42–47 column D or G for the stat labels (min, max, median, mean, 25th, 75th). Read row 50 for the "Regional Output Council" label. Check what's in H26:L31 (Rated Production Capacity block). Check column D or nearby columns for any labels.
   - On sheet `Data`: read rows 21–38 completely to understand the data layout — which row has which series code, which columns have which years, and the orientation (is data in rows or columns?).
   - Print all of this so you understand the exact structure before writing any formulas.

### Phase 1: Populate H12:L17, H19:L24, H26:L31 with lookup formulas

For each cell in these three blocks, write a spreadsheet **formula string** (not a computed value). The formula must use one of these patterns:
- `INDEX(MATCH, MATCH)` — typically the most reliable
- `VLOOKUP` with `MATCH`
- `HLOOKUP` with `MATCH`
- `XLOOKUP` with `MATCH`

The two lookup keys are:
- The series code from column D of the current row on sheet `Task`
- The year from row 10 of the current column on sheet `Task`

The lookup range is on sheet `Data` rows 21:38.

**Important formula construction notes:**
- Inspect the Data sheet carefully to determine orientation. If data is arranged with series codes in one column and years across columns, use INDEX/MATCH/MATCH or VLOOKUP/MATCH accordingly.
- Use absolute references (with `$`) for the data range and the lookup arrays so formulas can be placed in any cell in the block.
- Reference the series code cell (column D, same row) and the year cell (row 10, same column) with appropriate relative/absolute references.
- When writing formulas in openpyxl, set `cell.value = '=FORMULA...'` as a string starting with `=`.
- Use the actual sheet name for cross-sheet references, e.g., `Data!$A$21:$A$38` — adjust column letters and row numbers based on what you find in Phase 0.

### Phase 2: Net production slack in H35:L40

For each cell in H35:L40, write a formula:
```
=(corresponding_Finished_Output_cell - corresponding_Scrap_And_Rework_cell) / corresponding_Rated_Production_Capacity_cell * 100
```

Determine which of the three blocks (H12:L17, H19:L24, H26:L31) corresponds to "Finished Output", "Scrap And Rework", and "Rated Production Capacity" by reading the labels on the Task sheet (likely in column D or a header row above each block like rows 11, 18, 25). Then reference the correct cells. For example, if H12:L17 is Finished Output, H19:L24 is Scrap And Rework, and H26:L31 is Rated Production Capacity, then H35 = `=(H12-H19)/H26*100`.

### Phase 3: Summary statistics in H42:L47

For each column H through L, in rows 42–47, write formulas for column-wise statistics of the corresponding H35:L40 column. Map the six rows to the correct stat based on what labels you find in column D/G rows 42–47. Use:
- `=MIN(H35:H40)` for minimum
- `=MAX(H35:H40)` for maximum  
- `=MEDIAN(H35:H40)` for median
- `=AVERAGE(H35:H40)` for simple mean
- `=PERCENTILE(H35:H40, 0.25)` for 25th percentile (or `PERCENTILE.INC`)
- `=PERCENTILE(H35:H40, 0.75)` for 75th percentile (or `PERCENTILE.INC`)

Match each formula to the correct row based on the label.

### Phase 4: Weighted mean in H50:L50

For each column H through L in row 50, write a SUMPRODUCT formula:
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```
This computes the weighted mean of the Net production slack percentages (H35:H40) weighted by Rated Production Capacity (H26:L31).

### Phase 5: Save and validate
1. Save the workbook to `/root/output/result.xlsx` using openpyxl. Make sure to NOT use `data_only=True` when loading (so formulas are preserved). Keep existing formatting by not modifying styles, number formats, etc.
2. Re-open `/root/output/result.xlsx` and verify:
   - All cells in H12:L17, H19:L24, H26:L31 contain formula strings (start with `=`).
   - All cells in H35:L40 contain formula strings.
   - All cells in H42:L47 contain formula strings.
   - All cells in H50:L50 contain formula strings.
   - No new sheets were added.
   - Print a sample of formulas from each block to confirm correctness.

### Critical constraints
- Do NOT add new sheets, macros, VBA, external links, or helper tabs.
- Do NOT change existing formatting (fonts, colors, borders, number formats).
- Only write formula strings into cells; do not write computed numeric values.
- Use openpyxl for all Excel manipulation.
- If the Data sheet has merged cells or unusual structure, adapt the formula references accordingly based on what you observe in Phase 0.

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