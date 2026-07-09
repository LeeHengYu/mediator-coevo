# Task Instruction

## Task: Populate formulas in /root/data/workbook.xlsx and save to /root/output/result.xlsx

### Phase 0: Inspect the workbook
1. Create `/root/output/` directory if it doesn't exist.
2. Use `openpyxl` to open `/root/data/workbook.xlsx` and inspect:
   - **Sheet `Task`**: Print cells in row 10 (H10:L10) to see the years. Print column D rows 12-31 to see series codes. Print cells H35:H40 labels or column D/E rows 35-40 to understand the region layout. Print rows 42-47 column D/E/F/G to see what statistics are expected. Print row 50 to see the GCM label.
   - **Sheet `Data`**: Print rows 21-38 entirely (all columns up to ~O or beyond) to see the data layout — identify which row is the header row, which column has series codes, and which columns have year data. Pay special attention to row 21 (likely a header) and the structure.
3. Print the exact cell contents so you know the precise column/row references for the Data sheet lookup range.

### Phase 1: Populate lookup formulas in H12:L17, H19:L24, H26:L31

Using `openpyxl`, write **string formulas** (not computed values) into each cell. Use the `INDEX/MATCH` pattern for 2D lookups:

```
=INDEX(Data!$A$21:$O$38, MATCH($D12, Data!$A$21:$A$38, 0), MATCH(H$10, Data!$A$21:$O$21, 0))
```

**IMPORTANT**: Before writing formulas, verify from your Phase 0 inspection:
- The exact column range in Data sheet (it might be A:O or A:P or different — use what you actually see).
- The exact row that contains headers/years in the Data sheet (likely row 21).
- The exact column that contains series codes (likely column A).
- Adjust the formula references accordingly.

For each block:
- **H12:L17** — 6 rows × 5 columns. Row references: rows 12-17, column D has the series code, row 10 has the year.
- **H19:L24** — 6 rows × 5 columns. Row references: rows 19-24.
- **H26:L31** — 6 rows × 5 columns. Row references: rows 26-31.

Use `$D{row}` (absolute column, relative row) for the series code and `{col}$10` (relative column, absolute row) for the year. Make sure the Data range references are fully absolute with `$`.

### Phase 2: Net reliability gap formulas in H35:L40

The formula for Net reliability gap is:
```
(Successful API Requests - Failed API Requests) / Compute Capacity * 100
```

From Phase 0 inspection, identify which rows in the Task sheet correspond to:
- **Successful API Requests** block (likely H12:L17)
- **Failed API Requests** block (likely H19:L24)  
- **Compute Capacity** block (likely H26:L31)

For cell H35, the formula would be something like:
```
=(H12-H19)/H26*100
```
Adjust row offsets so that row 35 maps to the first region (same as row 12, 19, 26), row 36 maps to the second region (row 13, 20, 27), etc. through row 40.

Write these as string formulas in openpyxl.

### Phase 3: Summary statistics in H42:L47

For each column H through L, compute column-wise statistics over H35:L40 (the 6 Net reliability gap values):
- **Row 42**: Minimum → `=MIN(H35:H40)`
- **Row 43**: Maximum → `=MAX(H35:H40)`
- **Row 44**: Median → `=MEDIAN(H35:H40)`
- **Row 45**: Simple mean → `=AVERAGE(H35:H40)`
- **Row 46**: 25th percentile → `=PERCENTILE(H35:H40, 0.25)`
- **Row 47**: 75th percentile → `=PERCENTILE(H35:H40, 0.75)`

**IMPORTANT**: Verify from Phase 0 which row corresponds to which statistic by reading column D/E/F/G of rows 42-47. The order above is a guess — match the actual labels.

### Phase 4: Weighted mean in H50:L50

Use SUMPRODUCT with the Step 2 percentages (H35:H40 for column H) as values and Compute Capacity (H26:H31 for column H) as weights:
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```

Write this formula for each of H50 through L50.

### Phase 5: Save and verify
1. Save the workbook to `/root/output/result.xlsx`.
2. Re-open the saved file and verify:
   - Cells H12, L17, H19, L24, H26, L31 contain formula strings (start with `=`).
   - Cells H35, L40 contain formula strings.
   - Cells H42, L47 contain formula strings.
   - Cells H50, L50 contain formula strings.
   - No new sheets were added.
   - Print a sample of formulas to confirm correctness.

### Critical Notes
- Only write formulas as strings (e.g., `ws['H12'] = '=INDEX(...)'`). Do NOT use `data_only` mode.
- Do NOT change any formatting, do NOT add sheets, macros, VBA, or helper tabs.
- The Phase 0 inspection is essential — do NOT skip it. The exact references in the formulas depend on what you find in the actual workbook.
- If the PERCENTILE function name doesn't match the labels (e.g., PERCENTILE.INC vs PERCENTILE), use the standard `PERCENTILE` unless the labels indicate otherwise.
- Use `MATCH(...,0)` for exact match in all MATCH functions.

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