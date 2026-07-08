# Task Instruction

## Task: Update `/root/data/workbook.xlsx` with formulas and save to `/root/output/result.xlsx`

### Phase 0: Inspect the workbook
1. Create `/root/output/` directory if it doesn't exist.
2. Use `openpyxl` to open `/root/data/workbook.xlsx` and inspect:
   - Sheet `Task`: Print the contents of rows 10-50, columns D through L (values and any existing formulas). Pay special attention to:
     - Row 10 (years)
     - Column D rows 12-17, 19-24, 26-31 (series codes)
     - Column D rows 35-40 (port names/identifiers)
     - Any existing content in H42:L47, H50:L50
     - The yellow-highlighted regions (H12:L17, H19:L24, H26:L31)
   - Sheet `Data`: Print rows 21-38 to understand the data layout — identify which row contains headers, how series codes map, where years appear, and the orientation (rows vs columns).
3. Print the exact cell values for Data row 21 across all columns to understand the header structure, and rows 22-38 to see the data.

### Phase 1: Populate lookup formulas in H12:L17, H19:L24, H26:L31

Based on your inspection of the Data sheet layout:
- Each formula cell is at the intersection of a series code (column D of that row on Task sheet) and a year (row 10 on Task sheet).
- The lookup must find the correct row in `Data!$21:$38` matching the series code, and the correct column matching the year.
- Use `INDEX/MATCH` pattern. The exact references depend on the Data sheet layout you discover.

Typical pattern (adjust based on actual layout):
```
=INDEX(Data!<data_range>, MATCH($D12, Data!<series_code_column>, 0), MATCH(H$10, Data!<year_header_row>, 0))
```

IMPORTANT: Use `$D12` (column-absolute, row-relative) for the series code reference so it locks to column D but moves with the row. Use `H$10` (row-absolute, column-relative) for the year so it locks to row 10 but moves with the column.

Write these formulas into every cell in the three blocks (H12:L17, H19:L24, H26:L31) — that's 6 rows × 5 columns × 3 blocks = 90 formula cells.

Use openpyxl to write formula strings (not computed values). Make sure you write the formulas as strings starting with `=`.

### Phase 2: Net container flow in H35:L40

Based on the task description:
- `Loaded Containers Inbound` values are in H12:L17
- `Loaded Containers Outbound` values are in H19:L24  
- `Terminal Throughput Capacity` values are in H26:L31

Verify this mapping by checking the block labels/headers near rows 11, 18, 25 on the Task sheet.

For each cell in H35:L40, the formula should be:
```
=(H12-H19)/H26*100
```
(with appropriate row offsets for each of the 6 ports)

So H35 = `=(H12-H19)/H26*100`, H36 = `=(H13-H20)/H27*100`, etc.
And I35 = `=(I12-I19)/I26*100`, etc.

Verify the row mapping: row 35 corresponds to row 12, 19, 26; row 36 to 13, 20, 27; etc.

### Phase 3: Summary statistics in H42:L47

For each column (H through L), calculate column-wise statistics over the 6 net-flow values (rows 35:40):
- Row 42: Minimum → `=MIN(H35:H40)`
- Row 43: Maximum → `=MAX(H35:H40)`
- Row 44: Median → `=MEDIAN(H35:H40)`
- Row 45: Simple mean → `=AVERAGE(H35:H40)`
- Row 46: 25th percentile → `=PERCENTILE(H35:H40,0.25)`
- Row 47: 75th percentile → `=PERCENTILE(H35:H40,0.75)`

IMPORTANT: Verify the actual labels in column D (or nearby) for rows 42-47 to confirm the correct order of min/max/median/mean/25th/75th. Adjust the row assignments to match whatever labels are already present.

### Phase 4: Weighted mean in H50:L50

For each column (H through L):
```
=SUMPRODUCT(H35:H40,H26:H31)/SUM(H26:H31)
```
This computes the weighted mean of net container flow percentages using Terminal Throughput Capacity as weights.

### Phase 5: Save and validate
1. Save the workbook to `/root/output/result.xlsx`.
2. Reopen the saved file and verify:
   - All formula cells contain formula strings (start with `=`), not hardcoded values.
   - No new sheets were added.
   - Spot-check a few formulas from each block to confirm correct cell references.
   - Print formulas from representative cells: H12, L17, H19, L24, H26, L31, H35, L40, H42, H47, H50, L50.
3. Confirm the file exists at `/root/output/result.xlsx`.

### Critical constraints
- Use `openpyxl` to read and write. Do NOT use `data_only=True` when loading (you need to preserve formulas).
- Do NOT add sheets, macros, VBA, external links, or helper tabs.
- Do NOT change existing formatting — only write formulas into the specified cells.
- Do NOT delete or modify any existing content outside the specified cell ranges.
- Write formulas as strings, not computed Python values.

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