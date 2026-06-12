# Task Instruction

Execute the following steps to produce `/root/output/result.xlsx`.

## Step 0 – Inspect the workbook
1. `mkdir -p /root/output`
2. Open `/root/data/workbook.xlsx` with openpyxl (data_only=False).
3. Print sheet names to confirm `Task` and `Data` exist.
4. On sheet `Data`, print rows 20-39 (all columns up to ~col 20) so you can see the header row and the data layout (series codes, years, values). Pay special attention to:
   - Which column holds the series/code identifier.
   - Which row holds the year headers.
   - The exact range of data rows 21:38.
5. On sheet `Task`, print rows 1-55 (columns A-M) to see:
   - The series codes in column D for rows 12-17, 19-24, 26-31.
   - The year values in row 10 for columns H-L.
   - The labels in rows 35-40, 42-47, and 50.
   - Any existing formulas or values.

Record the exact column letters/numbers for the series-code column and year-header row on `Data`. You will need these for the MATCH formulas.

## Step 1 – Lookup formulas in H12:L17, H19:L24, H26:L31

For each of the three blocks, write an INDEX/MATCH formula into every cell in the 6-row × 5-column range. The formula pattern is:

```
=INDEX(Data!<data_area>, MATCH($D12, Data!<series_code_column>, 0), MATCH(H$10, Data!<year_header_row>, 0))
```

Where:
- `<data_area>` is the rectangular block on `Data` containing the numeric values (rows 21:38, from the first data column to the last data column).
- `<series_code_column>` is the single column on `Data` that holds the series codes (same rows 21:38).
- `<year_header_row>` is the single row on `Data` that holds the year labels (same columns as the data area).
- `$D12` locks the column to D (so copying across columns still reads D); the row number changes per row.
- `H$10` locks the row to 10 (so copying down rows still reads row 10); the column letter changes per column.

Use the exact cell references you discovered in Step 0. Double-check that the MATCH ranges align with the INDEX data area dimensions.

## Step 2 – Net budget buffer (H35:L40) and statistics (H42:L47)

### H35:L40 – Net budget buffer
The formula for each cell is:
```
=(H12 - H19) / H26 * 100
```
where H12 is Committed Funding, H19 is Operating Spend, H26 is Approved Budget Base (adjust row references for each department row: 12→35, 13→36, …, 17→40 mapping to the corresponding rows in the three lookup blocks: 12-17, 19-24, 26-31).

So for cell H35: `=(H12-H19)/H26*100`
For cell H36: `=(H13-H20)/H27*100`
... and so on through H40: `=(H17-H24)/H31*100`
Columns I-L follow the same pattern.

### H42:L47 – Summary statistics
For each column (H through L):
- Row 42 (Min): `=MIN(H35:H40)`
- Row 43 (Max): `=MAX(H35:H40)`
- Row 44 (Median): `=MEDIAN(H35:H40)`
- Row 45 (Mean): `=AVERAGE(H35:H40)`
- Row 46 (25th percentile): `=PERCENTILE(H35:H40,0.25)`
- Row 47 (75th percentile): `=PERCENTILE(H35:H40,0.75)`

IMPORTANT: Use `PERCENTILE` (not `PERCENTILE.INC`) to avoid `#NAME?` errors.

## Step 3 – Weighted mean (H50:L50)
For each column (H through L):
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```
This computes the weighted mean of the net budget buffer percentages using Approved Budget Base as weights.

## Step 4 – Save and validate
1. Save the workbook to `/root/output/result.xlsx`.
2. Reopen the saved file and print the following to confirm formulas were written (not None):
   - Cells H12, L17 (lookup block 1)
   - Cells H19, L24 (lookup block 2)
   - Cells H26, L31 (lookup block 3)
   - Cells H35, L40 (net budget buffer)
   - Cells H42, H47 (statistics)
   - Cell H50 (weighted mean)
3. All should show formula strings (starting with `=`), not None.
4. As an extra check, use openpyxl to evaluate or at least confirm that the Data sheet references are valid by verifying the series codes in column D match entries in the Data sheet's series-code column.

## Critical Notes
- Do NOT add new sheets, macros, VBA, external links, or helper tabs.
- Do NOT change existing formatting.
- Use `PERCENTILE` not `PERCENTILE.INC`.
- Use mixed references (`$D` for column lock, `$10` for row lock) in INDEX/MATCH formulas.
- The row-to-row mapping between the three lookup blocks (12-17, 19-24, 26-31) and the derived block (35-40) must be consistent: row 35 uses rows 12, 19, 26; row 36 uses rows 13, 20, 27; etc.
- Carefully inspect the Data sheet layout FIRST before writing any formulas. The exact column/row references on the Data sheet are critical.

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