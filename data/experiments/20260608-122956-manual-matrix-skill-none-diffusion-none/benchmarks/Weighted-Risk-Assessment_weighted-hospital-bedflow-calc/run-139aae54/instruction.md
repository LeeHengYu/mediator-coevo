# Task Instruction

Execute the following steps in a single Python script using openpyxl.

## 0. Inspect the Data sheet layout

Open `/root/data/workbook.xlsx` with `openpyxl.load_workbook('/root/data/workbook.xlsx')` (do NOT use `data_only=True`).

Before writing any formulas, inspect and print:
- **Task sheet**: Print cells D12:D17 (series codes for block 1), D19:D24 (block 2), D26:D31 (block 3). Print cells H10:L10 (year headers in row 10). Print cells D35:D40 (hospital names for net patient flow). Print cells D42:D47 (stat labels). Print cell D50 (weighted mean label).
- **Data sheet**: Print row 20 or row 21 area to find the header row. Print column A (or whichever column) for rows 21-38 to find series codes. Print the top row of the data block to find year headers. Identify: (a) which column contains the series codes, (b) which row contains the year headers, (c) the exact rectangular range of the data values.

Print all of this so you can see the exact layout before writing formulas.

## 1. Write lookup formulas (H12:L17, H19:L24, H26:L31)

Based on your inspection, construct INDEX/MATCH formulas. The general pattern should be:

```
=INDEX(Data!<data_range>, MATCH($D12, Data!<series_code_column>, 0), MATCH(H$10, Data!<year_header_row>, 0))
```

Critical details:
- Use `$D12` (column-absolute) so the series code reference stays fixed when filling across columns.
- Use `H$10` (row-absolute) so the year reference stays fixed when filling down rows.
- The `Data!<data_range>` must be the rectangular block of numeric values (not including headers).
- The `Data!<series_code_column>` must be the single column of series codes adjacent to the data.
- The `Data!<year_header_row>` must be the single row of year headers above the data.
- Make sure all three ranges align: the data range must have the same number of rows as the series code column and the same number of columns as the year header row.

Write these formulas into all cells in the three blocks (H12:L17, H19:L24, H26:L31), adjusting the row reference in $D for each row.

## 2. Write Net Patient Flow formulas (H35:L40)

For each hospital row (6 hospitals), the formula is:
```
=(H12 - H19) / H26 * 100
```
where H12 is the Patient Admissions value, H19 is the Patient Discharges value, and H26 is the Effective Bed Capacity value for that hospital in that year column. Adjust row references for each of the 6 hospitals and column references for each of the 5 years.

## 3. Write statistics formulas (H42:L47)

For each year column (H through L):
- Row 42 (Minimum): `=MIN(H35:H40)`
- Row 43 (Maximum): `=MAX(H35:H40)`
- Row 44 (Median): `=MEDIAN(H35:H40)`
- Row 45 (Mean): `=AVERAGE(H35:H40)`
- Row 46 (25th percentile): `=PERCENTILE(H35:H40,0.25)` — try `PERCENTILE` first; if the verifier rejects it, use `PERCENTILE.INC`
- Row 47 (75th percentile): `=PERCENTILE(H35:H40,0.75)` — same note

**Important**: Use `PERCENTILE` (not `PERCENTILE.INC` or `PERCENTILE.EXC`) as the primary choice since openpyxl and most Excel engines recognize it.

## 4. Write weighted mean formula (H50:L50)

For each year column:
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```
This computes the weighted mean of net patient flow percentages using Effective Bed Capacity as weights.

## 5. Save

Save to `/root/output/result.xlsx`. Create the `/root/output/` directory if it doesn't exist.

## 6. Verify

After saving, reload the file and print the formulas (not values) in a few sample cells (e.g., H12, L17, H35, H42, H46, H50) to confirm they were written correctly.

## Important constraints
- Do NOT use `data_only=True` when loading.
- Do NOT add new sheets, macros, VBA, or external links.
- Do NOT modify any existing formatting.
- Do NOT delete or rename any existing sheets.
- Work only inside the existing 'Task' and 'Data' sheets.

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