# Task Instruction

Execute the following multi-phase plan to build the workbook formulas.

## Phase 0 – Inspect the workbook

1. Copy `/root/data/workbook.xlsx` to `/root/output/result.xlsx`.
2. Open `/root/output/result.xlsx` with openpyxl (data_only=False).
3. Print the following so we can build correct formulas:
   - Sheet names.
   - `Task` sheet: cells D12:D17, D19:D24, D26:D31 (series codes), H10:L10 (year headers), and the labels in B35:B40, B42:B47, B50.
   - `Data` sheet: row 21 through row 38, columns A–Z (or however wide data extends). Print every cell value and its type (string vs number). Also print row 20 (or whichever row contains the year headers on Data) to see the header row.
   - Check whether Data has a column of series codes and a row of years; identify the exact column letter for series codes and the exact row number for years.
4. Print cell types for H10 on Task (is it int, float, string?) and for the corresponding year cell on Data sheet.

## Phase 1 – Write lookup formulas (H12:L31)

Based on the inspection, construct INDEX/MATCH formulas. Use this pattern (adjust ranges after inspection):

```
=INDEX(Data!$B$21:$<lastcol>$38, MATCH($D12, Data!$A$21:$A$38, 0), MATCH(H$10, Data!$B$20:$<lastcol>$20, 0))
```

Key rules:
- The series-code column reference in MATCH must exactly cover the same rows as the data block's first column.
- The year-header row reference in MATCH must exactly cover the same columns as the data block's first row.
- Use `$D12` (mixed reference: column absolute, row relative) so the formula can be filled down; use `H$10` (row absolute) so it can be filled across.
- **If the year in H10 is numeric but the Data header row stores years as strings (or vice versa), wrap the lookup value to match**: e.g., `MATCH(H$10*1, ...)` or `MATCH(TEXT(H$10,"0"), ...)`.
- Fill H12:L17 for the first block, H19:L24 for the second, H26:L31 for the third. Each block's D column has its own series codes; the pattern is the same.

After writing formulas, read back a few cells to confirm they contain formula strings (not None).

## Phase 2 – Net patient flow (H35:L40)

For each hospital row i (1–6), the formula is:
```
=(H{admissions_row} - H{discharges_row}) / H{capacity_row} * 100
```
where:
- Admissions rows = 12:17 (first block)
- Discharges rows = 19:24 (second block)
- Capacity rows = 26:31 (third block)

So H35 = `=(H12-H19)/H26*100`, H36 = `=(H13-H20)/H27*100`, etc. Fill across to column L.

## Phase 3 – Summary statistics (H42:L47)

For each column (H through L):
- Row 42 (Minimum): `=MIN(H35:H40)`
- Row 43 (Maximum): `=MAX(H35:H40)`
- Row 44 (Median): `=MEDIAN(H35:H40)`
- Row 45 (Mean): `=AVERAGE(H35:H40)`
- Row 46 (25th percentile): `=PERCENTILE(H35:H40,0.25)`
- Row 47 (75th percentile): `=PERCENTILE(H35:H40,0.75)`

**Use `PERCENTILE` (legacy name), NOT `PERCENTILE.INC`** — cross-task feedback shows `PERCENTILE.INC` causes `#NAME?` errors in this environment.

## Phase 4 – Weighted mean (H50:L50)

For each column:
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```

## Phase 5 – Save and verify

1. Save the workbook to `/root/output/result.xlsx`.
2. Re-open it and print all formula cells (H12:L17, H19:L24, H26:L31, H35:L40, H42:L47, H50:L50) to confirm every cell contains a formula string starting with '='.
3. Optionally load with data_only=True to check if any cell is None (which would indicate a broken formula reference).

## Important constraints
- Do NOT add sheets, macros, VBA, external links, or helper tabs.
- Do NOT change any existing formatting.
- Only write into the specified yellow cell ranges.
- Use openpyxl for all operations.

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