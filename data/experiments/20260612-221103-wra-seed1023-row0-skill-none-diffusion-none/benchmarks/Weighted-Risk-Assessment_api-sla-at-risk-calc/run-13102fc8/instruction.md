# Task Instruction

Execute the following steps in order.

## Step 0 – Inspect the workbook

Open `/root/data/workbook.xlsx` with openpyxl (data_only=False). Print:

1. **Task sheet:**
   - Row 10 (columns A–L) to see the year headers.
   - Column D, rows 12–31, to see the series codes for the three blocks.
   - Rows 35–50, columns A–L (values or formulas), to see labels for Net SLA buffer, stats, and weighted mean rows.
   - The fill color of cell H12 to confirm it is yellow / the target area.

2. **Data sheet:**
   - Row 1 (or the header row) through row 5, columns A–Z, to see column headers.
   - Rows 21–38, columns A–B (or A–C), to see the series/code column and identify which column holds the lookup key and which columns hold the year data.
   - The full row 21 across all populated columns to see the year labels in the Data sheet.

Print everything clearly so we can determine:
- Which column in `Data` holds the series code (e.g., column A or B).
- Which row in `Data` holds the year labels (e.g., row 21 or a header row above row 21).
- The exact extent of the data range (first and last populated column).

## Step 1 – Write lookup formulas in H12:L17, H19:L24, H26:L31

Based on the inspection, write `INDEX(MATCH,MATCH)` formulas into the 3 × 6 × 5 = 90 yellow cells. Each formula should:
- Match the series code in column D of the current row against the series-code column in `Data` rows 21:38.
- Match the year in row 10 of the current column against the year row in `Data`.
- Return the intersecting value from the `Data` range.

Use absolute references for the Data lookup array and the two lookup vectors; use a mixed reference so the series code (row) and year (column) shift correctly when the formula is placed across the 6×5 block.

Example pattern (adjust column letters and row numbers per inspection):
```
=INDEX(Data!$B$21:$Z$38, MATCH($D12, Data!$A$21:$A$38, 0), MATCH(H$10, Data!$B$20:$Z$20, 0))
```

## Step 2 – Net SLA buffer (H35:L40) and summary statistics (H42:L47)

For each of the 6 services (rows 35–40) and 5 year-columns (H–L):
```
= (H12 - H19) / H26 * 100
```
where H12 is Latency Budget Preserved, H19 is Latency Budget Consumed, H26 is Covered Request Capacity. Adjust row offsets so row 35 maps to the first service, row 36 to the second, etc., matching the order in the lookup blocks.

Then in rows 42–47 for each column (H–L), place:
- Row 42: `=MIN(H35:H40)`
- Row 43: `=MAX(H35:H40)`
- Row 44: `=MEDIAN(H35:H40)`
- Row 45: `=AVERAGE(H35:H40)`
- Row 46: `=PERCENTILE(H35:H40,0.25)`
- Row 47: `=PERCENTILE(H35:H40,0.75)`

Use `PERCENTILE` (no .INC/.EXC suffix) for compatibility.

## Step 3 – Weighted mean in H50:L50

For each year column (H–L):
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```

## Step 4 – Save and verify

1. Save to `/root/output/result.xlsx` (create the output directory if needed). Preserve all existing formatting; do not add sheets, macros, VBA, or external links.
2. Re-open the saved file with openpyxl (data_only=False) and print the formulas in cells H12, L17, H19, L24, H26, L31, H35, L40, H42, H47, H50, L50 to confirm they are present and correctly structured.
3. If any cell is None or empty, diagnose and fix before finishing.

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