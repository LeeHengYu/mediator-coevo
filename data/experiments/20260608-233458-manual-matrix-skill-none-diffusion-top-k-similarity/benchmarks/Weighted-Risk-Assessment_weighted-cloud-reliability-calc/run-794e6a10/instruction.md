# Task Instruction

Execute the following steps carefully and in order.

## Phase 0 — Inspect the workbook layout

1. Open `/root/data/workbook.xlsx` with openpyxl (data_only=False).
2. Print the sheet names.
3. On sheet `Task`:
   - Print rows 10-11 (to see the year headers in H10:L10).
   - Print rows 12-17 column A-L (to see series codes in column D, and the yellow target cells H12:L17).
   - Print rows 19-24 column A-L (second lookup block).
   - Print rows 26-31 column A-L (third lookup block — Compute Capacity).
   - Print rows 33-50 column A-L (derived metrics, stats, weighted mean).
   - Pay special attention to: what text is in D12:D17, D19:D24, D26:D31, D35:D40; what years are in H10:L10; what labels are in G42:G47 (min/max/median/mean/percentiles).
4. On sheet `Data`:
   - Print rows 19-40 columns A-Z (or at least A-T) to see the full data block in rows 21:38.
   - Determine: Are series codes in a column (which column?) or a row (which row)? Are years in a row (which row?) or a column (which column?)?
   - Print the exact cell references for the top-left corner, the series-code axis, and the year axis of the data block.

Record all findings before proceeding. Do NOT write any formulas until you have confirmed the data layout.

## Phase 1 — Construct lookup formulas (H12:L17, H19:L24, H26:L31)

Based on the layout discovered in Phase 0, write INDEX/MATCH formulas. The general pattern is:

```
=INDEX(Data!<data_range>, MATCH(<series_code>, Data!<series_code_range>, 0), MATCH(<year>, Data!<year_range>, 0))
```

Where:
- `<series_code>` = the cell in column D of the current row on sheet Task (e.g., $D12).
- `<year>` = the cell in row 10 of the current column on sheet Task (e.g., H$10).
- `<data_range>`, `<series_code_range>`, and `<year_range>` must be determined from Phase 0 inspection. Use absolute references with $ where appropriate so the formula can be filled across the 5 columns and 6 rows of each block.

IMPORTANT: If the Data sheet has series codes in a ROW and years in a COLUMN, you may need to swap the MATCH axes or use a transposed approach. Verify by checking that at least one formula resolves to the expected value.

Write formulas into all three blocks: H12:L17, H19:L24, H26:L31.

## Phase 2 — Net reliability gap (H35:L40)

For each of the 6 regions (rows 35-40) and 5 years (columns H-L), write a formula:

```
=(H12 - H19) / H26 * 100
```

Adjust row references to match the correct rows for:
- Successful API Requests (rows 12-17)
- Failed API Requests (rows 19-24)
- Compute Capacity (rows 26-31)

The region order in rows 35-40 must correspond to the same region order in rows 12-17 / 19-24 / 26-31. Verify by checking column D labels.

## Phase 3 — Summary statistics (H42:L47)

For each year column (H through L), calculate:
- Row 42 (Minimum): `=MIN(H35:H40)`
- Row 43 (Maximum): `=MAX(H35:H40)`
- Row 44 (Median): `=MEDIAN(H35:H40)`
- Row 45 (Mean): `=AVERAGE(H35:H40)`
- Row 46 (25th percentile): `=PERCENTILE(H35:H40,0.25)`
- Row 47 (75th percentile): `=PERCENTILE(H35:H40,0.75)`

CRITICAL: Use `PERCENTILE` (the legacy function name), NOT `PERCENTILE.INC` or `PERCENTILE.EXC`. The evaluator does not recognize the dotted variants. Verify the labels in column G to confirm which row is which statistic, and adjust row assignments if they differ from the above.

## Phase 4 — Weighted mean (H50:L50)

For each year column, use SUMPRODUCT with the net reliability gap values as the data and Compute Capacity as weights:

```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```

This computes the weighted mean for GCM.

## Phase 5 — Save and verify

1. Save the workbook to `/root/output/result.xlsx` (create the output directory if needed).
2. Re-open the saved file and print the formulas in a few sample cells (e.g., H12, L17, H35, H42, H46, H50) to confirm they were written correctly.
3. Confirm no extra sheets were added and the sheet names are unchanged.
4. Confirm the formula in H46 uses `PERCENTILE` (not `PERCENTILE.INC` or `PERCENTILE.EXC`).

Do NOT add macros, VBA, external links, helper tabs, or new sheets. Do NOT alter existing formatting.

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