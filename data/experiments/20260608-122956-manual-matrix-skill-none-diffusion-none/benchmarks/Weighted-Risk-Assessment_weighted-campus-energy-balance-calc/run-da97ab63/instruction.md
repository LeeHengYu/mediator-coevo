# Task Instruction

Execute the following steps carefully to produce /root/output/result.xlsx.

## Phase 0 – Inspect the workbook
1. `mkdir -p /root/output`
2. Open `/root/data/workbook.xlsx` with openpyxl (data_only=False).
3. Print sheet names.
4. For sheet `Task`:
   - Print cells D12:D17, D19:D24, D26:D31 (series codes for the three blocks).
   - Print row 10 from columns H to L (the year headers).
   - Print cells H35:H40 labels or D35:D40 if labels are there.
   - Print rows 42-47 column D or G (stat labels).
   - Print row 50 column D or G (weighted mean label).
   - Print any existing content/formulas in H12:L17, H19:L24, H26:L31, H35:L40, H42:L47, H50:L50.
5. For sheet `Data`:
   - Print rows 21 through 38 fully (all non-empty columns). Pay special attention to:
     a. Column A (or whichever column holds the series code) – note exact strings including spaces.
     b. The header row for years – which row is it? Is it row 21, or is row 21 the first data row? Check rows 19-21.
     c. The column range that holds years/data values.
   - Identify: which column has series codes, which row has year headers, and the rectangular data range.

Print everything clearly so we can construct correct formulas.

## Phase 1 – Write lookup formulas in H12:L17, H19:L24, H26:L31

Based on the inspection, construct INDEX/MATCH formulas. The pattern for each cell should be:

```
=INDEX(Data!<data_range>, MATCH($D12, Data!<series_code_column>, 0), MATCH(H$10, Data!<year_header_row>, 0))
```

Replace `<data_range>`, `<series_code_column>`, and `<year_header_row>` with the exact ranges found in Phase 0.

**Critical checks before writing formulas:**
- Verify that the series codes in column D of Task sheet match EXACTLY (character-for-character) with the series codes in the Data sheet. Print both side by side and compare lengths.
- Verify that the year values in row 10 of Task sheet match the type (number vs string) of the year headers in the Data sheet. If H10 contains 2019 as a number, the Data year header must also be a number for MATCH to work. Print `type()` and `repr()` of both.
- If there's a type mismatch, wrap the MATCH argument appropriately or note it.

Write the formulas for all three blocks (H12:L17, H19:L24, H26:L31) using a loop. Use absolute row references for the year row ($10) and absolute column references for the series code column ($D). Make sure the Data sheet ranges use absolute references ($ signs) so they don't shift.

## Phase 2 – Net renewable balance formulas in H35:L40

For each campus (6 rows) and each year (5 columns), write:
```
=(H12 - H19) / H26 * 100
```
where H12 is from the first block (Renewable Generation), H19 from the second block (Grid Consumption), and H26 from the third block (Baseline Energy Demand). Adjust row references for each campus row.

Specifically, for row 35 col H: `=(H12-H19)/H26*100`, for row 36 col H: `=(H13-H20)/H27*100`, etc.

## Phase 3 – Summary statistics in H42:L47

For each column (H through L):
- Row 42 (min): `=MIN(H35:H40)`
- Row 43 (max): `=MAX(H35:H40)`
- Row 44 (median): `=MEDIAN(H35:H40)`
- Row 45 (mean): `=AVERAGE(H35:H40)`
- Row 46 (25th percentile): `=PERCENTILE(H35:H40,0.25)`
- Row 47 (75th percentile): `=PERCENTILE(H35:H40,0.75)`

**IMPORTANT**: Use `PERCENTILE` (legacy name), NOT `PERCENTILE.INC`. Previous feedback confirms `PERCENTILE` works and `PERCENTILE.INC` causes #NAME? errors.

## Phase 4 – Weighted mean in H50:L50

For each column (H through L):
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```
This uses the net renewable balance percentages as values and the Baseline Energy Demand as weights.

## Phase 5 – Save and validate
1. Save the workbook to `/root/output/result.xlsx`.
2. Reopen the saved file and print all formula cells to confirm they contain formulas (not values or None).
3. Verify no cells contain #NAME?, #REF!, or other error indicators in the formula text.
4. Confirm the file exists and has reasonable size.

## Key constraints
- Do NOT add new sheets, macros, VBA, external links, or helper tabs.
- Do NOT change existing formatting (fonts, colors, borders, etc.).
- Work only inside sheets `Task` and `Data`.
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
Task metadata: author_email=catpaw@meituan.com, author_name=CatPaw Task Engineer, category=spreadsheet-formula-reuse, difficulty=medium, tags=[excel, formulas, lookup, statistics, weighted-mean].
Verifier config: timeout_sec=600.0.