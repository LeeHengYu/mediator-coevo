# Task Instruction

Execute the following steps to produce /root/output/result.xlsx from /root/data/workbook.xlsx.

## Phase 0 – Inspect the workbook
1. Open /root/data/workbook.xlsx with openpyxl (data_only=False).
2. Print the sheet names.
3. On sheet **Task**:
   - Print cells D12:D17, D19:D24, D26:D31 (series codes for the three lookup blocks).
   - Print row 10 from columns H through L (the year headers).
   - Print any labels in column B or C for rows 35-40 (cluster names for Net capacity headroom).
   - Print rows 42-47 column B/C (stat labels: min, max, median, mean, 25th, 75th).
   - Print row 50 column B/C/D (weighted mean label).
4. On sheet **Data**:
   - Print row 21 (header row) columns A-R or however far data extends.
   - Print column A or B for rows 22-39 (series codes in the data).
   - Identify which column holds the series codes and which row holds the year headers.
   - Confirm the exact year values (type and content) in the header row.

Record all findings before writing any formulas.

## Phase 1 – Lookup formulas (H12:L17, H19:L24, H26:L31)
For each of the three 6-row × 5-column blocks, write an INDEX/MATCH formula.

Pattern (adjust ranges based on Phase 0 findings):
```
=INDEX(Data!$C$22:$<lastcol>$39, MATCH($D12, Data!$B$22:$B$39, 0), MATCH(H$10, Data!$C$21:$<lastcol>$21, 0))
```
- Use **$D12** (column-absolute, row-relative) so the series code reference locks to column D but moves with the row.
- Use **H$10** (row-absolute, column-relative) so the year reference locks to row 10 but moves with the column.
- Adjust the Data sheet ranges ($B$22:$B$39 for series codes, $C$21:$<lastcol>$21 for year headers, $C$22:$<lastcol>$39 for the data body) based on what you discover in Phase 0.
- **Critical**: make sure the series code column used in MATCH is NOT included in the data body range passed to INDEX; they must be aligned but separate.

Write these formulas into all 90 cells (3 blocks × 6 rows × 5 columns).

## Phase 2 – Net capacity headroom (H35:L40)
For each cell in H35:L40, write:
```
=(H12 - H19) / H26 * 100
```
where H12 corresponds to "Available Care Slots" (rows 12-17), H19 to "Occupied Care Slots" (rows 19-24), and H26 to "Staffed Bed Capacity" (rows 26-31). Adjust row references so each of the 6 cluster rows maps correctly (row 35→rows 12,19,26; row 36→rows 13,20,27; etc.).

## Phase 3 – Summary statistics (H42:L47)
For each column H through L:
- Row 42: `=MIN(H35:H40)`
- Row 43: `=MAX(H35:H40)`
- Row 44: `=MEDIAN(H35:H40)`
- Row 45: `=AVERAGE(H35:H40)`
- Row 46: `=PERCENTILE(H35:H40,0.25)`
- Row 47: `=PERCENTILE(H35:H40,0.75)`

**Important**: Use exactly `PERCENTILE` (not `PERCENTILE.INC` or `PERCENTILE.EXC`) to avoid name errors. Verify the stat label order from Phase 0 and assign formulas accordingly (the order above is min/max/median/mean/25th/75th – adjust if the labels say otherwise).

## Phase 4 – Weighted mean (H50:L50)
For each column H through L:
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```
This computes the weighted mean of headroom percentages using Staffed Bed Capacity as weights.

## Phase 5 – Save and verify
1. Save to /root/output/result.xlsx (create /root/output/ if needed). Do NOT change any formatting, do NOT add sheets.
2. Re-open the saved file with openpyxl (data_only=False) and print a sample of formulas from each block to confirm they were written correctly:
   - H12, L17 (lookup block 1 corners)
   - H19, L24 (lookup block 2 corners)
   - H26, L31 (lookup block 3 corners)
   - H35, L40 (headroom corners)
   - H42, H47 (stats)
   - H50, L50 (weighted mean)
3. Confirm no extra sheets were added.

Do not use xlcalc or any formula evaluation library. Just write the formulas as strings into the cells.

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