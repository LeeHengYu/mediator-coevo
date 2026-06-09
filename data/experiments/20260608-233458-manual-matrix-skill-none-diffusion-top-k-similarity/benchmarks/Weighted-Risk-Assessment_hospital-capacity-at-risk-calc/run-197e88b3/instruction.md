# Task Instruction

Execute the following steps to produce /root/output/result.xlsx from /root/data/workbook.xlsx.

## Phase 0 – Setup & Inspect
1. `mkdir -p /root/output`
2. Open `/root/data/workbook.xlsx` with openpyxl (data_only=False).
3. Print the sheet names to confirm `Task` and `Data` exist.
4. Print the contents of `Task` sheet rows 10-50, columns D-L, so you can see:
   - Row 10 headers (years in H10:L10)
   - Column D series codes for rows 12-17, 19-24, 26-31
   - The layout of rows 35-50
5. Print `Data` sheet rows 21-38 (all columns) to understand the lookup source layout.
6. Note the exact column letters and row numbers before writing any formulas.

## Phase 1 – Lookup Formulas (H12:L17, H19:L24, H26:L31)
For each of the three blocks (rows 12-17, 19-24, 26-31), for each cell in columns H through L:
- Write an `INDEX(MATCH,MATCH)` formula that:
  - Uses `$D{row}` (absolute column, relative row) for the series code.
  - Uses `{col}$10` (relative column, absolute row) for the year.
  - Looks up against the Data sheet rows 21:38.
- Specifically, determine from your inspection which column on Data holds the series codes and which row holds the year headers, then build the formula accordingly. A typical pattern would be:
  `=INDEX(Data!$B$21:$S$38, MATCH($D12,Data!$A$21:$A$38,0), MATCH(H$10,Data!$B$20:$S$20,0))`
  Adjust the exact ranges based on what you see in the Data sheet.
- Use mixed references (`$D12` and `H$10`) so the formula can be written per-cell in a loop without manual adjustment.

## Phase 2 – Net Capacity Headroom (H35:L40)
For each cell in H35:L40 (6 rows × 5 columns):
- The formula computes: `(Available Care Slots - Occupied Care Slots) / Staffed Bed Capacity * 100`
- Map the three input blocks from Phase 1:
  - Block 1 (H12:L17) = one metric (e.g., Available Care Slots)
  - Block 2 (H19:L24) = another metric (e.g., Occupied Care Slots)
  - Block 3 (H26:L31) = Staffed Bed Capacity
- Verify which block is which by reading the labels in the Task sheet (likely in column C or D area, or a header row above each block like rows 11, 18, 25).
- Example formula for H35: `=(H12-H19)/H26*100`  (adjust row offsets to align the same cluster across blocks)

## Phase 3 – Summary Statistics (H42:L47)
For each column H through L, write these formulas:
- Row 42 (Minimum): `=MIN(H35:H40)`
- Row 43 (Maximum): `=MAX(H35:H40)`
- Row 44 (Median): `=MEDIAN(H35:H40)`
- Row 45 (Mean): `=AVERAGE(H35:H40)`
- Row 46 (25th percentile): `=PERCENTILE.INC(H35:H40,0.25)`
- Row 47 (75th percentile): `=PERCENTILE.INC(H35:H40,0.75)`

**IMPORTANT**: Use `PERCENTILE.INC` (not `PERCENTILE`) to avoid #NAME? errors in the verification environment. This is a known issue from cross-task feedback.

## Phase 4 – Weighted Mean (H50:L50)
For each column H through L:
- `=SUMPRODUCT(H35:H40,H26:H31)/SUM(H26:H31)`
- This uses the Net capacity headroom percentages as values and Staffed Bed Capacity as weights.

## Phase 5 – Save & Validate
1. Save the workbook to `/root/output/result.xlsx`.
2. Re-open `/root/output/result.xlsx` with openpyxl (data_only=False).
3. Print the formulas in a sample of cells to confirm they are correctly written:
   - H12, L17 (lookup block boundaries)
   - H35, L40 (headroom block boundaries)
   - H42, H46, H47 (MIN and PERCENTILE.INC)
   - H50, L50 (weighted mean)
4. Confirm no new sheets were added.
5. Confirm the file exists at the expected path.

## Key Constraints
- Do NOT use `PERCENTILE` — always use `PERCENTILE.INC`.
- Do NOT add sheets, macros, VBA, external links, or helper tabs.
- Do NOT alter existing formatting.
- Adjust all ranges based on actual inspection in Phase 0 — do not assume column/row positions without verifying.

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