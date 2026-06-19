# Task Instruction

Execute the following steps to produce `/root/output/result.xlsx`.

## Preparation

1. `mkdir -p /root/output`
2. Read `/root/data/workbook.xlsx` with openpyxl (`data_only=False`) to inspect:
   - Sheet `Task`: examine column D rows 12-17, 19-24, 26-31 (series codes), row 10 columns H-L (years), rows 35-47 labels, row 50 label.
   - Sheet `Data`: rows 21-38 structure — identify which column holds the series code and how years are laid out (row vs column).
   Print out enough cell values to understand the exact layout before writing any formulas.

## Step 1 — Lookup formulas in yellow blocks

For every cell in H12:L17, H19:L24, H26:L31, write an Excel formula that looks up the value from sheet `Data` rows 21:38 using:
- The series code from column D of the cell's row on sheet `Task`.
- The year from row 10 of the cell's column on sheet `Task`.

Choose INDEX/MATCH (safest cross-version pattern):
```
=INDEX(Data!<data_range>, MATCH($D12, Data!<series_code_column>, 0), MATCH(H$10, Data!<year_row>, 0))
```
Adjust `<data_range>`, `<series_code_column>`, and `<year_row>` based on what you discover in the Data sheet. Use absolute row references for the year row (`H$10`) and absolute column references for the series code (`$D12`) so the formula copies correctly across the 5×6 block.

Repeat the same pattern for all three blocks (rows 12-17, 19-24, 26-31), adjusting the row reference in column D accordingly.

## Step 2 — Net reliability gap (H35:L40)

Identify which block is "Successful API Requests" (likely rows 12-17), which is "Failed API Requests" (likely rows 19-24), and which is "Compute Capacity" (likely rows 26-31) by reading the labels in the Task sheet.

For each cell in H35:L40, write:
```
=(H12 - H19) / H26 * 100
```
(adjusting row references so each of the 6 regions maps correctly from the three blocks above).

## Step 2 continued — Summary statistics (H42:L47)

For each column H through L:
- Row 42 (Minimum): `=MIN(H35:H40)`
- Row 43 (Maximum): `=MAX(H35:H40)`
- Row 44 (Median): `=MEDIAN(H35:H40)`
- Row 45 (Mean): `=AVERAGE(H35:H40)`
- Row 46 (25th percentile): `=PERCENTILE(H35:H40,0.25)`
- Row 47 (75th percentile): `=PERCENTILE(H35:H40,0.75)`

**CRITICAL**: Use `PERCENTILE` (not `PERCENTILE.INC`). The previous run failed with `#NAME?` because `PERCENTILE.INC` (or a variant) was not recognized. The classic `PERCENTILE` function is universally supported. Double-check the exact string you write — it must be literally `PERCENTILE(range,k)` with no dots or suffixes.

## Step 3 — Weighted mean (H50:L50)

For each column H through L:
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```
This computes the weighted mean of the Net reliability gap percentages using Compute Capacity as weights.

## Saving

Save to `/root/output/result.xlsx` using `wb.save('/root/output/result.xlsx')`. Do NOT use `data_only=True` when loading (that strips formulas).

## Validation

After saving, reload the file and:
1. Print cells H12, L17, H19, L24, H26, L31 to confirm they contain formula strings (not None).
2. Print cells H35, L40 to confirm formula strings.
3. Print cells H42:H47 to confirm formula strings — especially H46 and H47 must show `=PERCENTILE(...)` with no dots.
4. Print cells H50:L50 to confirm SUMPRODUCT formulas.
5. Confirm no new sheets were added.

If any cell is None or contains `#NAME?` in its formula text, stop and debug before finishing.

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