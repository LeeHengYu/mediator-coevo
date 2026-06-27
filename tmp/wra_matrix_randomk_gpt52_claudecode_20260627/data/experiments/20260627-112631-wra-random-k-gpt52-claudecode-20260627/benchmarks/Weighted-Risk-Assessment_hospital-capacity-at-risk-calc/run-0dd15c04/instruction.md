# Task Instruction

Execute the following steps in order to produce /root/output/result.xlsx.

## 0 – Preparation
```bash
mkdir -p /root/output
```
Open and inspect `/root/data/workbook.xlsx` with openpyxl (data_only=False) to understand:
- Sheet names (expect `Task` and `Data`).
- The layout of the `Task` sheet: column D series codes, row 10 year headers (H10:L10), yellow target ranges.
- The layout of the `Data` sheet rows 21–38: how series codes and years are arranged (row vs column orientation). Print rows 19–40 and columns A–M to understand the header row and data layout.

## 1 – Write lookup formulas in H12:L17, H19:L24, H26:L31

For every cell in these three blocks, write an Excel formula that retrieves the value from the `Data` sheet rows 21:38. The formula must use one of the allowed patterns. Recommended pattern using INDEX/MATCH:

```
=INDEX(Data!$B$21:$Z$38, MATCH($D{row}, Data!$A$21:$A$38, 0), MATCH(H$10, Data!$B$20:$Z$20, 0))
```

Adjust the exact range references after inspecting the actual Data sheet layout:
- Identify which column holds the series codes on Data (likely column A or B).
- Identify which row holds the year headers on Data (likely row 20 or 21).
- The data matrix range must cover all data columns and rows 21–38.
- $D{row} is the absolute-column reference to the series code in column D of the current Task row.
- H$10 (or I$10, J$10 …) is the year from row 10, with absolute row.

Make sure the column D reference uses `$D` (absolute column) and the row 10 reference uses `$10` (absolute row) so the formula copies correctly across the 5 columns and down the rows.

## 2 – Net capacity headroom (H35:L40)

For each of the 6 hospital clusters (rows 35–40) and each year column (H–L), write:
```
=(H12 - H19) / H26 * 100
```
where H12 corresponds to 'Available Care Slots', H19 to 'Occupied Care Slots', H26 to 'Staffed Bed Capacity' for the same cluster and year. Adjust row references per cluster:
- Row 35: uses rows 12, 19, 26
- Row 36: uses rows 13, 20, 27
- Row 37: uses rows 14, 21, 28
- Row 38: uses rows 15, 22, 29
- Row 39: uses rows 16, 23, 30
- Row 40: uses rows 17, 24, 31

## 3 – Summary statistics (H42:L47)

For each year column (H–L):
- Row 42 (MIN):    `=MIN(H35:H40)`
- Row 43 (MAX):    `=MAX(H35:H40)`
- Row 44 (MEDIAN): `=MEDIAN(H35:H40)`
- Row 45 (MEAN):   `=AVERAGE(H35:H40)`
- Row 46 (25th %): `=PERCENTILE(H35:H40, 0.25)`
- Row 47 (75th %): `=PERCENTILE(H35:H40, 0.75)`

**IMPORTANT**: Use the legacy `PERCENTILE` function, NOT `PERCENTILE.INC`. The test evaluator does not recognize `PERCENTILE.INC` and will produce #NAME? errors.

Verify the row-to-statistic mapping by reading the labels in column D (or nearby) of the Task sheet for rows 42–47. Adjust the mapping if the labels differ from the order above.

## 4 – Weighted mean (H50:L50)

For each year column (H–L):
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```
This computes the weighted mean of the net-capacity-headroom percentages (Step 2 values) weighted by Staffed Bed Capacity.

## 5 – Save

Save the workbook to `/root/output/result.xlsx` using openpyxl. Do NOT use data_only mode when loading; preserve all existing formatting, styles, and other content.

## 6 – Validate

After saving, re-open `/root/output/result.xlsx` (data_only=False) and:
1. Print cells H12, H19, H26, H35, H42, H46, H47, H50 to confirm they contain formula strings (not None).
2. Confirm no cell in the target ranges is None or empty.
3. Confirm that the PERCENTILE formulas use `PERCENTILE(` and NOT `PERCENTILE.INC(`.
4. Run the test suite if available: `cd /root && python -m pytest test_output.py -v` (or whatever test file exists). Report results.

## Key Pitfalls to Avoid
- Do NOT use PERCENTILE.INC — use PERCENTILE.
- Do NOT leave any target cell empty/None — every yellow cell must have a formula.
- Do NOT add new sheets, macros, VBA, or external links.
- Do NOT alter existing formatting.
- Inspect the Data sheet layout carefully before writing formulas; wrong range references will produce #REF! or wrong values.

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