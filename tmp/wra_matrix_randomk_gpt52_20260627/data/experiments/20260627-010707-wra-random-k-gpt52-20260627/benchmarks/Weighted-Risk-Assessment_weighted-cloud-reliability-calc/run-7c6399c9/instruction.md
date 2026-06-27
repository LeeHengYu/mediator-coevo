# Task Instruction

Execute the following steps exactly to produce /root/output/result.xlsx.

## 0 – Environment & Inspection
```bash
mkdir -p /root/output
pip install openpyxl
```
Open `/root/data/workbook.xlsx` with openpyxl (data_only=False) and inspect:
- Sheet names (confirm `Task` and `Data` exist).
- On `Task`: read row 10 to find the year headers in columns H–L. Read column D rows 12–17, 19–24, 26–31 to find the series codes. Read row 35–40 labels, row 42–47 labels, row 50 label.
- On `Data`: read rows 21–38 to understand the layout (which column holds the series code, which row holds years, where values start).
Print all of this so you can build correct formulas.

## 1 – Lookup formulas in H12:L17, H19:L24, H26:L31

For each block, every yellow cell needs a formula that looks up the value from `Data!$21:$38` using the series code in column D of that row and the year in row 10 of that column.

Use the INDEX/MATCH pattern. The exact references depend on what you find during inspection, but the template is:
```
=INDEX(Data!<value_range>, MATCH($D12, Data!<series_code_column>, 0), MATCH(H$10, Data!<year_row>, 0))
```
Make sure:
- The series-code column reference and year row reference are taken from the actual Data sheet layout you inspected.
- Row references use `$D12` style (lock column, float row) and `H$10` style (float column, lock row) so the formula can be applied across the 5×6 grid of each block.
- Apply to all three blocks: rows 12–17, 19–24, 26–31, columns H–L.

## 2 – Net reliability gap (H35:L40)

Based on the task description, the three blocks correspond to three indicators for six regions. Identify which block is "Successful API Requests", which is "Failed API Requests", and which is "Compute Capacity" by reading labels near each block (e.g., labels in column B or C near rows 12, 19, 26). Then for each cell in H35:L40:
```
=(H12 - H19) / H26 * 100
```
(Adjust row references so that for each region i=0..5, the formula references the correct rows from each block. E.g., row 35 uses rows 12, 19, 26; row 36 uses rows 13, 20, 27; etc.)

IMPORTANT: Verify which block maps to which indicator by reading the labels. The order might differ from the example above.

## 3 – Summary statistics (H42:L47)

For each column (H through L), write these formulas:
- Row 42 (Minimum): `=MIN(H35:H40)`
- Row 43 (Maximum): `=MAX(H35:H40)`
- Row 44 (Median): `=MEDIAN(H35:H40)`
- Row 45 (Mean): `=AVERAGE(H35:H40)`
- Row 46 (25th percentile): `=PERCENTILE(H35:H40,0.25)`
- Row 47 (75th percentile): `=PERCENTILE(H35:H40,0.75)`

**CRITICAL**: Use `PERCENTILE` — NOT `PERCENTILE.INC`, NOT `_xlfn.PERCENTILE.INC`, NOT `PERCENTILE.EXC`. The previous run failed because of #NAME? errors from using an unrecognized function name. Use the classic `PERCENTILE` function only.

Also verify the row labels on the Task sheet to confirm which row is which statistic (min, max, median, mean, 25th, 75th). Adjust the mapping if the labels differ from the order above.

## 4 – Weighted mean (H50:L50)

For each column H–L:
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```
This computes the weighted mean of the Net reliability gap percentages using Compute Capacity as weights.

## 5 – Save

Save the workbook to `/root/output/result.xlsx`. Do NOT change formatting, do NOT add sheets, macros, VBA, external links, or helper tabs.

## 6 – Validation

After saving, reopen `/root/output/result.xlsx` with openpyxl (data_only=False) and:
1. Print a sample of formulas from each block (e.g., H12, H19, H26, H35, H42, H46, H47, H50) to confirm they are set correctly.
2. Confirm no cell contains `PERCENTILE.INC` or `_xlfn.` anywhere.
3. Confirm sheets are still only `Task` and `Data`.
4. If any test script exists at `/root/test_output.py` or similar, run it with `python -m pytest -xvs` and report results.

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