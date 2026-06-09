# Task Instruction

Execute the following steps to produce `/root/output/result.xlsx` from `/root/data/workbook.xlsx`.

## Preliminary inspection
1. Open `/root/data/workbook.xlsx` with openpyxl (data_only=False) and inspect:
   - Sheet `Task`: read row 10 (years in H10:L10), column D rows 12-17, 19-24, 26-31 (series codes), rows 35-40 labels, rows 42-47 labels, row 50 label.
   - Sheet `Data`: read rows 21-38 to understand the data layout (which row holds which series code, which columns hold which years). Print the first column and header row so you know the exact structure.
2. Print all findings before writing any formulas.

## Step 1 – Lookup formulas in H12:L17, H19:L24, H26:L31
For each yellow cell in those ranges, write an Excel formula that looks up the value from `Data!$A$21:$Z$38` (adjust the actual range based on your inspection). Use the series code from column D of the current row and the year from row 10.

Use the `INDEX/MATCH` pattern:
```
=INDEX(Data!$B$21:$XX$38, MATCH($D12, Data!$A$21:$A$38, 0), MATCH(H$10, Data!$B$20:$XX$20, 0))
```
Adjust the concrete column/row references to match the actual data layout you discovered. Lock references appropriately so the formula can be filled across H-L and down the rows.

## Step 2 – Net SLA buffer (H35:L40) and statistics (H42:L47)
For H35:L40, use:
```
=(H12 - H19) / H26 * 100
```
(Adjust row references per block: rows 12-17 = Latency Budget Preserved, rows 19-24 = Latency Budget Consumed, rows 26-31 = Covered Request Capacity. Map each of the 6 services accordingly: row 35 uses rows 12,19,26; row 36 uses rows 13,20,27; etc.)

For statistics in H42:L47, use these formulas (example for column H):
- H42 (MIN): `=MIN(H35:H40)`
- H43 (MAX): `=MAX(H35:H40)`
- H44 (MEDIAN): `=MEDIAN(H35:H40)`
- H45 (MEAN): `=AVERAGE(H35:H40)`
- H46 (25th percentile): `=_xlfn.PERCENTILE.INC(H35:H40,0.25)`
- H47 (75th percentile): `=_xlfn.PERCENTILE.INC(H35:H40,0.75)`

**CRITICAL**: For the PERCENTILE rows, you MUST use the `_xlfn.` prefix: `_xlfn.PERCENTILE.INC`. This is required because openpyxl does not automatically add the prefix for newer Excel functions, and without it the formula produces #NAME? errors. Do NOT use bare `PERCENTILE` or bare `PERCENTILE.INC` without the `_xlfn.` prefix.

Similarly, if you use MEDIAN, prefix it as `_xlfn.MEDIAN` — but first test whether MEDIAN works without prefix (it usually does as a legacy function). If in doubt, keep MEDIAN without prefix but always use `_xlfn.PERCENTILE.INC` for percentiles.

## Step 3 – Weighted mean in H50:L50
For each column (H through L):
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```
This computes the weighted mean of the Net SLA buffer percentages using Covered Request Capacity as weights.

## Saving
- Create `/root/output/` directory if it doesn't exist.
- Save the workbook to `/root/output/result.xlsx`.
- Do NOT add any new sheets, macros, VBA, external links, or helper tabs.
- Do NOT alter existing formatting.

## Validation
After saving, re-open `/root/output/result.xlsx` with openpyxl (data_only=False) and:
1. Print the formula in H46 and H47 to confirm they contain `_xlfn.PERCENTILE.INC`.
2. Print a sample lookup formula from H12 to confirm it references Data sheet correctly.
3. Print the formula in H50 to confirm SUMPRODUCT structure.
4. Confirm no new sheets were added.

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