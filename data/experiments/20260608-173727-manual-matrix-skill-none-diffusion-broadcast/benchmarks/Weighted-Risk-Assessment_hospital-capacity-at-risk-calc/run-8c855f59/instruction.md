# Task Instruction

Execute the following steps to produce `/root/output/result.xlsx`.

## 0 – Inspect the workbook
```
cp /root/data/workbook.xlsx /root/output/result.xlsx
```
Open `/root/output/result.xlsx` with openpyxl (data_only=False). Inspect:
- Sheet `Task`: read row 10 (years in H10:L10), column D for rows 12-17, 19-24, 26-31 (series codes), rows 35-40 labels, rows 42-47 labels, row 50 label. Print all of these so you know the exact layout.
- Sheet `Data`: read rows 21-38 completely (all columns). Print the header row and a few data rows to understand the lookup table structure (which column holds the series code, which row/column holds years, where values live).

Do NOT proceed until you have printed and understood both sheets.

## 1 – Populate lookup formulas in H12:L17, H19:L24, H26:L31

For each cell in these three blocks, write a formula that:
- Takes the series code from column D of the same row on `Task`
- Takes the year from row 10 of the same column on `Task`
- Looks up the value from sheet `Data` rows 21:38

Use `INDEX`/`MATCH` pattern. The exact references depend on what you discover in step 0. A typical pattern (adjust after inspection):
```
=INDEX(Data!$B$21:$XX$38, MATCH($D12, Data!$A$21:$A$38, 0), MATCH(H$10, Data!$B$20:$XX$20, 0))
```
Adjust the column/row references to match the actual data layout you observed. The MATCH for the series code should search the column containing series codes in Data rows 21-38. The MATCH for the year should search the row containing years in the Data sheet.

## 2 – Net capacity headroom (H35:L40)

These 6 rows correspond to the 6 hospital clusters. Based on the three blocks:
- H12:L17 = Available Care Slots (or similar – verify from row labels)
- H19:L24 = Occupied Care Slots (or similar – verify)
- H26:L31 = Staffed Bed Capacity (or similar – verify)

The formula for each cell in H35:L40 is:
```
= (H12 - H19) / H26 * 100
```
(Adjust row references so that row 35 uses rows 12, 19, 26; row 36 uses 13, 20, 27; etc.)

## 3 – Summary statistics (H42:L47)

For each column H through L:
- Row 42 (minimum): `=MIN(H35:H40)`
- Row 43 (maximum): `=MAX(H35:H40)`
- Row 44 (median): `=MEDIAN(H35:H40)`
- Row 45 (mean): `=AVERAGE(H35:H40)`
- Row 46 (25th percentile): `=_xlfn.PERCENTILE.INC(H35:H40,0.25)`
- Row 47 (75th percentile): `=_xlfn.PERCENTILE.INC(H35:H40,0.75)`

**IMPORTANT**: Use the `_xlfn.` prefix for PERCENTILE.INC when writing with openpyxl. Previous successful execution confirmed this works. Cross-task feedback about using legacy `PERCENTILE` is noted but the prior successful run on THIS task used `_xlfn.PERCENTILE.INC`, so stick with that.

## 4 – Weighted mean (H50:L50)

For each column H through L:
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```
This computes the weighted mean of Net capacity headroom using Staffed Bed Capacity as weights.

## 5 – Save and verify

Save the workbook to `/root/output/result.xlsx`. Then:
1. Reopen it with openpyxl (data_only=False) and print a sample of formulas from each block to confirm they were written correctly.
2. Verify no extra sheets were added.
3. Verify the file exists at `/root/output/result.xlsx`.

## Critical constraints
- Do NOT add sheets, macros, VBA, external links, or helper tabs.
- Do NOT change existing formatting.
- Work only inside sheets `Task` and `Data`.
- Read the actual sheet layout FIRST before writing any formulas. Adjust all references based on what you observe.
- If the summary stat labels in rows 42-47 are in a different order than min/max/median/mean/p25/p75, match the formulas to the actual labels.

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