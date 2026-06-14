# Task Instruction

Execute the following steps to produce /root/output/result.xlsx from /root/data/workbook.xlsx.

## 0 – Inspect the workbook
1. `mkdir -p /root/output`
2. Open `/root/data/workbook.xlsx` with openpyxl (data_only=False) and print:
   - Sheet names.
   - Task sheet: values in D12:D17, D19:D24, D26:D31 (series codes), row 10 columns H–L (years), H35:L40 labels if any, H42:L47 labels, H50:L50 label.
   - Data sheet: row 20 or 21 headers, column A values rows 21–38 (series codes in the lookup table), row 1 or the header row that contains years, and a sample cell to understand orientation (rows = series, columns = years, or transposed).
3. Print the exact row/column layout of the Data sheet so we know whether series codes are in a column and years across a row (VLOOKUP-style) or transposed.

## 1 – Populate lookup blocks (H12:L17, H19:L24, H26:L31)
Using openpyxl, write **formula strings** (not computed values) into every cell in the three 6-row × 5-column blocks.

Based on the previous successful run, use an INDEX/MATCH/MATCH pattern. The exact pattern depends on the Data sheet layout discovered in Step 0, but the template is:

```
=INDEX(Data!$B$21:$<lastcol>$38, MATCH($D12, Data!$A$21:$A$38, 0), MATCH(H$10, Data!$B$20:$<lastcol>$20, 0))
```

Adjust `$B$21`, `$A$21:$A$38`, `$B$20`, and `<lastcol>` to match the actual data range found in Step 0. The key contract:
- Row lookup key = the series code in column D of the current row (use `$D12`, `$D13`, … with the column locked).
- Column lookup key = the year in row 10 of the current column (use `H$10`, `I$10`, … with the row locked).
- Every cell gets its own correctly-addressed formula (row references shift down, column references shift right).

## 2 – Net production slack (H35:L40)
Write formulas that compute:
```
=(H12 - H19) / H26 * 100
```
where H12 is the Finished Output block, H19 is the Scrap And Rework block, and H26 is the Rated Production Capacity block. Adjust row references per plant row (row 35 uses rows 12, 19, 26; row 36 uses 13, 20, 27; etc.). Column shifts naturally (H→I→J→K→L).

Verify by inspection: row offsets between blocks are consistent (each block is 6 rows: 12–17, 19–24, 26–31 → gaps of 7 rows between block starts). So for row r in 35..40:
- Finished Output row = r - 23  (35→12, 36→13, …, 40→17)
- Scrap row = r - 16  (35→19, …, 40→24)
- Capacity row = r - 9  (35→26, …, 40→31)

## 3 – Summary statistics (H42:L47)
For each column c in {H, I, J, K, L}, write:
- Row 42 (min):    `=MIN(c35:c40)`
- Row 43 (max):    `=MAX(c35:c40)`
- Row 44 (median): `=MEDIAN(c35:c40)`
- Row 45 (mean):   `=AVERAGE(c35:c40)`
- Row 46 (25th):   `=PERCENTILE(c35:c40,0.25)`
- Row 47 (75th):   `=PERCENTILE(c35:c40,0.75)`

Check the labels in column D/E/F/G of rows 42–47 to confirm the ordering (min, max, median, mean, 25th, 75th). If the order differs, match the formulas to the labels.

## 4 – Weighted mean (H50:L50)
For each column c in {H, I, J, K, L}:
```
=SUMPRODUCT(c35:c40, c26:c31) / SUM(c26:c31)
```
This computes the weighted average of Net production slack using Rated Production Capacity as weights.

## 5 – Save
Save the workbook to `/root/output/result.xlsx`. Do NOT change any existing formatting, do not add sheets, macros, VBA, external links, or helper tabs.

## 6 – Validate
1. Re-open `/root/output/result.xlsx` with openpyxl (data_only=False).
2. Print formulas in H12, L17, H19, L24, H26, L31, H35, L40, H42, H47, H50, L50.
3. Confirm every yellow-range cell contains a formula string (starts with '='), not None or a literal value.
4. Confirm no new sheets were added.
5. If any cell is None or a bare value, diagnose and fix before finishing.

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