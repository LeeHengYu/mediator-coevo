# Task Instruction

Execute the following steps to produce /root/output/result.xlsx.

## 0. Preparation
```bash
mkdir -p /root/output
pip install openpyxl
```

## 1. Inspect the workbook structure
Open `/root/data/workbook.xlsx` with openpyxl (data_only=False). Print:
- Sheet names
- The contents of `Task!D12:D17`, `Task!D19:D24`, `Task!D26:D31` (series codes)
- The contents of `Task!H10:L10` (year headers)
- The contents of `Data!A21:A38` and `Data!B21:B38` (or more columns) to understand the Data layout (which column has series codes, which row/column has years, etc.)
- The contents of `Task!H35:L40`, `Task!H42:L47`, `Task!H50:L50` (should be empty or have placeholders)
- The contents of `Task!A42:G47` to see the labels for statistics rows (min, max, median, mean, 25th pct, 75th pct)
- The contents of `Task!A35:G40` to see the cluster labels
- The contents of `Task!A50:G50` to see the weighted mean label

This inspection is critical before writing any formulas.

## 2. Write formulas — Step 1: Lookup formulas in H12:L17, H19:L24, H26:L31

For each cell in these three blocks, write an INDEX+MATCH formula that:
- Looks up the series code from column D of the same row
- Looks up the year from row 10 of the same column
- Searches in `Data!$A$21:$A$38` for the series code (row match)
- Searches in `Data!$B$20:$Z$20` (or wherever the year headers are — determine from inspection) for the year (column match)
- Returns the value from the corresponding data range

Use this pattern (adjust ranges based on inspection):
```
=INDEX(Data!$B$21:$Z$38, MATCH($D12, Data!$A$21:$A$38, 0), MATCH(H$10, Data!$B$20:$Z$20, 0))
```
Adjust the data range boundaries based on what you find in the inspection step. Make sure row references use `$` for the data range and column D reference uses `$D`, and the year row reference uses `H$10` pattern so formulas copy correctly.

## 3. Write formulas — Step 2a: Net capacity headroom (H35:L40)

For each of the 6 hospital clusters (rows 35-40) and 5 year columns (H-L):
```
=(H12 - H19) / H26 * 100
```
where row 12 = Available Care Slots, row 19 = Occupied Care Slots, row 26 = Staffed Bed Capacity for the corresponding cluster. Map cluster rows:
- Row 35 uses rows 12, 19, 26
- Row 36 uses rows 13, 20, 27
- Row 37 uses rows 14, 21, 28
- Row 38 uses rows 15, 22, 29
- Row 39 uses rows 16, 23, 30
- Row 40 uses rows 17, 24, 31

## 4. Write formulas — Step 2b: Column-wise statistics (H42:L47)

Based on the labels found in inspection (rows 42-47), write the appropriate formulas. The expected order based on the task description is: minimum, maximum, median, simple mean, 25th percentile, 75th percentile.

**CRITICAL**: For percentile functions, use `PERCENTILE` (not `PERCENTILE.INC` or `PERCENTILE.EXC`). The previous attempt failed because the evaluator engine did not recognize the function name. Use the plain `PERCENTILE` function.

For column H (and similarly for I through L):
- Row 42 (min): `=MIN(H35:H40)`
- Row 43 (max): `=MAX(H35:H40)`
- Row 44 (median): `=MEDIAN(H35:H40)`
- Row 45 (mean): `=AVERAGE(H35:H40)`
- Row 46 (25th pct): `=PERCENTILE(H35:H40,0.25)`
- Row 47 (75th pct): `=PERCENTILE(H35:H40,0.75)`

Verify the row-to-statistic mapping matches the actual labels found during inspection. If labels differ from the assumed order, adjust accordingly.

## 5. Write formulas — Step 3: Weighted mean (H50:L50)

For each column H through L:
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```

## 6. Save
Save the workbook to `/root/output/result.xlsx`. Do NOT use data_only mode. Preserve all existing formatting.

## 7. Validate
Reopen the saved file and print:
- A sample of lookup formulas (e.g., H12, L17, H26)
- All formulas in H35:H40
- All formulas in H42:H47
- The formula in H50
- Confirm no cells contain #NAME? by checking if any formula string contains obvious typos
- Confirm the file exists and has reasonable size

## Key Warnings
- Use `PERCENTILE` not `PERCENTILE.INC` or `PERCENTILE.EXC` — this was the cause of the previous failure.
- Do not add any new sheets, macros, VBA, or external links.
- Do not change any existing formatting.
- Make sure all cell references in formulas are correct based on the actual workbook layout discovered during inspection.

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