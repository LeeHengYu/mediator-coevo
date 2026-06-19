# Task Instruction

Execute the following steps to produce /root/output/result.xlsx from /root/data/workbook.xlsx.

## Step 0 – Inspect the workbook
1. `mkdir -p /root/output`
2. Open `/root/data/workbook.xlsx` with openpyxl (data_only=False) and print:
   - Sheet names.
   - On sheet `Task`: cells D12:D17, D19:D24, D26:D31 (series codes), row 10 columns H–L (years), cells H35:L40 labels if any, H42:L47 labels, H50:L50 label, and any existing content/formatting notes.
   - On sheet `Data`: row 20 or 21 headers, rows 21–38 column A–Z (or however wide the data goes) so you can see the layout of the source data (series codes in which column, years in which row, values grid).
3. Print all of this before writing any formulas.

## Step 1 – Lookup formulas in H12:L17, H19:L24, H26:L31
For every cell in these three 6×5 blocks, write an Excel formula that uses INDEX-MATCH-MATCH (or an equivalent approved pattern) to look up the value from sheet `Data` rows 21:38. The two keys are:
  - The series code from column D of the same row on sheet `Task`.
  - The year from row 10 of the same column on sheet `Task`.

Use absolute references for the Data range and mixed references so formulas are correct per-cell. Example pattern (adjust ranges after inspection):
```
=INDEX(Data!$B$21:$Z$38, MATCH($D12, Data!$A$21:$A$38, 0), MATCH(H$10, Data!$B$20:$Z$20, 0))
```
Adjust column/row references based on what you see in Step 0.

## Step 2 – Net renewable balance (H35:L40)
For each of the 6 campus rows and 5 year columns, write a formula:
```
=(H12 - H19) / H26 * 100
```
where H12 corresponds to Renewable Generation (rows 12–17), H19 to Grid Consumption (rows 19–24), and H26 to Baseline Energy Demand (rows 26–31). Use matching row offsets for each campus.

## Step 3 – Summary statistics (H42:L47)
In each column H–L, write:
- Row 42: `=MIN(H35:H40)`
- Row 43: `=MAX(H35:H40)`
- Row 44: `=MEDIAN(H35:H40)`
- Row 45: `=AVERAGE(H35:H40)`
- Row 46: `=PERCENTILE(H35:H40, 0.25)`
- Row 47: `=PERCENTILE(H35:H40, 0.75)`

Check the labels in column D/E/F/G of rows 42–47 to confirm which row is which statistic, and map accordingly. Do NOT assume the order above; read the labels first.

## Step 4 – Weighted mean (H50:L50)
For each column H–L:
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```

## Step 5 – Save
Save the workbook to `/root/output/result.xlsx`. Do NOT change formatting, do NOT add sheets, macros, VBA, external links, or helper tabs.

## Step 6 – Validate
Re-open `/root/output/result.xlsx` with openpyxl (data_only=False) and print:
- A sample of formulas from H12, L17, H19, L24, H26, L31 to confirm lookup formulas exist.
- Formulas from H35, L40 to confirm net balance formulas.
- Formulas from H42:H47 to confirm stats.
- Formula from H50 to confirm weighted mean.
- Confirm no cells in the required ranges are None.

## Key Failure Mode to Avoid
From the hospital-bedflow failure: cells returned None because formulas were never written. Make sure every cell in every required range has a formula string, not None. After writing, re-read to verify.

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