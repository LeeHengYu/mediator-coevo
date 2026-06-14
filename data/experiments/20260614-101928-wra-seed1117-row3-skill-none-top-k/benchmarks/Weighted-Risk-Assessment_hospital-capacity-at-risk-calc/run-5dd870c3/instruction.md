# Task Instruction

Execute the following steps in order to produce /root/output/result.xlsx.

## 0 – Preparation
```bash
mkdir -p /root/output
```
Open and inspect `/root/data/workbook.xlsx` with openpyxl (data_only=False) to confirm:
- Sheet names (`Task`, `Data`).
- The layout of sheet `Task`: column D series codes, row 10 year headers (H10:L10), yellow target ranges H12:L17, H19:L24, H26:L31, H35:L40, H42:L47, H50:L50.
- The layout of sheet `Data` rows 21–38: which row holds headers, which column holds series codes, and how years map to columns.
Print representative cells so you know exact addresses before writing any formula.

## 1 – Lookup formulas (H12:L31)
For every cell in the three 6×5 blocks (H12:L17, H19:L24, H26:L31), write an Excel formula using the **INDEX / MATCH** pattern. The formula in cell H12 should look like:

```
=INDEX(Data!$A$21:$Z$38, MATCH($D12, Data!$A$21:$A$38, 0), MATCH(H$10, Data!$A$21:$Z$21, 0))
```

Adjust the actual column/row ranges to match what you discover during inspection (the data header row for years, the series-code column, and the full data rectangle). Use:
- `$D12` (mixed reference, column absolute, row relative) for the series code.
- `H$10` (mixed reference, column relative, row absolute) for the year.
- Absolute references for the data range and lookup vectors.

This lets the formula copy correctly across the 6 rows × 5 columns of each block.

## 2 – Net capacity headroom (H35:L40)
For each of the six hospital-cluster rows and five year columns, write a formula:
```
=(H12 - H19) / H26 * 100
```
where H12 corresponds to the Available Care Slots block, H19 to Occupied Care Slots, and H26 to Staffed Bed Capacity. Adjust row references so each cluster row in H35:L40 maps to the corresponding rows in the three lookup blocks above. Use relative references so the formula copies correctly across the 6×5 block.

## 3 – Summary statistics (H42:L47)
For each year column (H through L), write column-wise formulas over the six headroom values (e.g., H35:H40):
- Row 42: `=MIN(H35:H40)` (or whichever row the Task sheet labels as minimum)
- Row 43: `=MAX(H35:H40)`
- Row 44: `=MEDIAN(H35:H40)`
- Row 45: `=AVERAGE(H35:H40)`
- Row 46: `=PERCENTILE(H35:H40, 0.25)`
- Row 47: `=PERCENTILE(H35:H40, 0.75)`

**Read the actual row labels in column B/C/D of rows 42–47 on the Task sheet to confirm which statistic goes in which row.** Map accordingly.

⚠️ CRITICAL: Use `PERCENTILE`, **not** `PERCENTILE.INC` or `PERCENTILE.EXC`. The dotted variants cause `#NAME?` errors in some engines. Similarly use `MEDIAN`, `MIN`, `MAX`, `AVERAGE` (all classic names).

## 4 – Weighted mean (H50:L50)
For each year column, write:
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```
This computes the weighted average of the Net capacity headroom percentages, weighted by Staffed Bed Capacity.

## 5 – Save
Save the workbook to `/root/output/result.xlsx`. Do **not** add sheets, macros, VBA, external links, or helper tabs. Do not alter any existing formatting.

## 6 – Validate
Re-open `/root/output/result.xlsx` with openpyxl (data_only=False) and print:
- A sample lookup formula from each block (e.g., H12, H19, H26).
- A sample headroom formula (H35).
- All six stat formulas in column H (H42:H47).
- The weighted-mean formula in H50.
Confirm none are None or contain `#NAME?` text. Confirm no extra sheets exist beyond Task and Data.

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