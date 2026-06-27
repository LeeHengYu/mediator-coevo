# Task Instruction

Execute the following steps exactly to produce /root/output/result.xlsx.

## 0 – Preparation
```bash
mkdir -p /root/output
pip install openpyxl
```

## 1 – Inspect the workbook
Open `/root/data/workbook.xlsx` with openpyxl and inspect:
- Sheet `Task`: read row 10 (the year headers in H10:L10), column D for rows 12-17, 19-24, 26-31 to get the series codes, rows 35-40 column D for port names, row 50 for the CPA label.
- Sheet `Data`: read rows 21-38 to understand the layout (which column holds the series code, which row holds years, and where the numeric data lives).
Print all of this so you understand the exact structure before writing any formulas.

## 2 – Understand the Data sheet layout
Determine:
- Which column on Data contains the series codes (likely column A or B).
- Which row on Data contains the year headers.
- The column number offset needed for MATCH on the series code.
- The row number offset needed for MATCH on the year.
Print the first few rows/columns of the Data range (rows 21-38) so you can verify.

## 3 – Write lookup formulas in H12:L17, H19:L24, H26:L31
Use openpyxl to write Excel formulas (as strings starting with '=') into each yellow cell. Use the INDEX/MATCH pattern:
```
=INDEX(Data!$A$21:$Z$38, MATCH($D12,Data!$A$21:$A$38,0), MATCH(H$10,Data!$A$21:$Z$21,0))
```
Adjust the exact range references based on what you discovered in steps 1-2. The key points:
- The series code anchor is `$D12` (column D of the current row, with $ on D so it doesn't shift horizontally).
- The year anchor is `H$10` (row 10 of the current column, with $ on 10 so it doesn't shift vertically).
- Use absolute references ($) on the Data ranges.
- Make sure the MATCH ranges cover the correct column for series codes and the correct row for years within Data rows 21:38.

Repeat for all three blocks (H12:L17, H19:L24, H26:L31), adjusting row references naturally.

## 4 – Write Net Container Flow formulas in H35:L40
For each of the 6 ports (rows 35-40) and each year column (H-L), write:
```
=(H12-H19)/H26*100
```
where H12 is the Loaded Containers Inbound cell, H19 is the Loaded Containers Outbound cell, and H26 is the Terminal Throughput Capacity cell for the same port and year. Adjust row references for each port:
- Row 35 uses rows 12, 19, 26
- Row 36 uses rows 13, 20, 27
- Row 37 uses rows 14, 21, 28
- Row 38 uses rows 15, 22, 29
- Row 39 uses rows 16, 23, 30
- Row 40 uses rows 17, 24, 31

## 5 – Write statistics formulas in H42:L47
For each year column (H through L):
- Row 42 (MIN): `=MIN(H35:H40)`
- Row 43 (MAX): `=MAX(H35:H40)`
- Row 44 (MEDIAN): `=MEDIAN(H35:H40)`
- Row 45 (MEAN): `=AVERAGE(H35:H40)`
- Row 46 (25th percentile): `=PERCENTILE(H35:H40,0.25)`
- Row 47 (75th percentile): `=PERCENTILE(H35:H40,0.75)`

Use `PERCENTILE` (not `PERCENTILE.INC` or `PERCENTILE.EXC`) for maximum compatibility.

## 6 – Write weighted mean formula in H50:L50
For each year column (H through L):
```
=SUMPRODUCT(H35:H40,H26:H31)/SUM(H26:H31)
```
This computes the weighted mean of the Net Container Flow percentages using Terminal Throughput Capacity as weights.

## 7 – Save
Save the workbook to `/root/output/result.xlsx`. Do NOT use `data_only=True` when loading. Keep all formatting intact.

## 8 – Verify
Re-open `/root/output/result.xlsx` with openpyxl and:
- Print cells H12, H19, H26 to confirm they contain formula strings (starting with '=').
- Print cells H35, H42, H50 to confirm they contain formula strings.
- Confirm no cell in the target ranges is None or a bare number (they should all be formula strings).

## Critical reminders
- Use `PERCENTILE` not `PERCENTILE.INC`/`PERCENTILE.EXC`.
- All formulas must be Excel formula strings written via openpyxl, NOT Python-computed values.
- Do NOT add any new sheets, macros, VBA, external links, or helper tabs.
- Do NOT alter existing formatting.
- Adjust all range references based on your actual inspection of the Data sheet layout in steps 1-2. Do not assume column positions without checking.

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