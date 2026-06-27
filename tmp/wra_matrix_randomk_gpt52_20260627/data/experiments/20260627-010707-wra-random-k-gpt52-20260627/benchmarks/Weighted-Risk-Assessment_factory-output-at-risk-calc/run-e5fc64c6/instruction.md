# Task Instruction

Execute the following steps in order.

## 1 – Inspect the workbook
```
cp /root/data/workbook.xlsx /root/data/workbook_backup.xlsx
```
Open `/root/data/workbook.xlsx` with openpyxl (data_only=False). Inspect:
- Sheet `Task`: read the series codes in column D for rows 12-17, 19-24, 26-31 (these are the three blocks). Read the year headers in H10:L10. Read any labels in column A or B for those row ranges so you know which block is Finished Output, Scrap And Rework, and Rated Production Capacity. Also read row 35-40 labels (Net production slack plants), row 42-47 labels (min/max/median/mean/percentiles), and row 50 label.
- Sheet `Data`: read the header row and rows 21-38 to understand the layout (which column has the series code, which columns/rows have years, and how data is arranged — is it vertical with series codes in a column, or horizontal?).

Print all of this information so you can design correct formulas.

## 2 – Populate lookup formulas in H12:L17, H19:L24, H26:L31
For each cell in those three blocks, write an Excel formula using INDEX/MATCH that:
- Uses the series code from column D of that row (e.g., `$D12` with the column locked)
- Uses the year from row 10 of that column (e.g., `H$10` with the row locked)
- Looks up in the Data sheet rows 21:38

Based on the Data sheet layout, construct the INDEX/MATCH correctly. If Data has series codes in one column and years across a header row, use a two-dimensional INDEX with two MATCH calls. Example pattern:
```
=INDEX(Data!$B$21:$Z$38, MATCH($D12, Data!$A$21:$A$38, 0), MATCH(H$10, Data!$B$20:$Z$20, 0))
```
Adjust the exact ranges after inspecting the actual Data sheet layout. Make sure references are correct.

## 3 – Net Production Slack in H35:L40
For each of the 6 plants (rows 35-40) and each of the 5 year columns (H-L), write a formula:
```
=(Hxx - Hyy) / Hzz * 100
```
where xx is the corresponding Finished Output row (12-17), yy is the Scrap And Rework row (19-24), and zz is the Rated Production Capacity row (26-31). The row offsets must match: row 35 uses rows 12, 19, 26; row 36 uses rows 13, 20, 27; etc.

## 4 – Summary statistics in H42:L47
For each column H through L, write formulas in rows 42-47. Read the labels in column A/B/C to determine the order (min, max, median, mean, 25th percentile, 75th percentile). Use these Excel functions:
- MIN(H35:H40)
- MAX(H35:H40)
- MEDIAN(H35:H40)
- AVERAGE(H35:H40)
- PERCENTILE(H35:H40, 0.25)  — use PERCENTILE, NOT PERCENTILE.INC or PERCENTILE.EXC (to avoid #NAME? errors in the verifier)
- PERCENTILE(H35:H40, 0.75)

**CRITICAL**: Use `PERCENTILE` not `PERCENTILE.INC` or `PERCENTILE.EXC`. Use `AVERAGE` not `MEAN`. These dotted function names cause #NAME? errors. Similarly use `MEDIAN`, `MIN`, `MAX` — the classic non-dotted versions.

## 5 – Weighted mean in H50:L50
For each column H through L, write:
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```
This computes the weighted mean of Net Production Slack using Rated Production Capacity as weights.

## 6 – Save
Save the workbook to `/root/output/result.xlsx` (create the output directory if needed). Do NOT change any formatting, do not add sheets.

## 7 – Verify
Re-open `/root/output/result.xlsx` with openpyxl (data_only=False). Spot-check:
- A few lookup formulas in the three blocks to confirm they are present and well-formed.
- The Net Production Slack formulas reference the correct rows.
- The statistics formulas use non-dotted function names.
- The SUMPRODUCT formulas in row 50.
Print the formulas for a sample of cells from each section.

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