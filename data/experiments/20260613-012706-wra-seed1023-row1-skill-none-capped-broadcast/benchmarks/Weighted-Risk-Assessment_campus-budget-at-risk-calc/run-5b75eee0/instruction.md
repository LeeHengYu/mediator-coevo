# Task Instruction

Execute the following steps to produce `/root/output/result.xlsx` from `/root/data/workbook.xlsx`.

## 0 – Inspect the workbook
1. `mkdir -p /root/output`
2. Open `/root/data/workbook.xlsx` with openpyxl (data_only=False).
3. Print sheet names to confirm `Task` and `Data` exist.
4. Print the `Task` sheet rows 1-55, columns A-L, so you can see:
   - The series codes in column D for rows 12-17, 19-24, 26-31.
   - The years in row 10 for columns H-L.
   - The labels/structure of rows 35-50.
5. Print the `Data` sheet rows 18-40, all used columns, to see the lookup source table layout (headers, series codes, years, values).
6. Record exactly:
   - Which column on `Data` holds the series codes (the lookup key).
   - Which row on `Data` holds the year headers.
   - The data range rows 21:38 and their column span.

## 1 – Step 1: Populate H12:L17, H19:L24, H26:L31 with INDEX/MATCH formulas
For each cell in those three blocks, write an Excel formula using INDEX/MATCH that:
- Uses the series code from column D of the same row on `Task`.
- Uses the year from row 10 of the same column on `Task`.
- Looks up into the `Data` sheet rows 21:38.

Concrete pattern (adjust column letters and absolute references based on your inspection):
```
=INDEX(Data!$B$21:$XX$38, MATCH($D12, Data!$A$21:$A$38, 0), MATCH(H$10, Data!$B$20:$XX$20, 0))
```
Replace `$A$21:$A$38` with the actual series-code column, `$B$20:$XX$20` with the actual year-header row, and `$B$21:$XX$38` with the actual data block – all determined from your inspection.

Make sure every `$` anchor is correct so the formula copies across the 5 columns and 6 rows of each block.

## 2 – Step 2a: Net budget buffer in H35:L40
Identify which blocks correspond to:
- **Committed Funding** (one of H12:L17, H19:L24, H26:L31)
- **Operating Spend** (another of those blocks)
- **Approved Budget Base** (the remaining block)

Use the row labels on `Task` to determine the mapping. Then for each cell in H35:L40 write:
```
=(committed_funding_cell - operating_spend_cell) / approved_budget_base_cell * 100
```
where the three referenced cells are in the same column and corresponding department row.

## 2 – Step 2b: Summary statistics in H42:L47
For each column H through L, write formulas in rows 42-47. Check the row labels on `Task` to see which statistic goes where (min, max, median, mean, 25th percentile, 75th percentile). Use these Excel functions:
- `MIN(H35:H40)`
- `MAX(H35:H40)`
- `MEDIAN(H35:H40)`
- `AVERAGE(H35:H40)`
- `PERCENTILE(H35:H40, 0.25)`   ← use legacy `PERCENTILE`, NOT `PERCENTILE.INC`
- `PERCENTILE(H35:H40, 0.75)`   ← use legacy `PERCENTILE`, NOT `PERCENTILE.INC`

**Critical:** Do NOT use `PERCENTILE.INC` or `PERCENTILE.EXC` – they cause `#NAME?` errors in the verifier. Use only `PERCENTILE`.

## 3 – Step 3: Weighted mean in H50:L50
For each column (H through L), write:
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```
where `H26:H31` is the Approved Budget Base block (adjust if inspection shows a different block is the budget base).

## 4 – Save
Save the workbook to `/root/output/result.xlsx` with openpyxl. Do NOT change formatting, add sheets, macros, VBA, external links, or helper tabs.

## 5 – Validate
1. Re-open `/root/output/result.xlsx` with openpyxl.
2. Spot-check that cells H12, L17, H19, L24, H26, L31, H35, L40, H42, L47, H50, L50 all contain formula strings (start with `=`).
3. Confirm no cells contain `#NAME?` or literal error strings.
4. Print a few formulas from each block to confirm correctness.

If any step fails, re-read the relevant sheet region and fix before proceeding.

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