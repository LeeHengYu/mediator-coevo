# Task Instruction

Execute the following steps to produce `/root/output/result.xlsx` from `/root/data/workbook.xlsx`.

## Phase 0 – Investigation
1. Open `/root/data/workbook.xlsx` with openpyxl (data_only=False) and inspect:
   - Sheet `Task`: read the series codes in column D for rows 12-17, 19-24, 26-31 (these are the three blocks). Read the years in row 10 for columns H-L. Note the exact text/values.
   - Sheet `Data`: read rows 21-38 to understand the data layout. Determine which row contains headers (series codes) and which column contains year headers. Identify orientation: are series codes in a column and years across a row, or vice versa? Note the exact row/column positions.
   - Print all discovered values so you can construct correct formulas.

2. Also inspect rows 35-40 on `Task` to understand the labels (department names) and which rows correspond to Committed Funding, Operating Spend, and Approved Budget Base blocks. Inspect rows 42-47 for the stat labels (min, max, median, mean, 25th percentile, 75th percentile). Inspect row 50 for the weighted mean row.

## Phase 1 – Lookup Formulas (H12:L17, H19:L24, H26:L31)
For each cell in the three yellow blocks, write an `INDEX(MATCH,MATCH)` formula that:
- Uses the series code from column D of the same row as one lookup key.
- Uses the year from row 10 of the same column as the other lookup key.
- Looks up in the Data sheet rows 21:38.
- Use absolute references for the data range and relative/mixed references for the lookup keys so formulas are consistent.

Concrete pattern (adjust ranges based on Phase 0 findings):
```
=INDEX(Data!$B$22:$Z$38, MATCH($D12, Data!$A$22:$A$38, 0), MATCH(H$10, Data!$B$21:$Z$21, 0))
```
Adjust `$B$22:$Z$38`, `$A$22:$A$38`, `$B$21:$Z$21` to match the actual data layout discovered in Phase 0. The key point: the MATCH for series codes searches the column containing series codes, the MATCH for years searches the row containing years, and the INDEX range is the data body (excluding headers).

## Phase 2 – Net Budget Buffer (H35:L40)
The three blocks correspond to:
- Block 1 (rows 12-17): one metric (e.g., Committed Funding)
- Block 2 (rows 19-24): another metric (e.g., Operating Spend)  
- Block 3 (rows 26-31): another metric (e.g., Approved Budget Base)

Check the labels in column B/C for rows 12, 19, 26 to identify which block is which. Then for H35:L40:
```
=(H12 - H19) / H26 * 100
```
(Adjust row references if the block order differs: use Committed Funding block row - Operating Spend block row, divided by Approved Budget Base block row, times 100. Each of the 6 rows in H35:L40 corresponds to the same department row offset within each block.)

## Phase 3 – Summary Statistics (H42:L47)
For each column H through L, in rows 42-47 place:
- Row 42: `=MIN(H35:H40)`
- Row 43: `=MAX(H35:H40)`
- Row 44: `=MEDIAN(H35:H40)`
- Row 45: `=AVERAGE(H35:H40)`
- Row 46: `=PERCENTILE(H35:H40, 0.25)`
- Row 47: `=PERCENTILE(H35:H40, 0.75)`

Check the actual labels in column B/C/D for rows 42-47 to match the correct statistic to the correct row. Adjust row assignments accordingly.

## Phase 4 – Weighted Mean (H50:L50)
For each column (e.g., H):
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```
This computes the weighted mean of the Net Budget Buffer percentages using Approved Budget Base as weights.

## Phase 5 – Save
- Create `/root/output/` directory if needed.
- Save to `/root/output/result.xlsx`.
- Do NOT change any formatting, do NOT add sheets, macros, VBA, or helper tabs.

## Phase 6 – Validation
- Reopen the saved file with openpyxl (data_only=False) and spot-check:
  - That cells H12, L17, H19, L24, H26, L31 contain formula strings (not None or bare values).
  - That cells H35, L40 contain formula strings.
  - That cells H42:L47 contain formula strings.
  - That cells H50:L50 contain formula strings.
- Print a few formula samples to confirm correctness.
- If any cell is empty or has no formula, fix it before finishing.

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