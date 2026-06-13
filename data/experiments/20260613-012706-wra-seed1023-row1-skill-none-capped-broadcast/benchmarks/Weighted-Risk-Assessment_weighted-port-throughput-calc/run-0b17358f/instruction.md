# Task Instruction

Execute the following steps to produce /root/output/result.xlsx from /root/data/workbook.xlsx.

## 0 – Inspect the workbook
1. `mkdir -p /root/output`
2. Open `/root/data/workbook.xlsx` with openpyxl (data_only=False) and print:
   - Sheet names.
   - Sheet `Task`: values/formulas in D12:D17, D19:D24, D26:D31, D35:D40, row 10 (H10:L10), H42:H47 labels if any, H50 label area, and any existing content in the yellow target ranges.
   - Sheet `Data`: rows 21–38 (all columns with data) so you understand the lookup table layout — especially which column holds the series codes and which row holds the years.
3. Print the structure clearly before writing any formulas.

## 1 – Step 1: Lookup formulas in H12:L17, H19:L24, H26:L31
For every cell in these three blocks, write an INDEX/MATCH formula that:
- Looks up the series code from column D of the current row against the series-code column in Data!rows 21:38.
- Looks up the year from row 10 of the current column against the year row in Data!rows 21:38.
- Returns the intersection value.

Use absolute references for the Data ranges and the year row on Task, and relative references for the series code column D so the formula can be filled across the block. Confirm the exact Data sheet layout (which column has codes, which row has years) from your inspection before writing formulas.

IMPORTANT: Use `$` anchoring carefully:
- The series code reference should lock the column ($D) but keep the row relative.
- The year reference should lock the row ($10) but keep the column relative.
- All Data sheet range references should be fully absolute.

## 2 – Step 2: Net container flow (H35:L40) and statistics (H42:L47)
For H35:L40, write formulas implementing:
`(Loaded Containers Inbound - Loaded Containers Outbound) / Terminal Throughput Capacity * 100`

Use cell references pointing to the Step 1 lookup results in the three blocks (H12:L17 for one metric, H19:L24 for another, H26:L31 for the third). Identify which block corresponds to which metric by checking the labels in column D or nearby.

For H42:L47, write column-wise statistical formulas over H35:L40:
- MIN
- MAX
- MEDIAN
- AVERAGE (simple mean)
- PERCENTILE (or PERCENTILE.INC) with 0.25 for 25th percentile
- PERCENTILE (or PERCENTILE.INC) with 0.75 for 75th percentile

CRITICAL: Check the labels in column D/E/F/G for rows 42–47 to determine the correct order of these statistics. Match each statistic to its labeled row. Do NOT assume the order — read it from the sheet.

For PERCENTILE, use `PERCENTILE.INC` (or `PERCENTILE` — both are valid in Excel). Do NOT use `PERCENTILE.EXC` unless the label specifically says so. The failed hospital-capacity task got #NAME? errors in statistics rows — likely from using a function name Excel didn't recognize or from incorrect syntax. Double-check every formula string.

## 3 – Step 3: Weighted mean in H50:L50
Write a SUMPRODUCT formula for each column:
`=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)`

This computes the weighted mean of net-container-flow percentages weighted by Terminal Throughput Capacity. Adjust the row references if the blocks are in different positions than expected.

## 4 – Save and verify
1. Save the workbook to `/root/output/result.xlsx`.
2. Re-open the saved file with openpyxl (data_only=False) and print all formulas in the target ranges to confirm they are correctly written.
3. Do NOT add any new sheets, macros, VBA, external links, or helper tabs.
4. Do NOT change any existing formatting, values, or structure outside the target cells.

## Key pitfalls to avoid
- #NAME? errors from misspelled function names (use PERCENTILE.INC not PERCENTILE_INC, use AVERAGE not AVG).
- Wrong absolute/relative references causing formulas to point to wrong cells when filled across columns/rows.
- Incorrect identification of which Data rows/columns hold codes vs years.
- Writing Python-computed values instead of Excel formulas — all target cells must contain formula strings.

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