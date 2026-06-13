# Task Instruction

Execute the following steps to produce `/root/output/result.xlsx`.

## 0 – Setup
```bash
mkdir -p /root/output
pip install openpyxl
```

## 1 – Inspect the workbook
Open `/root/data/workbook.xlsx` with openpyxl (do NOT use `data_only=True`). Print:
- Sheet names.
- Sheet `Task`: cells D12:D17, D19:D24, D26:D31 (series codes), row 10 columns H–L (years), the labels in column B or C for rows 12–50, and any existing content/formatting in the yellow target ranges.
- Sheet `Data`: rows 21–38, especially the header row and the first column to understand the lookup table layout (which column holds series codes, which row holds years).

Record:
- `data_code_col`: the column number (1-based) in `Data!21:38` that contains the series codes.
- `data_year_row`: the row number in `Data` that contains the year headers.
- The column range in `Data` that holds the year values.

## 2 – Write lookup formulas in H12:L17, H19:L24, H26:L31
For each cell in these three blocks, write a formula that combines the series code from column D of the same row on `Task` with the year from row 10 of the same column on `Task`, looking up the value from `Data!$row_start:$row_end`.

Choose ONE of these patterns (INDEX/MATCH is safest for openpyxl compatibility):
```
=INDEX(Data!<data_range>, MATCH($D12, Data!<code_column>, 0), MATCH(H$10, Data!<year_row>, 0))
```
Adjust the absolute/relative references so that:
- `$D12` locks the column but lets the row vary when conceptually dragging down.
- `H$10` locks the row but lets the column vary when dragging right.
- The Data ranges are fully absolute.

IMPORTANT: Use the exact range addresses you discovered in Step 1. Do not guess.

## 3 – Write Net Budget Buffer formulas in H35:L40
The six departments correspond to the same six rows as in the earlier blocks. Identify which block is Committed Funding, which is Operating Spend, and which is Approved Budget Base from the labels in column B/C of `Task`.

For each cell (r, c) in H35:L40:
```
= (CommittedFunding_cell - OperatingSpend_cell) / ApprovedBudgetBase_cell * 100
```
where the three referenced cells are in the same column `c` and in the corresponding department row of each block.

## 4 – Write summary statistics in H42:L47
For each column H–L, calculate over the six Net-Budget-Buffer cells (e.g., H35:H40):
- Row 42: `=MIN(H35:H40)`
- Row 43: `=MAX(H35:H40)`
- Row 44: `=MEDIAN(H35:H40)`
- Row 45: `=AVERAGE(H35:H40)` (simple mean)
- Row 46: `=PERCENTILE(H35:H40,0.25)` — use `PERCENTILE` not `PERCENTILE.INC` or `PERCENTILE.EXC` to avoid #NAME? errors in openpyxl/Excel evaluation.
- Row 47: `=PERCENTILE(H35:H40,0.75)` — same note.

**CRITICAL**: Verify by reading the labels in rows 42–47 to confirm which row is MIN, MAX, MEDIAN, MEAN, 25th, 75th. Map your formulas to the actual labels, not to the order above. The order above is a guess; the labels are authoritative.

## 5 – Write weighted mean in H50:L50
For each column c in H–L:
```
=SUMPRODUCT(<NetBudgetBuffer_col>, <ApprovedBudgetBase_col>) / SUM(<ApprovedBudgetBase_col>)
```
where `<NetBudgetBuffer_col>` is e.g. `H35:H40` and `<ApprovedBudgetBase_col>` is e.g. `H26:H31`.

## 6 – Save
Save the workbook to `/root/output/result.xlsx`. Do not change formatting, do not add sheets.

## 7 – Validate
Reopen `/root/output/result.xlsx` with openpyxl (not data_only). Print the formula content of:
- A sample cell from each lookup block (e.g., H12, H19, H26)
- H35, H40 (first and last Net Budget Buffer)
- H42:H47 (all stats for column H)
- H50 (weighted mean)

Confirm no cell is empty or contains a Python value instead of a formula string. Confirm `PERCENTILE` is used (not `PERCENTILE.INC` or `PERCENTILE.EXC`) unless the labels specifically indicate otherwise.

If any formula looks wrong, fix it and re-save before finishing.

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