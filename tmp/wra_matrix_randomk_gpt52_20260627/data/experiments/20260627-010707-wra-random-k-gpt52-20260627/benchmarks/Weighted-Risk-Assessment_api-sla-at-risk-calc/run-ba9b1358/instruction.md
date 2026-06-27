# Task Instruction

You must create a Python script that reads `/root/data/workbook.xlsx`, populates formulas in the `Task` sheet, and saves the result to `/root/output/result.xlsx`. Follow these steps precisely:

## 0. Setup
```bash
mkdir -p /root/output
pip install openpyxl
```

## 1. Inspect the workbook
Open `/root/data/workbook.xlsx` with openpyxl (data_only=False). Inspect:
- Sheet `Task`: read cell D12:D17 (series codes block 1), D19:D24 (block 2), D26:D31 (block 3). Read row 10 columns H through L (years). Read the labels in column A or B for rows 35-50 to understand the layout. Print all of these values.
- Sheet `Data`: read rows 21-38 to understand the data layout (column headers, row labels). Print the first column and first row of this range.

This inspection is critical — do NOT skip it. The exact cell references, series codes, year values, and data layout must be confirmed before writing any formulas.

## 2. Write lookup formulas in H12:L17, H19:L24, H26:L31
For each cell in these three blocks, write a formula that looks up data from `Data!$A$21:$Z$38` (adjust column range based on inspection). Use the **INDEX/MATCH** pattern (non-array):
```
=INDEX(Data!$B$21:$Z$38, MATCH(D12, Data!$A$21:$A$38, 0), MATCH(H$10, Data!$B$20:$Z$20, 0))
```
Adjust the exact ranges based on your inspection of the Data sheet. Key points:
- The series code reference should use the cell in column D of the current row (e.g., D12, D13, etc.)
- The year reference should use the cell in row 10 of the current column with the row locked (e.g., H$10, I$10, etc.)
- Lock ranges on the Data sheet appropriately with $ signs
- Verify the Data sheet's row and column header positions exactly

## 3. Write Net SLA buffer formulas in H35:L40
The formula for each cell is:
```
=(HXX - HYY) / HZZ * 100
```
where HXX is the corresponding cell from the "Latency Budget Preserved" block (H12:L17), HYY is from the "Latency Budget Consumed" block (H19:L24), and HZZ is from the "Covered Request Capacity" block (H26:L31). Map row 35→rows 12,19,26; row 36→rows 13,20,27; etc.

Confirm which block is which by reading the labels in the Task sheet before writing formulas.

## 4. Write statistics formulas in H42:L47
For each column H through L, calculate over the range H35:H40 (or the corresponding column):
- Row 42: `=MIN(H35:H40)`
- Row 43: `=MAX(H35:H40)`
- Row 44: `=MEDIAN(H35:H40)`
- Row 45: `=AVERAGE(H35:H40)`
- Row 46: `=_xlfn.PERCENTILE.INC(H35:H40, 0.25)`
- Row 47: `=_xlfn.PERCENTILE.INC(H35:H40, 0.75)`

**CRITICAL**: For PERCENTILE.INC, you MUST use the `_xlfn.` prefix. This was the key success factor from the previous run. Without it, Excel/openpyxl will produce #NAME? errors. Do NOT use PERCENTILE without the prefix. Do NOT use PERCENTILE.EXC.

Verify the row-to-statistic mapping by reading the labels in column A/B/C for rows 42-47.

## 5. Write weighted mean formula in H50:L50
For each column, use SUMPRODUCT with the Net SLA buffer values (H35:H40) as values and the Covered Request Capacity block (H26:H31) as weights:
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```
Adjust column letters for I, J, K, L accordingly.

## 6. Save
Save to `/root/output/result.xlsx`. Do NOT change formatting, do NOT add sheets, macros, VBA, external links, or helper tabs.

## 7. Validate
After saving, reopen `/root/output/result.xlsx` and:
- Print the formula in cells H12, L17, H19, L24, H26, L31 to confirm lookup formulas
- Print the formula in H35 and L40 to confirm net buffer formulas
- Print the formula in H42, H46, H47 to confirm stats formulas (especially the _xlfn. prefix)
- Print the formula in H50 and L50 to confirm weighted mean
- Confirm no new sheets were added

If any formula looks wrong, fix it before finishing.

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