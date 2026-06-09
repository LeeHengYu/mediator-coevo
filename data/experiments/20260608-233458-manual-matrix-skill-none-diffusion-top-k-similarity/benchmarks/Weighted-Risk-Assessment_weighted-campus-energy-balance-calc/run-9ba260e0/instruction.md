# Task Instruction

Execute the following steps to produce `/root/output/result.xlsx`.

## 0. Inspect the workbook
```
cp /root/data/workbook.xlsx /root/output/result.xlsx
```
Open `/root/output/result.xlsx` with openpyxl (data_only=False) and inspect:
- Sheet `Task`: print rows 10-11 (to see year headers in H10:L10), rows 12-17 column D (series codes for block 1), rows 19-24 column D (series codes for block 2), rows 26-31 column D (series codes for block 3), rows 35-40 column D (campus names / labels), row 50 column D (MCEC label). Print cell values for columns D through L for all these rows.
- Sheet `Data`: print rows 21-38 fully (all non-empty columns) to understand the layout — identify where series codes live (likely column A or row 21), where years live (likely row 21 or column A), and where the numeric data matrix is.

This inspection is critical. Do NOT skip it. Print enough to see the exact cell coordinates of series codes and years on the Data sheet.

## 1. Populate lookup formulas (H12:L17, H19:L24, H26:L31)

Based on the inspection, construct INDEX/MATCH formulas. The general pattern should be:
```
=INDEX(Data!<data_range>, MATCH($D<row>, Data!<series_code_range>, 0), MATCH(H$10, Data!<year_range>, 0))
```
where:
- `<data_range>` covers the numeric block on the Data sheet (rows 21-38, columns with data)
- `<series_code_range>` is the column of series codes on Data (same rows 21-38)
- `<year_range>` is the row of years on Data
- `$D<row>` is the series code in column D of the current Task row (use absolute column reference)
- `H$10` is the year in row 10 (use absolute row reference)

Adjust references precisely based on what the inspection reveals. The formula must resolve to actual numbers when evaluated.

Write these formulas into every cell in H12:L17, H19:L24, and H26:L31 using openpyxl. Use `ws['H12'] = '=INDEX(...)'` syntax (string starting with `=`).

## 2. Net renewable balance formulas (H35:L40)

For each campus row r (35-40) and each column c (H-L), the formula is:
```
=(H12 - H19) / H26 * 100
```
where H12, H19, H26 correspond to the same campus row offset in blocks 1, 2, 3. Specifically:
- Row 35 uses rows 12, 19, 26
- Row 36 uses rows 13, 20, 27
- Row 37 uses rows 14, 21, 28
- Row 38 uses rows 15, 22, 29
- Row 39 uses rows 16, 23, 30
- Row 40 uses rows 17, 24, 31

Formula for cell in column c, row r: `=(<c><12+r-35> - <c><19+r-35>) / <c><26+r-35> * 100`

## 3. Summary statistics (H42:L47)

For each column c (H through L):
- H42: `=MIN(H35:H40)`
- H43: `=MAX(H35:H40)`
- H44: `=MEDIAN(H35:H40)`
- H45: `=AVERAGE(H35:H40)`
- H46: `=PERCENTILE(H35:H40,0.25)`
- H47: `=PERCENTILE(H35:H40,0.75)`

Verify the row labels (42-47) match what the Task sheet expects by checking column D or nearby labels. Adjust if the inspection shows different row assignments for min/max/median/mean/25th/75th.

## 4. Weighted mean (H50:L50)

For each column c (H through L):
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```

## 5. Preserve formatting

Do NOT alter any cell formatting, styles, fills, fonts, or sheet structure. Only write formula strings into the specified cells.

## 6. Save and validate

Save to `/root/output/result.xlsx`. Then re-open with openpyxl (data_only=False) and print the formula content of a sample of cells (e.g., H12, L17, H35, L40, H42, H47, H50, L50) to confirm formulas were written correctly.

Also try opening with openpyxl data_only=True (or use a formula evaluator if available) to check if values resolve, but note openpyxl cannot evaluate formulas natively — the formulas just need to be syntactically correct Excel formulas.

If a test runner exists (e.g., `test_outputs.py`), run it to validate.

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