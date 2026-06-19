# Task Instruction

Execute the following steps carefully to produce `/root/output/result.xlsx`.

## 0 – Preparation

```bash
mkdir -p /root/output
pip install openpyxl
```

## 1 – Inspect the workbook

Open `/root/data/workbook.xlsx` with openpyxl (data_only=False) and inspect:
- Sheet `Task`: read row 10 (H10:L10) to see the year headers. Read column D rows 12-17, 19-24, 26-31 to see the series codes. Read row 35-40 column D or B to see department identifiers. Read any labels in column A-G for rows 42-47 (stat labels) and row 50 (Campus Budget Council). Print all of this.
- Sheet `Data`: read rows 21-38 to understand the data layout – specifically what is in column A (or B) and which row/column holds years. Print the first 10 columns of rows 20-38 so you see headers and data.

Do NOT proceed until you have printed and understood the layout.

## 2 – Write formulas with openpyxl

Use `openpyxl` to open the workbook (no data_only), and write formulas into the cells. Important rules:
- Use `load_workbook('/root/data/workbook.xlsx')` without data_only so formulas are preserved.
- Write formulas as strings starting with `=`.
- Do NOT add sheets, do NOT change formatting, do NOT add VBA/macros.

### Step 1 – Lookup formulas in H12:L17, H19:L24, H26:L31

For each cell in these three blocks, write an INDEX/MATCH formula that:
- Looks up the series code from column D of the same row
- Looks up the year from row 10 of the same column
- Searches in `Data!$A$21:$A$38` (or whichever column holds series codes) for the row match
- Searches in the header row of Data sheet for the column match
- Returns the value from the data range on sheet Data rows 21:38

The exact formula pattern (adjust column references based on what you see in step 1):
```
=INDEX(Data!$A$21:$Z$38, MATCH($D12, Data!$A$21:$A$38, 0), MATCH(H$10, Data!$A$20:$Z$20, 0))
```
Adjust the range letters/numbers based on actual data layout. The key: use `$D12` (mixed ref – column absolute, row relative) and `H$10` (column relative, row absolute).

Make sure to verify the Data sheet layout first. If series codes are in column B, adjust accordingly. If the header row with years is row 20, use that.

### Step 2 – Net budget buffer in H35:L40

The formula is: `(Committed Funding - Operating Spend) / Approved Budget Base * 100`

From the layout:
- Committed Funding block = H12:L17
- Operating Spend block = H19:L24  
- Approved Budget Base block = H26:L31

So for cell H35: `=(H12-H19)/H26*100`
For H36: `=(H13-H20)/H27*100` etc.
Adjust row offsets so each department row in 35-40 maps to the corresponding rows in the three blocks above.

### Step 2 continued – Summary statistics in H42:L47

For each column (H through L):
- Row 42 (MIN): `=MIN(H35:H40)`
- Row 43 (MAX): `=MAX(H35:H40)`
- Row 44 (MEDIAN): `=MEDIAN(H35:H40)`
- Row 45 (MEAN): `=AVERAGE(H35:H40)`
- Row 46 (25th percentile): `=PERCENTILE.INC(H35:H40,0.25)`
- Row 47 (75th percentile): `=PERCENTILE.INC(H35:H40,0.75)`

**CRITICAL**: The previous run failed with #NAME? errors on percentile rows. This is likely because openpyxl or the test evaluator computes formulas using a limited function set. To be safe, try BOTH approaches:
1. First try `=PERCENTILE.INC(H35:H40,0.25)` — this is the standard Excel 2010+ function.
2. If the test evaluator doesn't recognize dotted function names, also try `=PERCENTILE(H35:H40,0.25)` — the legacy Excel function.

**Decision**: Use the legacy `PERCENTILE` function (without `.INC`) since the test environment likely uses a formula evaluator that doesn't support dotted names:
- Row 46: `=PERCENTILE(H35:H40,0.25)`  
- Row 47: `=PERCENTILE(H35:H40,0.75)`

### Step 3 – Weighted mean in H50:L50

For each column (H through L):
`=SUMPRODUCT(H35:H40,H26:H31)/SUM(H26:H31)`

This computes the weighted mean using Net budget buffer percentages as values and Approved Budget Base as weights.

## 3 – Verify row-label mapping

Before writing formulas, confirm by reading the Task sheet:
- What labels are in rows 42-47 (check columns A-G)? Map MIN/MAX/MEDIAN/MEAN/25th/75th to the correct rows.
- What label is in row 50? Confirm it says Campus Budget Council.
- What are the department names in rows 35-40 and do they correspond 1:1 with rows 12-17?

Adjust row numbers if the labels don't match my assumptions.

## 4 – Save

Save the workbook to `/root/output/result.xlsx`.

## 5 – Validate

Reopen the saved file with openpyxl (data_only=False) and print:
- The formula in H12, H19, H26, H35, H42, H46, H47, H50
- Confirm none are None/empty
- Confirm H46 and H47 use `PERCENTILE` (not `PERCENTILE.INC`)

Then if there is a test script at `/root/test_output.py` or similar, run it:
```bash
cd /root && python -m pytest test_output.py -v 2>&1 | head -80
```

If tests fail, read the error messages carefully and fix. Common issues:
- Wrong row/column references in Data sheet
- Off-by-one in department row mapping
- Function name not recognized (use legacy names without dots)
- SUMPRODUCT formula incorrect for weighted mean

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