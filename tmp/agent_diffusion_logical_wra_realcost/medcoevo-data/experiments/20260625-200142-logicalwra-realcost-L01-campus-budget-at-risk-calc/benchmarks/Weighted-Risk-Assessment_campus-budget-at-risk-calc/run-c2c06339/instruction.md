# Task Instruction

## Objective

Populate formula cells in `/root/data/workbook.xlsx` sheet `Task` and save the result to `/root/output/result.xlsx`. Do NOT add sheets, macros, VBA, external links, or helper tabs. Preserve all existing formatting.

## Detailed Steps

### 0. Setup & Inspection

```bash
mkdir -p /root/output
```

1. Open `/root/data/workbook.xlsx` with openpyxl (NOT data_only) and inspect:
   - Sheet `Task`: read row 10 to find the year headers in columns H through L (e.g., H10, I10, J10, K10, L10).
   - Sheet `Task`: read column D rows 12–17, 19–24, 26–31 to find the series codes for each block.
   - Sheet `Data`: read row 20 or 21 header area and rows 21–38 to understand the data layout. Specifically identify:
     - Which column contains the series/lookup codes (likely column D).
     - Which row contains the year headers that match Task row 10.
     - The data range for values (likely H21:L38 or similar).
   - Print all of this so you understand the exact structure before writing any formulas.

### 1. Populate Lookup Formulas (H12:L17, H19:L24, H26:L31)

For each cell in these three 6×5 blocks, write an `INDEX/MATCH` formula that:
- Looks up the series code from column D of the current row in `Data!$D$21:$D$38`
- Looks up the year from row 10 of the current column in `Data!$H$20:$L$20` (or wherever the year headers are on the Data sheet — confirm by inspection)
- Returns the corresponding value from `Data!$H$21:$L$38`

The formula pattern for cell H12 would be something like:
```
=INDEX(Data!$H$21:$L$38,MATCH($D12,Data!$D$21:$D$38,0),MATCH(H$10,Data!$H$20:$L$20,0))
```
Adjust the Data sheet row references based on what you actually find during inspection. The key constraints:
- Column reference for series code must be absolute on column D, relative on row ($D12)
- Row reference for year must be absolute on row 10, relative on column (H$10)
- Data ranges must be fully absolute so the formula can be replicated across the block

Apply the same formula pattern to all three blocks (H12:L17, H19:L24, H26:L31), adjusting only the row reference in $D{row}.

### 2. Net Budget Buffer (H35:L40)

The three lookup blocks correspond to three data series. Based on the task description and typical WRA layout:
- Block 1 (H12:L17): one metric (e.g., Committed Funding)
- Block 2 (H19:L24): another metric (e.g., Operating Spend)  
- Block 3 (H26:L31): another metric (e.g., Approved Budget Base)

**Inspect column B or C rows 11, 18, 25 (or nearby) to find the block labels** to determine which block is which. Then map them to the formula:

```
Net budget buffer = (Committed Funding - Operating Spend) / Approved Budget Base * 100
```

For each cell in H35:L40, write a formula referencing the corresponding cells in the three blocks. For example, if Block 1 = Committed Funding (rows 12–17), Block 2 = Operating Spend (rows 19–24), Block 3 = Approved Budget Base (rows 26–31), then H35 would be:
```
=(H12-H19)/H26*100
```
Adjust the block mapping based on actual inspection.

### 3. Summary Statistics (H42:L47)

For each column H through L, compute column-wise statistics over the 6 values in rows 35–40:
- Row 42: `=MIN(H35:H40)` (or `=MIN(H$35:H$40)`)
- Row 43: `=MAX(H35:H40)`
- Row 44: `=MEDIAN(H35:H40)`
- Row 45: `=AVERAGE(H35:H40)`
- Row 46: `=PERCENTILE(H35:H40,0.25)`
- Row 47: `=PERCENTILE(H35:H40,0.75)`

**Inspect the labels in column B/C/D for rows 42–47 to confirm which row gets which statistic.** Match the label to the function. Do not assume the order above — use the actual labels.

### 4. Weighted Mean (H50:L50)

For each column H through L:
```
=SUMPRODUCT(H35:H40,H26:H31)/SUM(H26:H31)
```
This computes the weighted mean of Net budget buffer values using Approved Budget Base as weights. Again, confirm H26:H31 is indeed the Approved Budget Base block.

### 5. Cache Numeric Values in XML

This is CRITICAL for the verifier. After writing all formulas with openpyxl:

1. Save the workbook to `/root/output/result.xlsx`.
2. Re-open it, and for every cell you wrote a formula into, **compute the expected numeric value using Python** (by reading the Data sheet values and doing the math yourself).
3. Use openpyxl to set the `value` attribute of each cell's internal XML `<v>` element. The way to do this with openpyxl:
   - After writing the formula string to `cell.value`, also manually compute the number.
   - Then, after saving, unzip the xlsx, parse the `xl/worksheets/sheet1.xml` (or whichever sheet is Task), find each cell element, and inject/update the `<v>` tag with the computed numeric value.
   - Re-zip and save.

**Alternative simpler approach**: Use openpyxl's internal API. After setting `cell.value = '=FORMULA...'`, you can set `cell._value` to the formula but also inject a cached value. The cleanest way:
   - Write all formulas and save.
   - Unzip the xlsx.
   - Parse the Task sheet XML.
   - For each formula cell, add `<v>COMPUTED_NUMBER</v>` inside the `<c>` element.
   - Rezip.

To compute the values:
- Read Data sheet values into a Python dict keyed by (series_code, year).
- For lookup cells: just look up the value.
- For net budget buffer: apply the formula.
- For statistics: compute min, max, median, mean, percentile.
- For weighted mean: compute SUMPRODUCT/SUM.

### 6. Validation

1. Reopen `/root/output/result.xlsx` with openpyxl (data_only=True) and verify that cells in H12, H35, H42, H50 have numeric values (not None).
2. Reopen with data_only=False and verify formulas are present.
3. Print a sample of values to confirm correctness.

## Key Warnings

- **Do NOT skip the cached value injection step.** The verifier likely reads the xlsx with data_only=True or converts to CSV, so it needs pre-computed values.
- **Inspect before writing.** The exact row where year headers live on the Data sheet, the block labels on Task, and the statistic labels on Task must be confirmed by reading the file.
- **Preserve formatting.** Do not clear cells, delete rows, or change styles.
- **Use INDEX/MATCH** for the lookup formulas (not plain cell references).
- **PERCENTILE function** in Excel uses `PERCENTILE(range, k)` where k is 0.25 or 0.75.

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