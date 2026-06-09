# Task Instruction

Execute the following steps carefully to complete the task.

## 0. Setup
```bash
mkdir -p /root/output
cp /root/data/workbook.xlsx /root/output/result.xlsx
```

## 1. Inspect the workbook structure
Using openpyxl (or xlsxwriter if needed), open `/root/output/result.xlsx` and inspect:
- Sheet names (confirm `Task` and `Data` exist)
- On sheet `Task`: read row 10 to see the years in columns H–L; read column D rows 12–17, 19–24, 26–31 to see the series codes; read rows 35–40 to see port names/structure; read row 50 for CPA label; read rows 42–47 for stat labels.
- On sheet `Data`: read rows 21–38 to understand the data layout — identify which column contains the series codes, which row contains the years, and how the data is arranged (is it a vertical table with series codes in one column and years across columns, or something else?).

Print all of this information before proceeding. You need to understand the exact layout to write correct formulas.

## 2. Determine the Data sheet layout
From the inspection, identify:
- The column on `Data` that holds the series/indicator codes (likely column A or B in rows 21–38)
- The row on `Data` that holds the year headers
- The data range for values

This is critical for writing correct VLOOKUP/INDEX-MATCH formulas.

## 3. Write formulas using openpyxl
Open the workbook with `openpyxl.load_workbook('/root/output/result.xlsx')`. For each cell, set the `.value` to a string formula. Do NOT use `data_only=True` when loading.

### Step 1: Populate H12:L17, H19:L24, H26:L31

For each cell in these ranges, write an INDEX-MATCH formula. The formula pattern should be:
```
=INDEX(Data!<data_columns>, MATCH($D<row>, Data!<series_code_column>, 0), MATCH(H$10, Data!<year_header_row>, 0))
```

Where:
- `$D<row>` is the series code in column D of the current row on Task sheet (use $ to lock column D)
- `H$10` (or I$10, J$10, etc.) is the year from row 10 (use $ to lock row 10)
- `Data!<data_columns>` is the rectangular data block on the Data sheet (rows 21:38, from the first data column to the last)
- `Data!<series_code_column>` is the column with series codes (same rows 21:38)
- `Data!<year_header_row>` is the row with year headers (same columns as data)

IMPORTANT: Adjust the exact references based on your inspection in steps 1-2. The INDEX range, MATCH lookup arrays must be exactly right.

Alternatively, if the data is arranged with series codes in a column and years in a row header, you could use:
```
=INDEX(Data!$<startcol>$21:$<endcol>$38, MATCH($D12, Data!$<codecol>$21:$<codecol>$38, 0), MATCH(H$10, Data!$<startcol>$<yearrow>:$<endcol>$<yearrow>, 0))
```

### Step 2: Net container flow in H35:L40

The formula for each cell should be:
```
=(<loaded_inbound_cell> - <loaded_outbound_cell>) / <throughput_capacity_cell> * 100
```

Where:
- Loaded Containers Inbound is from H12:L17 block
- Loaded Containers Outbound is from H19:L24 block  
- Terminal Throughput Capacity is from H26:L31 block

So for cell H35: `=(H12-H19)/H26*100`
For cell H36: `=(H13-H20)/H27*100`
...and so on for all 6 ports × 5 years.

### Step 2 continued: Statistics in H42:L47

For each column (H through L):
- H42 (min): `=MIN(H35:H40)`
- H43 (max): `=MAX(H35:H40)`
- H44 (median): `=MEDIAN(H35:H40)`
- H45 (mean): `=AVERAGE(H35:H40)`
- H46 (25th percentile): `=PERCENTILE(H35:H40,0.25)`
- H47 (75th percentile): `=PERCENTILE(H35:H40,0.75)`

Verify the labels in rows 42–47 to confirm the order (min, max, median, mean, 25th, 75th) — adjust if the labels differ.

### Step 3: Weighted mean in H50:L50

For each column:
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```

This computes the weighted mean of net container flow percentages weighted by throughput capacity.

## 4. Save and verify
Save the workbook. Then re-open it and verify:
- All formula cells contain string formulas (not None or numeric values)
- The formulas reference the correct sheets and ranges
- No new sheets were added
- Print a sample of formulas to confirm correctness

## Critical Notes
- Use `openpyxl.load_workbook(filename)` without `data_only=True`
- Set cell values as strings starting with `=`
- Do NOT delete or modify any existing formatting, values, or structure
- Do NOT add sheets, macros, or external links
- The final file must be saved to `/root/output/result.xlsx`
- Before writing any formula, print the actual cell contents from your inspection to verify your understanding of the layout

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