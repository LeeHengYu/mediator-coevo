# Task Instruction

## Task: Update hospital capacity workbook with formulas

You must update `/root/data/workbook.xlsx` and save the result to `/root/output/result.xlsx`.

### Preliminary: Inspect the workbook

1. Create `/root/output/` directory if it doesn't exist.
2. Use `openpyxl` to open `/root/data/workbook.xlsx` and inspect:
   - Sheet `Task`: Read the layout carefully. Print rows 1-55 for columns A-M to understand the structure. Pay special attention to:
     - Column D (series codes) for rows 12-17, 19-24, 26-31
     - Row 10 (years) for columns H-L
     - The labels/structure around H35:L40, H42:L47, H50:L50
   - Sheet `Data`: Print rows 21-38 to understand the data layout. Note which row contains headers, which column has series codes, and how years are arranged.
3. Print the exact cell values for D12:D17, D19:D24, D26:D31 to know the series codes.
4. Print the exact cell values for H10:L10 to know the years.
5. Print the Data sheet structure: identify the header row, the series code column, and the year columns.

### Step 1: Populate H12:L17, H19:L24, H26:L31 with lookup formulas

For each cell in these ranges, write a spreadsheet formula (not Python computation) that looks up data from `Data!$21:$38`. The formula must use two inputs:
- The series code from column D of the current row (e.g., `$D12`)
- The year from row 10 of the current column (e.g., `H$10`)

Use `INDEX`/`MATCH` pattern. The exact formula structure depends on the Data sheet layout you discover during inspection. For example, if Data has series codes in column A and years in a header row:
```
=INDEX(Data!<data_range>, MATCH($D12, Data!<series_code_column>, 0), MATCH(H$10, Data!<year_header_row>, 0))
```

Adapt the ranges based on what you find. Make sure:
- The series code reference locks the column (`$D12`) so it doesn't shift when copied across columns.
- The year reference locks the row (`H$10`) so it doesn't shift when copied down rows.
- The Data sheet reference range covers rows 21-38 as specified.

### Step 2a: Net capacity headroom in H35:L40

For each of the six hospital clusters (rows 35-40) and each year column (H-L), write a formula:
```
=(Available_Care_Slots - Occupied_Care_Slots) / Staffed_Bed_Capacity * 100
```

The three blocks correspond to:
- H12:L17 = one metric (identify which from labels)
- H19:L24 = another metric
- H26:L31 = another metric

Determine which block is "Available Care Slots", "Occupied Care Slots", and "Staffed Bed Capacity" from the row labels on the Task sheet. The six clusters in rows 35-40 should correspond to the six clusters in each block.

For example, if rows 12-17 are Available, 19-24 are Occupied, 26-31 are Staffed Bed Capacity, then for H35:
```
=(H12-H19)/H26*100
```

### Step 2b: Summary statistics in H42:L47

For each year column (H-L), calculate column-wise statistics over H35:L40:
- Row 42: `=MIN(H35:H40)` (minimum)
- Row 43: `=MAX(H35:H40)` (maximum)
- Row 44: `=MEDIAN(H35:H40)` (median)
- Row 45: `=AVERAGE(H35:H40)` (simple mean)
- Row 46: `=PERCENTILE(H35:H40, 0.25)` (25th percentile)
- Row 47: `=PERCENTILE(H35:H40, 0.75)` (75th percentile)

**IMPORTANT**: Check the labels in column A/B/C for rows 42-47 to determine the correct order of these statistics. Match the formula to the label in each row.

### Step 3: Weighted mean in H50:L50

For each year column, use SUMPRODUCT with the Net capacity headroom values (H35:H40) and the Staffed Bed Capacity values (H26:H31) as weights:
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```

### Important implementation notes

1. Use `openpyxl` to write formulas as strings (e.g., `ws['H12'] = '=INDEX(...)'`). Do NOT compute values in Python.
2. Preserve all existing formatting. Do not clear cells outside the target ranges. Do not modify cell styles.
3. When writing formulas, use the `Translator` or manual string construction to fill across the range, or write each cell individually.
4. Do NOT add sheets, macros, VBA, external links, or helper tabs.
5. Save to `/root/output/result.xlsx`.
6. After saving, re-open the file and verify that the formula cells contain formula strings (not None or numeric values). Print a sample of cells from each range to confirm.

### Verification checklist
- H12:L17 all contain INDEX/MATCH formulas referencing Data sheet
- H19:L24 all contain INDEX/MATCH formulas referencing Data sheet  
- H26:L31 all contain INDEX/MATCH formulas referencing Data sheet
- H35:L40 all contain capacity headroom formulas
- H42:L47 all contain statistical formulas (MIN, MAX, MEDIAN, AVERAGE, PERCENTILE)
- H50:L50 all contain SUMPRODUCT formulas
- No extra sheets added
- File saved to /root/output/result.xlsx

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