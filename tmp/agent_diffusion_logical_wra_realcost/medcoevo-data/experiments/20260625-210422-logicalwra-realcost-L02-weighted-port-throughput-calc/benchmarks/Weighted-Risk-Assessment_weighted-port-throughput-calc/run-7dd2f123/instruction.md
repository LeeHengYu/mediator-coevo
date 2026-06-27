# Task Instruction

## Objective

Update `/root/data/workbook.xlsx` with spreadsheet formulas and save the result to `/root/output/result.xlsx`. Work only inside the existing sheets `Task` and `Data`. Do not add sheets, macros, VBA, external links, or helper tabs. Preserve all existing formatting.

## Detailed Steps

### 0. Setup & Inspection

```bash
mkdir -p /root/output
pip install openpyxl lxml
```

Open the workbook with `openpyxl` (NOT data_only) and inspect:
- **Sheet `Task`**: Print rows 10-50, columns D through L, to understand the layout:
  - Row 10 contains year headers in columns H:L.
  - Column D (rows 12-17, 19-24, 26-31) contains series codes.
  - Rows 12-17 = first data block (e.g., Loaded Containers Inbound)
  - Rows 19-24 = second data block (e.g., Loaded Containers Outbound)
  - Rows 26-31 = third data block (e.g., Terminal Throughput Capacity)
  - Rows 35-40 = Net container flow
  - Rows 42-47 = summary statistics
  - Row 50 = weighted mean
- **Sheet `Data`**: Print rows 21-38, columns D through L, to see the lookup source data structure. Identify which column holds the series codes and which columns hold the year data.

Print all of these clearly before writing any formulas. You MUST know the exact cell references.

### 1. Step 1 — Lookup Formulas in H12:L17, H19:L24, H26:L31

For each cell in these three blocks, write an `INDEX/MATCH` formula that:
- Looks up the series code from column D of the current row
- Looks up the year from row 10 of the current column
- Searches in `Data` sheet rows 21:38

The exact formula pattern (adjust column/row references based on your inspection):
```
=INDEX(Data!$H$21:$L$38, MATCH($D12, Data!$D$21:$D$38, 0), MATCH(H$10, Data!$H$20:$L$20, 0))
```

**IMPORTANT**: Before writing formulas, verify:
- Which row in `Data` contains the year headers (likely row 20 or row 21 — inspect carefully)
- Which column in `Data` contains series codes (likely column D)
- The exact row range for data (rows 21:38)

Adjust the formula references accordingly. The MATCH for years should reference the header row in Data, and the MATCH for series codes should reference the series code column in Data.

### 2. Step 2 — Net Container Flow in H35:L40

For each cell in H35:L40, calculate:
```
=(H12 - H19) / H26 * 100
```
where the row offsets correspond to the same port (row 35 uses rows 12, 19, 26; row 36 uses rows 13, 20, 27; etc.).

So for cell H35: `=(H12-H19)/H26*100`
For cell H36: `=(H13-H20)/H27*100`
... and so on through row 40, and across columns H through L.

### 3. Step 2 — Summary Statistics in H42:L47

For each column H through L:
- Row 42: `=MIN(H35:H40)` (minimum)
- Row 43: `=MAX(H35:H40)` (maximum)
- Row 44: `=MEDIAN(H35:H40)` (median)
- Row 45: `=AVERAGE(H35:H40)` (simple mean)
- Row 46: `=PERCENTILE(H35:H40,0.25)` (25th percentile)
- Row 47: `=PERCENTILE(H35:H40,0.75)` (75th percentile)

**IMPORTANT**: Check the labels in column D or nearby columns for rows 42-47 to confirm the correct order (MIN, MAX, MEDIAN, AVERAGE, 25th, 75th). Adjust the row assignments if the labels indicate a different order.

### 4. Step 3 — Weighted Mean in H50:L50

For each column H through L:
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```

This uses the Net container flow percentages as values and Terminal Throughput Capacity as weights.

### 5. Cache Numeric Values in XML

After writing all formulas, you MUST inject cached `<v>` values into the worksheet XML so that openpyxl `data_only=True` and CSV-based verifiers can read numeric results.

Procedure:
1. Save the workbook with openpyxl first to `/root/output/result.xlsx`.
2. Reopen the workbook with `data_only=False` and also load `Data` sheet values.
3. Manually compute the numeric value for every formula cell by:
   - First, reading all Data sheet values into a dictionary keyed by (series_code, year).
   - Computing lookup results for H12:L31.
   - Computing net container flow for H35:L40.
   - Computing statistics for H42:L47.
   - Computing weighted means for H50:L50.
4. Use `lxml` to open the saved xlsx (it's a zip of XML files) and inject `<v>numeric_value</v>` into each formula cell's XML element. Set the cell type to `str` if not already, or just ensure `<v>` is present.
5. Save the modified zip back.

Detailed XML injection approach:
```python
import zipfile, shutil, copy
from lxml import etree

# Open the xlsx as a zip
with zipfile.ZipFile('/root/output/result.xlsx', 'r') as zin:
    # Find the Task sheet XML (likely xl/worksheets/sheet1.xml — check)
    # Parse it, find each cell by reference (e.g., 'H12'), 
    # add or update <v> element with the computed float value
    # Write back
```

Alternatively, a simpler approach: after computing all values in Python, write them using openpyxl's internal `cell._value` and `cell.data_type` manipulation, or just set `cell.value` to the computed number temporarily, then overwrite with the formula. Actually, the cleanest way:

```python
import openpyxl
from openpyxl.worksheet.formula import ArrayFormula

wb = openpyxl.load_workbook('/root/output/result.xlsx')
ws = wb['Task']

# For each formula cell, store the formula string, then:
# ws['H12'].value = '=INDEX(...)' sets the formula
# To cache: we need to manipulate the XML after save
```

Use the lxml/zipfile approach to inject `<v>` tags. For each formula cell at coordinates like H12:
- Find the `<c r="H12">` element in the sheet XML
- Ensure it has a child `<v>computed_value</v>`
- The computed_value should be the float representation

### 6. Validation

After saving the final file:
1. Reopen with `openpyxl.load_workbook('/root/output/result.xlsx', data_only=True)`
2. Print values from H12:L17, H19:L24, H26:L31, H35:L40, H42:L47, H50:L50
3. Verify they are numbers (not None, not strings)
4. Verify Net container flow formula logic with a manual spot-check
5. Verify weighted mean with a manual spot-check

### Critical Notes
- Read the actual sheet structure BEFORE writing any formulas. Print row 10 headers, column D series codes, Data sheet structure.
- The order of statistics rows 42-47 must match the labels in the sheet. READ THE LABELS.
- Do not skip the cached value injection step — the verifier likely reads values, not formulas.
- Preserve all existing formatting, merged cells, styles, etc.
- Do not add any new sheets or remove anything.

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