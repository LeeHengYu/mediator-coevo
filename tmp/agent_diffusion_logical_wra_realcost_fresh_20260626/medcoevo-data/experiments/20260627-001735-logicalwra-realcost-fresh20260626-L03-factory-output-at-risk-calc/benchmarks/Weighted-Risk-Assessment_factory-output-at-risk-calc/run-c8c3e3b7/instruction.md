# Task Instruction

Execute the following steps to complete the task.

## 0. Inspect the workbook

```
cp /root/data/workbook.xlsx /root/data/workbook_backup.xlsx
```

Open `/root/data/workbook.xlsx` with openpyxl (data_only=False) and inspect:
- Sheet names (confirm `Task` and `Data` exist).
- `Task` sheet: read row 10 (especially H10:L10) to find the year headers. Read column D rows 12-17, 19-24, 26-31 to find the series codes for each block. Read any labels in column B or C for the three blocks to understand which block is Finished Output, Scrap And Rework, and Rated Production Capacity. Read rows 35-40 column D for the plant names/codes. Read row 42-47 column C or D for the stat labels (min, max, median, mean, 25th, 75th). Read row 50 to find the "Regional Output Council" label.
- `Data` sheet: read rows 21-38, especially column D (series codes) and columns H-L (data values). Also read Data row 20 or whichever row has the year headers for columns H-L to confirm alignment with Task row 10.

Print all of this so you understand the exact layout before writing any formulas.

## 1. Write lookup formulas in H12:L31 on Task sheet

For each cell in the three blocks (H12:L17, H19:L24, H26:L31), write a formula using INDEX/MATCH that:
- Looks up the series code from column D of the current row in the Data sheet's series code column (Data!$D$21:$D$38).
- Looks up the year from row 10 of the Task sheet in the Data sheet's year header row.
- Returns the corresponding value from Data!$H$21:$L$38.

The exact formula pattern for cell HXX should be:
```
=INDEX(Data!$H$21:$L$38, MATCH($D{row}, Data!$D$21:$D$38, 0), MATCH(H$10, Data!$H$20:$L$20, 0))
```
Adjust the Data year header row reference ($H$20:$L$20) based on what you found in step 0 — use whichever row in Data contains the year labels that align with Task row 10. Verify the row number by inspection.

IMPORTANT: Use absolute references for the data range and series code column ($D$21:$D$38, $H$21:$L$38) and mixed references for the lookup keys ($D{row} for series code with column locked, H$10 for year with row locked) so formulas copy correctly across the grid.

## 2. Write Net production slack formulas in H35:L40

For each cell in H35:L40, compute:
```
=(H12 - H19) / H26 * 100
```
where:
- H12 corresponds to the Finished Output block (rows 12-17)
- H19 corresponds to the Scrap And Rework block (rows 19-24)  
- H26 corresponds to the Rated Production Capacity block (rows 26-31)

The row offset mapping: row 35→(12,19,26), row 36→(13,20,27), row 37→(14,21,28), row 38→(15,22,29), row 39→(16,23,30), row 40→(17,24,31).

So for cell in row r, col c:
```
=({c}{r-23} - {c}{r-16}) / {c}{r-9} * 100
```

Verify the block identity labels before writing. If the blocks are in a different order than assumed, adjust accordingly.

## 3. Write summary statistics in H42:L47

For each column c in H through L:
- Row 42 (MIN): `=MIN({c}35:{c}40)`
- Row 43 (MAX): `=MAX({c}35:{c}40)`
- Row 44 (MEDIAN): `=MEDIAN({c}35:{c}40)`
- Row 45 (MEAN): `=AVERAGE({c}35:{c}40)`
- Row 46 (25th percentile): `=PERCENTILE({c}35:{c}40,0.25)`
- Row 47 (75th percentile): `=PERCENTILE({c}35:{c}40,0.75)`

IMPORTANT: Check the actual labels in column C or D of rows 42-47 to determine the correct order. Map each stat to the correct row based on the label you read. Do NOT assume the order above — verify it.

## 4. Write weighted mean in H50:L50

For each column c in H through L:
```
=SUMPRODUCT({c}35:{c}40, {c}26:{c}31) / SUM({c}26:{c}31)
```
This uses Net production slack values as the values and Rated Production Capacity as weights.

## 5. Inject cached numeric values into the XML

After writing all formulas, save the workbook. Then:

1. Re-open the saved file with openpyxl (data_only=False).
2. For every cell that has a formula (all cells you wrote in steps 1-4), manually compute the expected numeric value using the raw data from the Data sheet.
3. Set each cell's internal cached value so that `cell.value` returns the formula but the XML `<v>` element contains the numeric result.

To do this with openpyxl:
```python
import openpyxl
from openpyxl.worksheet._writer import WorksheetWriter
# After writing formulas and saving once, reopen:
wb = openpyxl.load_workbook('/root/output/result.xlsx')
ws = wb['Task']
# For each formula cell, e.g.:
cell = ws['H12']
# The formula is stored; now we need to also cache the value.
# openpyxl stores formula cells with value=None by default.
# We need to manually set the cached value.
```

The reliable approach: use the `xlcalc` or manual computation approach:
- Read all data values from the Data sheet into a Python dict keyed by (series_code, year).
- For each lookup cell, compute the value in Python and cache it.
- For each derived cell (steps 2-4), compute from the looked-up values.
- To inject the cached value, after writing the formula string, also write the numeric value into the cell's internal `_value` attribute while keeping the data_type as 'f'. 

Actually, the most reliable method:
```python
# Write formula
cell.value = '=INDEX(...)'
# Then set the cached value that will appear in XML <v> tag
cell._value = computed_number  # This replaces the formula!
```

That won't work because it replaces the formula. Instead, use this approach:

**Method: Write formulas, save, then patch the XML directly.**

1. Save the workbook with formulas to `/root/output/result.xlsx`.
2. Unzip the xlsx file.
3. Parse `xl/worksheets/sheet1.xml` (or whichever sheet is Task).
4. For each `<c>` element that has an `<f>` child (formula cells you wrote), add or update a `<v>` child with the computed numeric value.
5. Rezip and save.

This is the most reliable way to ensure both formulas AND cached values exist.

Alternatively, a simpler openpyxl approach that works:
```python
# When writing formulas, write them normally
ws['H12'] = '=INDEX(...)' 
# Save the workbook
wb.save(...)

# Then use xlcalc or manual patching
```

Use the XML patching approach. Steps:
```python
import zipfile, shutil, os
from lxml import etree

# After saving with openpyxl:
shutil.copy('/root/output/result.xlsx', '/root/output/result_tmp.xlsx')

with zipfile.ZipFile('/root/output/result_tmp.xlsx', 'r') as zin:
    with zipfile.ZipFile('/root/output/result.xlsx', 'w') as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == 'xl/worksheets/sheet1.xml':  # adjust sheet name
                # Parse and patch
                root = etree.fromstring(data)
                ns = {'s': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
                for c_elem in root.findall('.//s:c', ns):
                    ref = c_elem.get('r')
                    if ref in computed_values_dict:
                        f_elem = c_elem.find('s:f', ns)
                        if f_elem is not None:
                            v_elem = c_elem.find('s:v', ns)
                            if v_elem is None:
                                v_elem = etree.SubElement(c_elem, '{http://schemas.openxmlformats.org/spreadsheetml/2006/main}v')
                            v_elem.text = str(computed_values_dict[ref])
                data = etree.tostring(root, xml_declaration=True, encoding='UTF-8', standalone=True)
            zout.writestr(item, data)
```

Build `computed_values_dict` mapping cell references (e.g., 'H12') to their numeric values, computed in Python from the Data sheet values.

## 6. Save and verify

Save to `/root/output/result.xlsx`.

Verify by:
1. Re-open with openpyxl (data_only=True) and check that cells have numeric values.
2. Re-open with openpyxl (data_only=False) and check that cells have formula strings.
3. Spot-check a few values against manual calculation.

## Critical notes
- Do NOT add any new sheets, macros, VBA, or external links.
- Preserve all existing formatting.
- Make sure to create `/root/output/` directory if it doesn't exist.
- The Task sheet might not be sheet1.xml — check the workbook.xml rels or sheet order to find the correct XML file for the Task sheet.
- When patching XML, preserve the namespace correctly. The `<v>` element must be in the spreadsheetml namespace.
- Double-check that PERCENTILE uses the Excel-compatible function name (PERCENTILE or PERCENTILE.INC — use PERCENTILE for maximum compatibility).
- Verify block assignments (which rows are Finished Output, Scrap And Rework, Rated Production Capacity) by reading the actual labels before writing formulas.

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