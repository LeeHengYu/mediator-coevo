# Task Instruction

Execute the following steps precisely to produce `/root/output/result.xlsx`.

## 0 – Inspect the workbook
```bash
pip install openpyxl
```
```python
import openpyxl
wb = openpyxl.load_workbook('/root/data/workbook.xlsx')
for s in wb.sheetnames:
    print(f'--- {s} ---')
    ws = wb[s]
    print(f'  dims: {ws.dimensions}')
    # Print key regions
    if s == 'Task':
        # Row 10 (years row), col D labels, yellow target areas
        for r in range(10, 11):
            print(f'  Row {r}:', [(c.column_letter, c.value) for c in ws[r]])
        for r in range(12, 32):
            print(f'  Row {r}: D={ws.cell(r,4).value}  H-L={[ws.cell(r,c).value for c in range(8,13)]}')
        for r in range(35, 51):
            print(f'  Row {r}: D={ws.cell(r,4).value}  H-L={[ws.cell(r,c).value for c in range(8,13)]}')
    if s == 'Data':
        # Print rows 20-39 to see structure
        for r in range(20, 40):
            print(f'  Row {r}:', [(ws.cell(r,c).column_letter, ws.cell(r,c).value) for c in range(1, 13)])
```
Study the output carefully. Identify:
- The exact years in Task row 10 (columns H–L).
- The series codes in Task column D for rows 12–17, 19–24, 26–31.
- The labels in Task column D for rows 35–40, 42–47, 50.
- The Data sheet layout: which column holds the series code, which row holds years, and the data block rows 21–38.

## 1 – Write lookup formulas in H12:L17, H19:L24, H26:L31

For each cell in these ranges, write an INDEX/MATCH formula that:
- Looks up the series code from column D of the current row against the series-code column on Data (determine exact column from inspection).
- Looks up the year from Task row 10 against the year row on Data (determine exact row from inspection).
- Returns the intersection from the Data block.

Use absolute references for the Data ranges and mixed references so the formula can be filled across columns and down rows. The pattern should be:
```
=INDEX(Data!$H$21:$L$38, MATCH($D12, Data!$D$21:$D$38, 0), MATCH(H$10, Data!$H$20:$L$20, 0))
```
Adjust the exact Data column for series codes and the exact Data row for years based on what you found in step 0. The key contract: the series-code column on Data and the year header row on Data must match exactly what the workbook contains.

## 2 – Write Net production slack formulas in H35:L40

Identify which of the three blocks (H12:L17, H19:L24, H26:L31) corresponds to "Finished Output", "Scrap And Rework", and "Rated Production Capacity" by reading the block header labels (likely in rows 11, 18, 25 or nearby). Then for each cell in H35:L40:
```
= (FinishedOutput_cell - ScrapAndRework_cell) / RatedProductionCapacity_cell * 100
```
For example if Finished Output is rows 12–17, Scrap And Rework is rows 19–24, Rated Production Capacity is rows 26–31:
```
H35 = (H12 - H19) / H26 * 100
```
Fill for all 6 rows × 5 columns.

## 3 – Write summary statistics in H42:L47

For each column (H through L), with the data range being the 6 cells in that column from rows 35–40:
- Row 42: `=MIN(H35:H40)`
- Row 43: `=MAX(H35:H40)`
- Row 44: `=MEDIAN(H35:H40)`
- Row 45: `=AVERAGE(H35:H40)`
- Row 46: `=PERCENTILE(H35:H40, 0.25)`
- Row 47: `=PERCENTILE(H35:H40, 0.75)`

Verify the row-to-statistic mapping against the labels in column D for rows 42–47. Adjust if the order differs.

## 4 – Write weighted mean in H50:L50

For each column (H through L):
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```
This uses the Net production slack percentages as values and Rated Production Capacity as weights.

## 5 – Cache numeric values into the XML

After writing all formulas, the verifier likely reads values via openpyxl `data_only=True` or CSV export, which requires cached `<v>` elements. To handle this:

1. Save the workbook with formulas first to a temp path.
2. Re-open it, and for every formula cell you wrote, compute the numeric value in Python by evaluating the formula logic against the Data sheet values.
3. Write those computed values into the cell's `value` attribute of a `data_only`-style cache. Specifically, use openpyxl's internal API or the zipfile/XML approach:
   - Open the saved xlsx as a zip.
   - Parse the Task sheet XML.
   - For each cell that has a formula (`<f>` element), add a `<v>` sibling with the computed numeric value.
   - Rewrite the zip.

Alternatively, a simpler approach:
- After writing all formulas with openpyxl, also set `cell.value` to the computed number BUT keep the formula. openpyxl doesn't support this natively, so use the XML injection approach:
  ```python
  from openpyxl.worksheet.cell import Cell
  # For each target cell:
  #   ws.cell(row, col).value = '=FORMULA(...)'
  # Then after saving, patch XML to add <v> tags
  ```

Here is the recommended XML-patching approach:
```python
import zipfile, shutil, os
from lxml import etree

# First, compute all numeric values in Python by reading Data sheet
# Build a dict: (row, col) -> numeric_value for all formula cells
# Then patch the XML

temp_path = '/root/output/result_temp.xlsx'
final_path = '/root/output/result.xlsx'
wb.save(temp_path)

# Find the Task sheet's XML path in the xlsx zip
# Typically xl/worksheets/sheet1.xml or sheet2.xml
# Read [Content_Types].xml or xl/workbook.xml to find the right sheet

ns = {'s': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
with zipfile.ZipFile(temp_path, 'r') as zin:
    # Find which sheetN.xml corresponds to 'Task'
    wbxml = etree.fromstring(zin.read('xl/workbook.xml'))
    sheets = wbxml.findall('.//s:sheet', ns)
    task_rid = None
    for sh in sheets:
        if sh.get('name') == 'Task':
            task_rid = sh.get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id')
    # Read rels to find filename
    rels = etree.fromstring(zin.read('xl/_rels/workbook.xml.rels'))
    rns = {'r': 'http://schemas.openxmlformats.org/package/2006/relationships'}
    task_file = None
    for rel in rels.findall('.//r:Relationship', rns):
        if rel.get('Id') == task_rid:
            task_file = 'xl/' + rel.get('Target')
    
    # Parse the sheet XML
    sheet_xml = etree.fromstring(zin.read(task_file))
    
    # For each cell with a formula, inject <v> with computed value
    # Build computed_values dict first (see below)
    for row_el in sheet_xml.findall('.//s:sheetData/s:row', ns):
        for c_el in row_el.findall('s:c', ns):
            ref = c_el.get('r')
            f_el = c_el.find('s:f', ns)
            if f_el is not None and ref in computed_values:
                v_el = c_el.find('s:v', ns)
                if v_el is None:
                    v_el = etree.SubElement(c_el, '{http://schemas.openxmlformats.org/spreadsheetml/2006/main}v')
                v_el.text = str(computed_values[ref])
    
    # Rewrite zip
    with zipfile.ZipFile(final_path, 'w', zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            if item.filename == task_file:
                zout.writestr(item, etree.tostring(sheet_xml, xml_declaration=True, standalone=True))
            else:
                zout.writestr(item, zin.read(item.filename))
```

To compute `computed_values`, read the Data sheet values into a Python dict, then for each formula cell, replicate the formula logic in Python:
- For lookup cells: use the same INDEX/MATCH logic in Python.
- For net production slack: `(finished - scrap) / capacity * 100`.
- For stats: use Python's statistics or numpy.
- For weighted mean: `sum(slack * capacity) / sum(capacity)`.

## 6 – Validate

After saving:
1. Re-open `/root/output/result.xlsx` with `openpyxl.load_workbook(data_only=True)`.
2. Print all cells in H12:L17, H19:L24, H26:L31, H35:L40, H42:L47, H50:L50.
3. Confirm all have numeric values (not None).
4. Re-open with `data_only=False` and confirm all those cells have formulas.
5. Verify no extra sheets were added.

IMPORTANT: Adapt all row/column references based on what you discover in Step 0. Do not blindly use the references above if the actual workbook layout differs. The critical contract is that the formulas must reference the correct Data sheet ranges and the correct Task sheet cells.

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