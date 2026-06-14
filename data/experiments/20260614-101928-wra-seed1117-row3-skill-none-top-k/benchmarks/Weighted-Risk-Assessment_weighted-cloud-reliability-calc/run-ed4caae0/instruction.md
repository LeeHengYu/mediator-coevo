# Task Instruction

## Task: Populate formulas in /root/data/workbook.xlsx and save to /root/output/result.xlsx

### Preliminary Investigation
1. First, create the output directory: `mkdir -p /root/output`
2. Install openpyxl if not already available: `pip install openpyxl`
3. Read and thoroughly inspect the workbook `/root/data/workbook.xlsx` using a Python script. You need to understand:
   - The structure of the `Task` sheet: what's in columns A-G and rows 1-50+, especially:
     - Column D (series codes) for rows 12-17, 19-24, 26-31
     - Row 10 (years in columns H-L)
     - The layout of H35:L40 (Net reliability gap), H42:L47 (statistics), H50:L50 (weighted mean)
   - The structure of the `Data` sheet, especially rows 21-38: column headers, how series codes map to rows, how years map to columns
   - What the yellow-highlighted cells look like (check fill colors)
   - Print out all cell values in the relevant ranges so you understand the exact layout

Use code like:
```python
import openpyxl
wb = openpyxl.load_workbook('/root/data/workbook.xlsx')
for name in wb.sheetnames:
    print(f'=== Sheet: {name} ===')
    ws = wb[name]
    print(f'Dimensions: {ws.dimensions}')
    for row in ws.iter_rows(min_row=1, max_row=ws.max_row, max_col=ws.max_column, values_only=False):
        for cell in row:
            if cell.value is not None:
                print(f'  {cell.coordinate}: {cell.value}')
```

Also specifically print:
- Task sheet column D rows 10-31 (series codes)
- Task sheet row 10 columns H-L (years)
- Task sheet rows 35-50 columns A-L
- Data sheet rows 21-38 with all columns
- Data sheet row 1 or header row to understand column structure

### Step 1: Populate lookup formulas in H12:L17, H19:L24, H26:L31

After understanding the layout, write a Python script using openpyxl to insert spreadsheet formulas (NOT computed values) into the yellow cells.

For each cell in these ranges, the formula must:
- Use the series code from column D of that row on the Task sheet
- Use the year from row 10 of that column on the Task sheet  
- Look up the corresponding value from the Data sheet rows 21:38
- Use one of the allowed patterns: INDEX/MATCH, VLOOKUP/MATCH, HLOOKUP/MATCH, or XLOOKUP/MATCH

IMPORTANT: You must inspect the Data sheet to determine:
- Which column contains the series codes (to match against Task!D)
- Which row contains the years (to match against Task row 10)
- The exact data range for the lookup
- Whether the data is organized with series codes in rows and years in columns, or vice versa

Based on the layout, construct the appropriate formula. For example, if Data sheet has series codes in column A and years in row 20 (or some header row), an INDEX/MATCH formula would look like:
`=INDEX(Data!$B$21:$XX$38, MATCH($D12, Data!$A$21:$A$38, 0), MATCH(H$10, Data!$B$20:$XX$20, 0))`

Adjust all references based on what you actually find in the workbook. Use absolute references ($) appropriately so formulas can be conceptually dragged across rows and columns ($ on the lookup arrays and the row/column anchors).

### Step 2: Net reliability gap formulas in H35:L40 and statistics in H42:L47

For H35:L40 (6 regions × 5 years):
- Formula: `(Successful API Requests - Failed API Requests) / Compute Capacity * 100`
- Successful API Requests are in H12:L17
- Failed API Requests are in H19:L24  
- Compute Capacity is in H26:L31
- VERIFY these mappings by checking the Task sheet labels in column A or B for rows 12-17, 19-24, 26-31, and 35-40. The six regions must correspond in the same order.
- Example formula for H35: `=(H12-H19)/H26*100`

For H42:L47 (statistics for each year column):
- Row 42: MIN of H35:H40 (or the corresponding column)
- Row 43: MAX
- Row 44: MEDIAN
- Row 45: AVERAGE (simple mean)
- Row 46: PERCENTILE (25th) or PERCENTILE.INC
- Row 47: PERCENTILE (75th) or PERCENTILE.INC
- VERIFY the order by checking labels in column A/B/C for rows 42-47. The order of min/max/median/mean/25th/75th percentile must match whatever labels are there.
- Example: `=MIN(H35:H40)`, `=MAX(H35:H40)`, `=MEDIAN(H35:H40)`, `=AVERAGE(H35:H40)`, `=PERCENTILE.INC(H35:H40,0.25)`, `=PERCENTILE.INC(H35:H40,0.75)`

### Step 3: Weighted mean in H50:L50

For each column H through L:
- Use SUMPRODUCT with the Net reliability gap values (H35:H40) and Compute Capacity weights (H26:H31)
- Formula: `=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)`
- This computes the weighted mean where weights are Compute Capacity values.

### Saving
- Save the workbook to `/root/output/result.xlsx`
- Do NOT change formatting, do NOT add sheets, macros, VBA, external links, or helper tabs
- When loading the workbook, do NOT use data_only=True (you want to preserve and write formulas)

### Validation
After saving, reload the workbook and verify:
1. All cells in H12:L17, H19:L24, H26:L31 contain formula strings (start with '=')
2. All cells in H35:L40 contain formula strings
3. All cells in H42:L47 contain formula strings
4. All cells in H50:L50 contain formula strings
5. The formulas reference the correct ranges
6. No extra sheets were added
7. Spot-check a few formulas by printing them out

### Critical Notes
- You MUST inspect the actual workbook structure before writing any formulas. Do not assume cell positions.
- The formulas must be Excel formulas stored as strings in openpyxl (they will start with '=')
- Make sure row and column references in the formulas exactly match the actual data layout
- Use mixed references appropriately: anchor the series code column with $D, anchor the year row with $10, and anchor lookup ranges with $ on both row and column

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
Task metadata: author_email=catpaw@meituan.com, author_name=CatPaw Task Engineer, category=spreadsheet-formula-reuse, difficulty=easy, tags=[excel, formulas, lookup, statistics, weighted-mean].
Verifier config: timeout_sec=600.0.