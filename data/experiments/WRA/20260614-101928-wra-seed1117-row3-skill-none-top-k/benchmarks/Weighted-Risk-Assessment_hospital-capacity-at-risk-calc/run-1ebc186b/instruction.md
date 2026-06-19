# Task Instruction

Execute the following steps precisely to complete the hospital capacity risk workbook.

## 0. Setup
```bash
mkdir -p /root/output
pip install openpyxl
```

## 1. Inspect the workbook structure
Open `/root/data/workbook.xlsx` with openpyxl and inspect:
- Sheet names (confirm `Task` and `Data` exist)
- On `Task` sheet: read rows 10-50, columns D through L, to understand the layout:
  - Row 10: header row with years in columns H-L
  - Column D rows 12-17: series codes for block 1
  - Column D rows 19-24: series codes for block 2  
  - Column D rows 26-31: series codes for block 3
  - Rows 35-40: Net capacity headroom (6 hospital clusters)
  - Rows 42-47: summary statistics (min, max, median, mean, 25th, 75th percentile)
  - Row 50: weighted mean for Regional Care Grid
- On `Data` sheet: inspect rows 21-38 to understand the data layout (which row/column holds series codes, which holds year headers, which holds values). Print out the first few columns and rows to understand orientation.

Print all of this information before proceeding. This is critical for writing correct formulas.

## 2. Write formulas using openpyxl
Use a Python script with openpyxl to open the workbook and write formulas. Important: load with `data_only=False` to preserve existing formulas. Do NOT use `keep_vba` unless the file is .xlsm.

### Step 1 formulas: H12:L17, H19:L24, H26:L31
For each cell in these ranges, write an INDEX/MATCH formula that:
- Uses the series code from column D of the same row on the `Task` sheet
- Uses the year from row 10 of the same column on the `Task` sheet  
- Looks up in the `Data` sheet rows 21:38

The exact formula pattern depends on the Data sheet layout. After inspecting:
- If Data has series codes in a column (say column A or B) and years across a row (say row 20 or 21), use:
  `=INDEX(Data!<data_range>, MATCH(D12, Data!<series_code_column>, 0), MATCH(H10, Data!<year_row>, 0))`
- Adjust the ranges based on actual inspection.

Make sure to use absolute references where needed (e.g., `$D12` for the series code column, `H$10` for the year row) so formulas are consistent across the block.

### Step 2 formulas: H35:L40 and H42:L47
For H35:L40 (Net capacity headroom for 6 clusters):
- The formula is: `(Available Care Slots - Occupied Care Slots) / Staffed Bed Capacity * 100`
- Determine which of the three blocks (H12:L17, H19:L24, H26:L31) corresponds to Available Care Slots, Occupied Care Slots, and Staffed Bed Capacity by reading the labels in the Task sheet (likely in column C or nearby).
- For each row i (0-5) and column j (H-L):
  `=(BlockA_cell - BlockB_cell) / BlockC_cell * 100`
  where BlockA = Available Care Slots, BlockB = Occupied Care Slots, BlockC = Staffed Bed Capacity.

For H42:L47 (column-wise statistics):
- Read the labels in column C/D/E for rows 42-47 to confirm which statistic goes where.
- Use these formulas for each column (H through L):
  - MIN: `=MIN(H35:H40)`
  - MAX: `=MAX(H35:H40)`
  - MEDIAN: `=MEDIAN(H35:H40)`
  - AVERAGE: `=AVERAGE(H35:H40)`
  - 25th percentile: `=PERCENTILE(H35:H40, 0.25)`
  - 75th percentile: `=PERCENTILE(H35:H40, 0.75)`
- Match the row to the correct statistic based on the labels.

### Step 3 formula: H50:L50
For each column (H through L):
`=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)`
This computes the weighted mean using Net capacity headroom values as the values and Staffed Bed Capacity as weights.

## 3. Save
Save the workbook to `/root/output/result.xlsx`. Do NOT change any formatting, do NOT add sheets, macros, or VBA.

## 4. Verify
Reopen `/root/output/result.xlsx` with openpyxl and confirm:
- All target cells contain formula strings (not None or plain values)
- Formulas reference the correct ranges
- No extra sheets were added
- Print a sample of formulas from each block for verification

## Critical Notes
- You MUST inspect the Data sheet layout before writing any formulas. The exact row/column references depend on the actual structure.
- Use `$` for mixed references in lookup formulas: lock the column for series codes (`$D12`) and lock the row for years (`H$10`).
- The formula strings must use Excel syntax, not Python syntax.
- When writing formulas with openpyxl, assign the formula as a string starting with `=`.
- Preserve all existing content and formatting. Only write to the specified yellow cells.

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