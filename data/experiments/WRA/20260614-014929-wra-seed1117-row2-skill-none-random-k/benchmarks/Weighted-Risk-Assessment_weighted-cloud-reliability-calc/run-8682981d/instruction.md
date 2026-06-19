# Task Instruction

## Task: Update workbook with formulas and save to output

### Setup
1. `mkdir -p /root/output`
2. Inspect the workbook `/root/data/workbook.xlsx` to understand its structure before making any changes. Specifically:
   - Read sheet `Task`: examine the layout of columns A-L, especially rows 10-50. Note what's in column D (series codes), row 10 (years), and the structure of the yellow cell ranges.
   - Read sheet `Data`: examine rows 21-38 to understand the data layout (which column has series codes, which row has years, where the values are).
   - Note the exact series codes in column D of the `Task` sheet for rows 12-17, 19-24, 26-31, and 35-40.
   - Note the exact years in row 10 of columns H-L.
   - Note the data layout on `Data` sheet rows 21-38: which column contains the series code identifier, which row contains the year headers, and where the numeric data lives.

### Step 1: Populate lookup formulas in H12:L17, H19:L24, H26:L31

For each cell in these ranges, write a formula that looks up data from the `Data` sheet rows 21:38. The formula should use:
- The series code from column D of the current row on sheet `Task`
- The year from row 10 of the current column on sheet `Task`

Use `INDEX/MATCH` (or one of the other allowed patterns: VLOOKUP+MATCH, HLOOKUP+MATCH, XLOOKUP+MATCH). The formula pattern should be something like:

`=INDEX(Data!<data_range>, MATCH($D12, Data!<series_code_column>, 0), MATCH(H$10, Data!<year_row>, 0))`

Adjust the exact ranges based on what you find when inspecting the sheets:
- `<data_range>`: the block of numeric values on Data sheet (rows 21-38, columns with year data)
- `<series_code_column>`: the column on Data sheet that contains the series codes (likely column A or B of rows 21-38)
- `<year_row>`: the row on Data sheet that contains the year headers

IMPORTANT: Use absolute references for the Data ranges and mixed references ($D for column D, $10 for row 10) so the formula can be applied across the entire range correctly. Make sure the INDEX range, the MATCH lookup column, and the MATCH lookup row are all consistently aligned.

Apply this formula to ALL cells in H12:L17, H19:L24, and H26:L31 (that's 6 rows × 5 columns = 30 cells per block, 90 cells total).

### Step 2: Net reliability gap in H35:L40 and statistics in H42:L47

For H35:L40, calculate for each region (6 regions, rows 35-40) and each year (columns H-L):
```
= (Successful API Requests - Failed API Requests) / Compute Capacity * 100
```

Based on the sheet structure:
- Successful API Requests should be in H12:L17 (or whichever block corresponds - verify by checking the labels)
- Failed API Requests should be in H19:L24 (verify)
- Compute Capacity should be in H26:L31 (verify)

So the formula for H35 would be something like: `=(H12-H19)/H26*100`

Verify which blocks correspond to which metric by reading the labels in the Task sheet before writing formulas.

For H42:L47, calculate column-wise statistics over H35:L40:
- Row 42: MIN → `=MIN(H35:H40)` (or `=MIN(H$35:H$40)`)
- Row 43: MAX → `=MAX(H35:H40)`
- Row 44: MEDIAN → `=MEDIAN(H35:H40)`
- Row 45: AVERAGE → `=AVERAGE(H35:H40)`
- Row 46: 25th percentile → `=PERCENTILE(H35:H40, 0.25)` or `=PERCENTILE.INC(H35:H40, 0.25)`
- Row 47: 75th percentile → `=PERCENTILE(H35:H40, 0.75)` or `=PERCENTILE.INC(H35:H40, 0.75)`

Verify the exact row assignments by reading the labels in column A-G of rows 42-47.

### Step 3: Weighted mean in H50:L50

For each column H-L, calculate the weighted mean for Global Cloud Mesh (GCM):
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```

This uses the Net reliability gap percentages (H35:H40) as values and Compute Capacity (H26:L31) as weights.

### Implementation approach
Use `openpyxl` in Python to:
1. Load the workbook with `data_only=False` (to preserve existing formulas)
2. Write the formulas as strings into the appropriate cells
3. Save to `/root/output/result.xlsx`
4. Do NOT change any formatting, do NOT add sheets, macros, VBA, external links, or helper tabs

### Validation
After saving, reload the workbook and verify:
- Cells H12, L17, H19, L24, H26, L31 contain formula strings (starting with '=')
- Cells H35, L40 contain formula strings
- Cells H42, L47 contain formula strings
- Cells H50, L50 contain formula strings
- The workbook still has exactly the same sheet names as before
- No new sheets were added

### Critical Notes
- Read the actual sheet structure FIRST before writing any formulas. The exact row/column references on the Data sheet are essential.
- Pay careful attention to whether the Data sheet uses row 21 as a header row or as data.
- Make sure MATCH references point to the correct lookup arrays.
- The formula must be written as a string (e.g., cell.value = '=INDEX(...)'), not as a computed value.

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