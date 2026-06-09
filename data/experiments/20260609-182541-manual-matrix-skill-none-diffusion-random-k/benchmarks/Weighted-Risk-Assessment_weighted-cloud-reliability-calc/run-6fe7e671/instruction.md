# Task Instruction

## Task: Populate formulas in /root/data/workbook.xlsx and save to /root/output/result.xlsx

### Preliminary Investigation
1. First, read the workbook structure carefully using openpyxl:
   - Open `/root/data/workbook.xlsx` with `openpyxl.load_workbook('/root/data/workbook.xlsx')` (use `data_only=False` to preserve formulas).
   - Inspect sheet names to confirm `Task` and `Data` exist.
   - On sheet `Task`: print the contents of row 10 (especially H10:L10) to see the years. Print column D for rows 12-17, 19-24, 26-31 to see the series codes. Print any labels in column A or nearby for rows 12-17, 19-24, 26-31 to understand what each block represents (likely: block 1 = Successful API Requests, block 2 = Failed API Requests, block 3 = Compute Capacity). Print rows 35-40 column D or A to see region names. Print rows 42-47 column A/G to see stat labels (min, max, median, mean, 25th, 75th percentile). Print row 50 to see the GCM label.
   - On sheet `Data`: print rows 21-38 to understand the data layout — identify which row has headers, which column has series codes, and how years are arranged (row-wise or column-wise). Print row 20 or the header row as well.
   - Print all findings before writing any formulas.

2. Pay special attention to:
   - The exact cell references: Are years in row 10 of `Task` sheet in H10, I10, J10, K10, L10?
   - The series codes in column D of `Task` sheet for each row.
   - The data layout on `Data` sheet rows 21:38 — is it a vertical table (series codes in a column, years across columns) or horizontal?
   - Identify which lookup pattern fits best. The instruction says to use one of: VLOOKUP+MATCH, HLOOKUP+MATCH, XLOOKUP+MATCH, or INDEX+MATCH.

### Step 1: Lookup Formulas in H12:L17, H19:L24, H26:L31

For each cell in these ranges, write a spreadsheet formula (as a string starting with `=`) that:
- Takes the series code from column D of the same row on `Task` sheet
- Takes the year from row 10 of the same column on `Task` sheet  
- Looks up the value from `Data` sheet rows 21:38

Use INDEX+MATCH (or whichever pattern fits the data layout). For example, if Data has series codes in column A and years in row 20:
```
=INDEX(Data!$B$21:$XX$38, MATCH($D12, Data!$A$21:$A$38, 0), MATCH(H$10, Data!$B$20:$XX$20, 0))
```
Adjust the exact column/row references based on what you discover in the data. The key is:
- The row match should find the series code from column D of the current Task row within the series code column of Data rows 21:38.
- The column match should find the year from row 10 of the current Task column within the year header row of Data.
- Use absolute references (`$`) appropriately: lock the series code column reference (`$D12`) and the year row reference (`H$10`), and lock the Data ranges entirely so formulas can be filled across the grid.

Write these formulas using openpyxl by assigning formula strings to each cell. Do NOT use `data_only=True`. Do NOT try to compute values in Python — write actual Excel formulas.

### Step 2: Net Reliability Gap (H35:L40) and Statistics (H42:L47)

For H35:L40, the formula is:
```
(Successful API Requests - Failed API Requests) / Compute Capacity * 100
```
Based on the block layout:
- Rows 12-17 = Successful API Requests (6 regions)
- Rows 19-24 = Failed API Requests (6 regions)
- Rows 26-31 = Compute Capacity (6 regions)

So for cell H35: `=(H12-H19)/H26*100`, H36: `=(H13-H20)/H27*100`, etc. Map each of the 6 regions in rows 35-40 to the corresponding rows in the three blocks above. Verify the region order matches by checking labels.

For H42:L47, calculate column-wise statistics over H35:L40:
- Row 42: `=MIN(H35:H40)` (adjust column per cell)
- Row 43: `=MAX(H35:H40)`
- Row 44: `=MEDIAN(H35:H40)`
- Row 45: `=AVERAGE(H35:H40)`
- Row 46: `=PERCENTILE(H35:H40,0.25)`
- Row 47: `=PERCENTILE(H35:H40,0.75)`

Verify the order of these statistics by reading the labels in column A or G for rows 42-47. Assign the correct formula to the correct row based on the actual label.

### Step 3: Weighted Mean in H50:L50

For each column H through L:
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```
This computes the weighted mean of Net reliability gap percentages (H35:H40) weighted by Compute Capacity (H26:H31).

### Saving
- Create `/root/output/` directory if it doesn't exist: `os.makedirs('/root/output', exist_ok=True)`
- Save the workbook to `/root/output/result.xlsx`
- Do NOT add any new sheets, macros, VBA, external links, or helper tabs.
- Do NOT change any existing formatting.

### Validation
After saving:
1. Reopen `/root/output/result.xlsx` with openpyxl (data_only=False).
2. Print sample cells from each formula region to confirm formulas are written (not None or empty).
3. Verify that all 30 lookup cells (H12:L17, H19:L24, H26:L31) contain formula strings.
4. Verify that all 6×5=30 cells in H35:L40 contain formulas.
5. Verify that all 6×5=30 cells in H42:L47 contain formulas.
6. Verify that all 5 cells in H50:L50 contain formulas.
7. Confirm no extra sheets were added.
8. Print the sheet names and total formula count as a final summary.

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