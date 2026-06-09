# Task Instruction

Complete the following task to update an Excel workbook with formulas.

## Setup
1. `mkdir -p /root/output`
2. Install openpyxl if needed: `pip install openpyxl`
3. Read `/root/data/workbook.xlsx` to understand the structure of sheets `Task` and `Data`.

## Inspection (do this FIRST before writing any code)
Using openpyxl, inspect and print:
- Sheet `Task`: cells in column D rows 12-31 (series codes), row 10 columns H-L (year headers), and any existing content/labels in rows 35-50.
- Sheet `Data`: rows 21-38 to understand the data layout (which column has series codes, which row has year headers, where the data values are).
- Note the exact column letters and row numbers for the Data sheet's series code column and year header row.

## Implementation
Write a Python script using openpyxl to open the workbook and insert formulas. Do NOT compute values — insert Excel formula strings into cells.

### Step 1: Lookup formulas in H12:L17, H19:L24, H26:L31
For each cell in these ranges, create an INDEX/MATCH formula that:
- Uses the series code from column D of that row on sheet `Task`
- Uses the year from row 10 of that column on sheet `Task`
- Looks up the value from sheet `Data` rows 21:38

The formula pattern should be:
```
=INDEX(Data!$<data_range>, MATCH(D<row>, Data!$<series_code_column>, 0), MATCH(<col>10, Data!$<year_header_row>, 0))
```
Replace placeholders with the actual ranges discovered during inspection. Make sure the data range covers the full block of data rows and columns in the Data sheet.

### Step 2: Net capacity headroom in H35:L40
Based on the labels, determine which rows in the lookup blocks correspond to:
- Available Care Slots (likely H12:L17)
- Occupied Care Slots (likely H19:L24)  
- Staffed Bed Capacity (likely H26:L31)

Verify by checking the labels/series codes. The formula for each cell in H35:L40 is:
```
=(<Available_Care_Slots_cell> - <Occupied_Care_Slots_cell>) / <Staffed_Bed_Capacity_cell> * 100
```
Map row offsets correctly (row 35 corresponds to the first cluster, row 40 to the sixth).

### Summary statistics in H42:L47
For each column H through L, calculate over the range of that column in rows 35:40:
- Row 42: `=MIN(<col>35:<col>40)`
- Row 43: `=MAX(<col>35:<col>40)`
- Row 44: `=MEDIAN(<col>35:<col>40)`
- Row 45: `=AVERAGE(<col>35:<col>40)`
- Row 46: `=PERCENTILE(<col>35:<col>40, 0.25)` — use legacy PERCENTILE, NOT PERCENTILE.INC
- Row 47: `=PERCENTILE(<col>35:<col>40, 0.75)` — use legacy PERCENTILE, NOT PERCENTILE.INC

**CRITICAL**: Use `PERCENTILE` (legacy function), not `PERCENTILE.INC` or `_xlfn.PERCENTILE.INC`. The previous successful execution confirmed this is required for this specific task.

Verify the row-to-statistic mapping by checking labels in column D or G for rows 42-47.

### Step 3: Weighted mean in H50:L50
For each column H through L:
```
=SUMPRODUCT(<col>35:<col>40, <col>26:<col>31) / SUM(<col>26:<col>31)
```
This uses the Step 2 percentages as values and Staffed Bed Capacity as weights.

## Saving
- Save to `/root/output/result.xlsx`
- Do NOT change formatting, add sheets, macros, VBA, external links, or helper tabs.

## Validation
After saving, reopen the file and print:
1. A sample lookup formula (e.g., H12) to verify the INDEX/MATCH pattern
2. A sample headroom formula (e.g., H35)
3. A sample PERCENTILE formula (e.g., H46) — confirm it says `PERCENTILE` not `PERCENTILE.INC`
4. A sample weighted mean formula (e.g., H50)
5. Confirm no new sheets were added
6. Confirm the formulas reference the correct Data sheet ranges

## Important Notes
- Inspect the actual workbook structure BEFORE writing formulas. Do not assume column/row positions.
- The row-to-statistic mapping for rows 42-47 must match the labels in the workbook. Print and verify the labels.
- Use `data_only=False` when reading to see formulas, not cached values.

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