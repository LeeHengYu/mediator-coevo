# Task Instruction

Execute the following steps carefully to complete the task.

## 0. Setup
```bash
mkdir -p /root/output
pip install openpyxl
```

## 1. Inspect the workbook structure
Open `/root/data/workbook.xlsx` with openpyxl and inspect:
- Sheet names
- Sheet `Task`: Print rows 10-50 for columns A-L to understand the layout (series codes in column D, years in row 10, yellow cell regions)
- Sheet `Data`: Print rows 21-38 to understand the data source structure (column headers, row labels, data layout)

Pay special attention to:
- The exact series codes in column D of sheet `Task` for rows 12-17, 19-24, 26-31
- The exact years in row 10 for columns H-L
- The structure of the Data sheet rows 21-38 (is the series code in column A? What columns hold the year data?)
- Whether Data!A21:A38 contains the series codes and which row/column holds years

Print all of this before writing any formulas.

## 2. Write a Python script using openpyxl to populate formulas

Use openpyxl to open the workbook and write formulas into the cells. Important: use `Translator` or direct formula strings. Do NOT use `data_only=True` when opening.

### Step 1: Populate H12:L17, H19:L24, H26:L31 with lookup formulas

For each cell in these ranges, write a formula that looks up the value from `Data` sheet rows 21:38 using:
- The series code from column D of the same row on sheet `Task`
- The year from row 10 of the same column on sheet `Task`

Based on the Data sheet structure you discovered in step 1, choose the appropriate pattern. The most likely pattern is INDEX-MATCH-MATCH:

```
=INDEX(Data!$B$21:$XX$38, MATCH($D12, Data!$A$21:$A$38, 0), MATCH(H$10, Data!$B$20:$XX$20, 0))
```

Adjust the ranges based on what you actually find in the Data sheet. The key points:
- The first MATCH finds the row by matching the series code in column D against the series code column in Data
- The second MATCH finds the column by matching the year in row 10 against the year header row in Data
- Use absolute references for the Data ranges and mixed references ($D12 and H$10) so the formula works across the block

You MUST use one of these patterns: VLOOKUP+MATCH, HLOOKUP+MATCH, XLOOKUP+MATCH, or INDEX+MATCH.

### Step 2: Populate H35:L40 with Net renewable balance

The formula for each cell (campus, year) is:
```
=(H12 - H19) / H26 * 100
```
where row 12 corresponds to Renewable Generation (rows 12-17), row 19 to Grid Consumption (rows 19-24), and row 26 to Baseline Energy Demand (rows 26-31). The mapping is positional: row 35 uses rows 12, 19, 26; row 36 uses rows 13, 20, 27; etc.

Write these as cell-reference formulas (not hardcoded values).

### Step 2 continued: Populate H42:L47 with statistics

For each column (H through L):
- Row 42: `=MIN(H35:H40)` (minimum)
- Row 43: `=MAX(H35:H40)` (maximum)  
- Row 44: `=MEDIAN(H35:H40)` (median)
- Row 45: `=AVERAGE(H35:H40)` (simple mean)
- Row 46: `=PERCENTILE(H35:H40, 0.25)` (25th percentile)
- Row 47: `=PERCENTILE(H35:H40, 0.75)` (75th percentile)

But FIRST check the labels in column A/B/C/D for rows 42-47 to see which row is which statistic, and match accordingly. Do NOT assume the order above; verify it.

### Step 3: Populate H50:L50 with weighted mean

For each column:
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```

This uses the Net renewable balance percentages as values and Baseline Energy Demand as weights.

## 3. Save and verify

Save the workbook to `/root/output/result.xlsx`.

After saving, reopen the file and verify:
- Cells H12, H19, H26 contain formula strings (not None or hardcoded values)
- Cells H35, H42, H50 contain formula strings
- No new sheets were added
- Print a sample of formulas to confirm correctness

## Critical constraints
- Do NOT add new sheets, macros, VBA, external links, or helper tabs
- Do NOT change existing formatting
- Open with openpyxl without data_only to preserve and write formulas
- All formulas must be Excel-compatible spreadsheet formulas, not Python calculations

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