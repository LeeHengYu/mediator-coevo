# Task Instruction

## Task: Populate formulas in /root/data/workbook.xlsx and save to /root/output/result.xlsx

This task requires writing Excel formulas into specific cells of an existing workbook. Follow these steps precisely.

### Step 0: Inspect the workbook
1. `mkdir -p /root/output`
2. Use Python with openpyxl to open `/root/data/workbook.xlsx` and inspect:
   - Sheet names (should include `Task` and `Data`)
   - On sheet `Task`: read the contents of column D rows 12-17, 19-24, 26-31 (series codes), and row 10 columns H-L (years). Also read row labels in rows 35-40 (region names), rows 42-47 (stat names: min, max, median, mean, 25th, 75th percentile), and row 50.
   - On sheet `Data`: read rows 21-38 to understand the data layout — specifically which row contains headers, which column contains series codes, and how years are arranged (row-wise or column-wise). Print the first few columns and all rows 20-40 to understand the structure.
   - Print cell values so you can see exact series codes, year values, and data orientation.

### Step 1: Write lookup formulas in H12:L17, H19:L24, H26:L31

Based on the inspection, write formulas using INDEX/MATCH (or XLOOKUP with MATCH) that:
- Use the series code from column D of the current row on `Task` sheet
- Use the year from row 10 of the current column on `Task` sheet
- Look up the value from `Data` sheet rows 21:38

**CRITICAL**: Pay close attention to:
- Whether the Data sheet has series codes in a column (e.g., column A or B) and years in a row (e.g., row 20 or row 21)
- Use absolute references for the lookup ranges on the Data sheet
- The exact match type (0 for exact match in MATCH)
- Make sure the series codes on the Task sheet match exactly (character-for-character) with those on the Data sheet. Print both to verify.
- If Data is organized with series codes in a column and years in a header row, use: `=INDEX(Data!$B$21:$Z$38, MATCH($D12, Data!$A$21:$A$38, 0), MATCH(H$10, Data!$B$20:$Z$20, 0))` — but adjust the actual ranges based on your inspection.

For each of the three blocks (H12:L17, H19:L24, H26:L31), iterate over every cell and write the appropriate formula. The formula should use mixed references: `$D12` (column fixed, row relative) for the series code, and `H$10` (row fixed, column relative) for the year.

### Step 2: Net reliability gap formulas in H35:L40

For each cell in H35:L40, calculate:
`=(H12 - H19) / H26 * 100`

Where:
- H12 block (rows 12-17) = Successful API Requests
- H19 block (rows 19-24) = Failed API Requests  
- H26 block (rows 26-31) = Compute Capacity

The row offset should correspond: row 35 uses rows 12, 19, 26; row 36 uses rows 13, 20, 27; etc.

### Step 2b: Statistics in H42:L47

For each column H through L:
- Row 42: `=MIN(H35:H40)`
- Row 43: `=MAX(H35:H40)`
- Row 44: `=MEDIAN(H35:H40)`
- Row 45: `=AVERAGE(H35:H40)`
- Row 46: `=PERCENTILE(H35:H40, 0.25)`
- Row 47: `=PERCENTILE(H35:H40, 0.75)`

### Step 3: Weighted mean in H50:L50

For each column H through L:
`=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)`

This calculates the weighted mean of Net reliability gap percentages, weighted by Compute Capacity.

### Step 3b: Save and validate
1. Save to `/root/output/result.xlsx` preserving all formatting.
2. Reopen the saved file and verify:
   - Formulas are present in the expected cells (not just values)
   - Print a sample of formula strings from cells H12, H19, H26, H35, H42, H50
   - The file is valid xlsx

### Important notes
- Do NOT add new sheets, macros, VBA, or external links
- Do NOT modify any existing formatting
- Use `openpyxl` to read/write. When writing formulas, assign formula strings (starting with `=`) to cells.
- Before writing any formula, confirm the exact layout of the Data sheet by printing actual cell values.

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