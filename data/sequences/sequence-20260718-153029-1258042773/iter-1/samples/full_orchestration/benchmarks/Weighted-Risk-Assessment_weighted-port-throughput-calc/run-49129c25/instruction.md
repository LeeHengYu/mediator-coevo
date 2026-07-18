# Task Instruction

## Task: Populate formulas in /root/data/workbook.xlsx and save to /root/output/result.xlsx

### Step 0 — Inspect the workbook structure
1. Create `/root/output/` directory if it doesn't exist.
2. Open `/root/data/workbook.xlsx` with openpyxl (data_only=False) and inspect:
   - **Sheet `Task`**: Print rows 10-50 for columns D through L to understand:
     - Row 10: the year headers in H10:L10
     - Column D rows 12-17, 19-24, 26-31: the series codes
     - Rows 35-40: port names and any existing labels
     - Rows 42-47: labels (min, max, median, mean, 25th pctl, 75th pctl)
     - Row 50: CPA weighted mean label
   - **Sheet `Data`**: Print rows 21-38 fully (all columns) to understand the data layout:
     - Which column contains series codes
     - Which row contains year headers
     - How many columns of data exist
   - Print the exact cell values so you know the precise series codes and year positions.

### Step 1 — Lookup formulas in H12:L17, H19:L24, H26:L31
For each block of 6 rows × 5 columns, write an Excel formula using INDEX/MATCH that:
- Looks up the series code from column D of the current row
- Looks up the year from row 10 of the current column
- Searches in the `Data` sheet rows 21:38

Use this pattern (adjust exact Data sheet ranges based on your Step 0 inspection):
```
=INDEX(Data!$B$21:$XX$38, MATCH($D12, Data!$A$21:$A$38, 0), MATCH(H$10, Data!$B$20:$XX$20, 0))
```
Adjust the column/row references based on what you actually find in Step 0:
- The MATCH for series codes should search in whatever column of Data contains the series codes
- The MATCH for years should search in whatever row of Data contains the year headers
- The INDEX range should cover the data area (excluding the header column and header row)

**Important**: Use `$D12` (absolute column, relative row) for the series code and `H$10` (relative column, absolute row) for the year, so the formula copies correctly across the 6×5 block. But write each cell's formula individually with the correct row/column references (don't rely on Excel copy — you're writing via openpyxl).

### Step 2a — Net container flow in H35:L40
The formula for each cell is:
```
=(H12 - H19) / H26 * 100
```
where:
- H12:L17 = Loaded Containers Inbound (first block)
- H19:L24 = Loaded Containers Outbound (second block)  
- H26:L31 = Terminal Throughput Capacity (third block)

For cell at row r (35-40) and column c (H-L):
- Inbound row = r - 23 (so row 35→12, 36→13, ...40→17)
- Outbound row = r - 16 (so row 35→19, 36→20, ...40→24)
- Capacity row = r - 9 (so row 35→26, 36→27, ...40→31)

Write the formula referencing the Task sheet cells directly.

### Step 2b — Summary statistics in H42:L47
For each column (H through L), write these formulas referencing H35:H40 (adjust column letter):
- Row 42 (MIN): `=MIN(H35:H40)`
- Row 43 (MAX): `=MAX(H35:H40)`
- Row 44 (MEDIAN): `=MEDIAN(H35:H40)`
- Row 45 (MEAN): `=AVERAGE(H35:H40)`
- Row 46 (25th percentile): `=PERCENTILE(H35:H40,0.25)`
- Row 47 (75th percentile): `=PERCENTILE(H35:H40,0.75)`

**Important**: Check the actual labels in column D/E/F/G for rows 42-47 during Step 0 to confirm which row is which statistic. Assign formulas according to the actual labels, not assumed order.

### Step 3 — Weighted mean in H50:L50
For each column (H through L):
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```
This uses the Net container flow percentages as values and Terminal Throughput Capacity as weights.

### Step 4 — Save
Save the workbook to `/root/output/result.xlsx`. Do NOT change any formatting, do NOT add sheets, macros, VBA, external links, or helper tabs.

### Step 5 — Verify
Reopen `/root/output/result.xlsx` with openpyxl (data_only=False) and print:
- A sample formula from each block (e.g., H12, H19, H26, H35, H42, H45, H46, H47, H50)
- Confirm they are formula strings (start with '=')
- Confirm no cells in the target ranges are None or empty

### Key Reminders
- The cross-task artifact confirms INDEX/MATCH works well for these lookup tasks.
- Write formulas as strings (e.g., ws['H12'] = '=INDEX(...)'), not computed values.
- All formulas must be Excel formulas, not Python-computed values.
- Preserve all existing formatting and content outside the target cells.

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