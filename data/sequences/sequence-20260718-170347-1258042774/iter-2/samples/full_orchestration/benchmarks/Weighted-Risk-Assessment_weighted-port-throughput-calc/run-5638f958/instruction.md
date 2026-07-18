# Task Instruction

## Task: Weighted-Risk-Assessment/weighted-port-throughput-calc

You must update an Excel workbook with formulas and save it. Follow these steps precisely.

### Step 0: Inspect the workbook
1. Copy the workbook: `cp /root/data/workbook.xlsx /root/output/result.xlsx`
2. Use `openpyxl` to open `/root/output/result.xlsx` and inspect:
   - Sheet `Task`: Read the structure carefully. Print rows 10-50 for columns D through L to understand:
     - What series codes are in column D for rows 12-17, 19-24, 26-31, 35-40
     - What years are in row 10 for columns H through L
     - What labels are in rows 42-47 (min, max, median, mean, 25th, 75th percentile)
     - What is in row 50 (CPA weighted mean)
   - Sheet `Data`: Print rows 21-38 to understand the data layout (which row has headers, how series codes map, where years are)
3. Print cell formats/fills for a few yellow cells to confirm which cells need formulas.

### Step 1: Populate lookup formulas in H12:L17, H19:L24, H26:L31

For each cell in these ranges, write a formula that looks up data from sheet `Data` rows 21:38. The formula must use the series code from column D of the current row and the year from row 10 of the current column.

Use `INDEX(MATCH, MATCH)` pattern. Before writing formulas, inspect the Data sheet to determine:
- Whether row 21 contains headers or data
- Which column contains series codes and which row contains years
- The exact range references needed

Example pattern (adjust based on actual Data sheet layout):
`=INDEX(Data!$B$22:$F$38, MATCH($D12, Data!$A$22:$A$38, 0), MATCH(H$10, Data!$B$21:$F$21, 0))`

You MUST verify the actual column/row layout before writing formulas. The column that holds series codes and the row that holds years on the Data sheet must be identified by inspection.

### Step 2: Net container flow formulas in H35:L40

For each port (6 ports, rows 35-40), calculate:
`= (Loaded_Inbound - Loaded_Outbound) / Terminal_Throughput_Capacity * 100`

The three blocks from Step 1 correspond to three metrics. Identify which block (rows 12-17, 19-24, 26-31) corresponds to which metric by reading the labels in the Task sheet. Then write formulas referencing the appropriate cells.

For example, if rows 12-17 = Loaded Containers Inbound, rows 19-24 = Loaded Containers Outbound, rows 26-31 = Terminal Throughput Capacity, then:
`H35 = (H12 - H19) / H26 * 100`

### Step 2b: Summary statistics in H42:L47

For each column H through L, calculate column-wise statistics over the 6 net-flow values (rows 35-40):
- Row 42: MIN
- Row 43: MAX  
- Row 44: MEDIAN
- Row 45: AVERAGE (simple mean)
- Row 46: PERCENTILE (25th) - use `PERCENTILE(H35:H40, 0.25)` or `PERCENTILE.INC`
- Row 47: PERCENTILE (75th) - use `PERCENTILE(H35:H40, 0.75)` or `PERCENTILE.INC`

**IMPORTANT**: Verify the actual labels in rows 42-47 before assigning formulas. The order above is a guess - match the actual labels.

### Step 3: Weighted mean in H50:L50

For each column, use SUMPRODUCT to calculate the weighted mean:
`=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)`

This uses the net container flow percentages as values and Terminal Throughput Capacity as weights.

### Important implementation notes:

1. Use `openpyxl` to write formulas as strings (e.g., `ws['H12'] = '=INDEX(...)'`). Do NOT compute values in Python.
2. Do NOT change any formatting, do NOT add sheets, macros, VBA, or helper tabs.
3. When writing formulas, use Excel-style references. Use `$` for absolute references where needed.
4. Save to `/root/output/result.xlsx`.
5. After saving, reopen the file and verify that formulas were written to the expected cells (spot-check a few cells in each range).
6. Make sure `/root/output/` directory exists before saving.

### Verification checklist:
- All cells in H12:L17, H19:L24, H26:L31 contain lookup formulas
- All cells in H35:L40 contain net flow formulas
- All cells in H42:L47 contain statistical formulas
- All cells in H50:L50 contain SUMPRODUCT weighted mean formulas
- No formatting changes, no extra sheets
- File saved to `/root/output/result.xlsx`

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