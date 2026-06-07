# Task Instruction

## Task: Populate formulas in /root/data/workbook.xlsx and save to /root/output/result.xlsx

### Overview
You must open `/root/data/workbook.xlsx`, add spreadsheet **formulas** (not hardcoded values) to specific cell ranges on the existing `Task` sheet, and save the result to `/root/output/result.xlsx`. Do NOT add sheets, macros, VBA, external links, or helper tabs. Preserve all existing formatting.

### Step 0: Inspect the workbook thoroughly
1. Open `/root/data/workbook.xlsx` using openpyxl (with `data_only=False` to preserve formulas).
2. Read the `Task` sheet: inspect row 10 (the year headers in columns H through L), column D rows 12-31 (series codes), rows 26-31 (Baseline Energy Demand block), row 35-40 labels, row 42-47 labels, row 50 label.
3. Read the `Data` sheet: inspect rows 21-38 to understand the data layout — identify which row contains which series code, and which columns contain which years. Pay special attention to whether the data is arranged with years in columns or rows.
4. Print out all of these inspected values so you can construct correct formulas.

### Step 1: Populate lookup formulas in H12:L17, H19:L24, H26:L31
For each cell in these three blocks, write an Excel formula that looks up data from `Data!$21:$38` using:
- The series code from column D of the current row on `Task`
- The year from row 10 on `Task`

Use one of these patterns: `INDEX/MATCH`, `VLOOKUP/MATCH`, `HLOOKUP/MATCH`, or `XLOOKUP/MATCH`.

**Critical**: Before writing formulas, verify the exact layout of the Data sheet:
- Are years in a header row? Which row? Which columns?
- Are series codes in a column? Which column?
- Use absolute references (`$`) where appropriate to allow the formula to be consistent across the block.

A typical INDEX/MATCH pattern might be:
`=INDEX(Data!$B$21:$XX$38, MATCH($D12, Data!$A$21:$A$38, 0), MATCH(H$10, Data!$B$20:$XX$20, 0))`
But you MUST adjust the exact ranges based on your inspection of the Data sheet.

**Important openpyxl note**: When writing formulas with openpyxl, do NOT use `data_only=True`. Write the formula as a string starting with `=`. Make sure you do NOT accidentally wrap the formula in extra quotes.

### Step 2: Net renewable balance in H35:L40
For each of the 6 campuses (rows 35-40) and each year (columns H-L), calculate:
`= (Renewable_Generation - Grid_Consumption) / Baseline_Energy_Demand * 100`

where:
- Renewable Generation values are in H12:L17 (the first block)
- Grid Consumption values are in H19:L24 (the second block)  
- Baseline Energy Demand values are in H26:L31 (the third block)

**Verify**: Check which block corresponds to which metric by reading the labels in column B or C near rows 12, 19, 26. The mapping above is a hypothesis — adjust based on actual labels.

So for H35: `=(H12-H19)/H26*100` (adjust row references based on actual block mapping).

### Step 2b: Statistics in H42:L47
For each year column (H through L), calculate column-wise statistics over the 6 net-balance values (H35:H40 etc.):
- Row 42: MIN
- Row 43: MAX  
- Row 44: MEDIAN
- Row 45: AVERAGE (simple mean)
- Row 46: 25th percentile — use `PERCENTILE` (not `PERCENTILE.INC` or `PERCENTILE.EXC` unless you verify the verifier expects it; `PERCENTILE` is safest for compatibility)
- Row 47: 75th percentile — use `PERCENTILE`

**Critical warning from cross-task feedback**: A similar task failed with `#NAME?` errors on percentile/statistics functions. Use `PERCENTILE` (not `PERCENTILE.INC`) and `MEDIAN` (not `MEDIAN.INC`). These are the universally recognized function names.

**Verify**: Check the actual labels in column B/C/D for rows 42-47 to confirm which statistic goes in which row.

### Step 3: Weighted mean in H50:L50
For each year column, calculate the weighted mean using SUMPRODUCT:
`=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)`

This uses the net balance percentages as values and Baseline Energy Demand as weights.

### Step 4: Save
1. Create `/root/output/` directory if it doesn't exist.
2. Save the workbook to `/root/output/result.xlsx`.

### Step 5: Validate
1. Reopen `/root/output/result.xlsx` with openpyxl (data_only=False).
2. Check that cells H12, L17, H19, L24, H26, L31, H35, L40, H42, L47, H50, L50 all contain formula strings (starting with `=`).
3. Print several formula samples to confirm they reference the correct sheets and ranges.
4. Also open with data_only=True to check if any cells return None (which would indicate formula errors or that the file needs to be opened in Excel — this is expected with openpyxl but the formulas should still be present in data_only=False mode).

### Key Pitfalls to Avoid
- Do NOT hardcode values; use Excel formulas.
- Do NOT use `PERCENTILE.INC` or `PERCENTILE.EXC` — use `PERCENTILE`.
- Do NOT add new sheets or modify sheet names.
- Do NOT lose existing formatting (use openpyxl carefully, avoid overwriting styles).
- Make sure formula references to the Data sheet use the exact sheet name (e.g., `Data!...`).
- Verify all row/column references by inspection before writing formulas.

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