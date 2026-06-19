# Task Instruction

## Task: Update workbook with formulas and save to /root/output/result.xlsx

### Phase 0: Inspect the workbook
1. `mkdir -p /root/output`
2. Use Python with `openpyxl` to open `/root/data/workbook.xlsx` and inspect:
   - Sheet names (confirm `Task` and `Data` exist).
   - On sheet `Task`: read the layout around rows 10-50, columns D-L. Specifically:
     - Row 10: identify the years in H10:L10.
     - Column D rows 12-17, 19-24, 26-31: identify the series codes.
     - Rows 12-17, 19-24, 26-31: understand what each block represents (likely Loaded Containers Inbound, Loaded Containers Outbound, Terminal Throughput Capacity).
     - Rows 35-40: the six ports for Net container flow.
     - Row 42-47: labels for min, max, median, mean, 25th, 75th percentile.
     - Row 50: Coastal Port Alliance (CPA) weighted mean.
   - On sheet `Data`: read rows 21-38 to understand the data layout (which row has headers, which column has series codes, which columns/rows have year data).
   - Print all of this information so you can construct correct formulas.

### Phase 1: Populate lookup formulas in H12:L17, H19:L24, H26:L31

For each cell in these ranges, write an Excel formula that looks up data from sheet `Data` rows 21:38. The formula must use two inputs:
- The series code from column D of the current row on sheet `Task`
- The year from row 10 on sheet `Task`

Use `INDEX(MATCH, MATCH)` pattern (or one of the other allowed patterns: VLOOKUP+MATCH, HLOOKUP+MATCH, XLOOKUP+MATCH). Choose whichever pattern fits the Data sheet layout best.

IMPORTANT: After inspecting the Data sheet layout, determine:
- Whether series codes are in a row or column on Data sheet
- Whether years are in a row or column on Data sheet
- The exact ranges to reference

Then construct the formula template. For example, if Data has series codes in column A and years in row 20 (or wherever the header row is), an INDEX/MATCH formula might look like:
`=INDEX(Data!$B$21:$F$38, MATCH($D12, Data!$A$21:$A$38, 0), MATCH(H$10, Data!$B$20:$F$20, 0))`

Adjust ranges based on actual inspection. Use absolute references for the data range and mixed references ($D12 for series code column, H$10 for year row) so the formula can be placed in each cell correctly.

Write these formulas using openpyxl by setting each cell's value to the formula string (e.g., `ws['H12'] = '=INDEX(...)'`).

### Phase 2: Net container flow in H35:L40

For each of the 6 ports (rows 35-40) and 5 years (columns H-L), calculate:
`(Loaded Containers Inbound - Loaded Containers Outbound) / Terminal Throughput Capacity * 100`

The three blocks are:
- Loaded Containers Inbound: H12:L17 (rows 12-17)
- Loaded Containers Outbound: H19:L24 (rows 19-24)  
- Terminal Throughput Capacity: H26:L31 (rows 26-31)

Verify which block is which by checking labels on the Task sheet. The formula for H35 would be something like:
`=(H12-H19)/H26*100`

Adjust row references if the port ordering differs between blocks and the Net container flow section. Check if column D in rows 35-40 matches the same ports in the same order as rows 12-17. If not, you may need to adjust the row mapping.

### Phase 3: Summary statistics in H42:L47

For each year column (H through L), calculate column-wise statistics over H35:L40:
- Row 42: `=MIN(H35:H40)` (or whichever row is minimum - check labels)
- Row 43: `=MAX(H35:H40)`
- Row 44: `=MEDIAN(H35:H40)`
- Row 45: `=AVERAGE(H35:H40)` (simple mean)
- Row 46: `=PERCENTILE(H35:H40,0.25)` (25th percentile)
- Row 47: `=PERCENTILE(H35:H40,0.75)` (75th percentile)

Check the actual labels in column D or nearby to confirm which row gets which statistic. Adjust the row assignments accordingly.

### Phase 4: Weighted mean in H50:L50

For each year column, use SUMPRODUCT with the Net container flow percentages (H35:H40) as values and Terminal Throughput Capacity (H26:H31) as weights:
`=SUMPRODUCT(H35:H40,H26:H31)/SUM(H26:H31)`

Apply this for columns H through L in row 50.

### Phase 5: Save and validate
1. Save the workbook to `/root/output/result.xlsx`.
2. Re-open the saved file and verify:
   - All formula cells in H12:L17, H19:L24, H26:L31 contain formula strings (start with `=`).
   - All cells in H35:L40 contain formulas.
   - All cells in H42:L47 contain formulas.
   - All cells in H50:L50 contain formulas.
   - No new sheets were added.
   - Print a sample of formulas from each section to confirm correctness.
3. Do NOT add any sheets, macros, VBA, external links, or helper tabs.

### Critical Notes
- Use `openpyxl` to read and write. Open with `data_only=False` to preserve existing formulas.
- When writing formulas, set cell values as strings starting with `=`.
- Do not alter any existing formatting, merged cells, or other content outside the specified ranges.
- Inspect before writing. Print the actual cell contents and layout before constructing formulas.
- If any cell already has content that should not be overwritten (e.g., labels), skip it.
- Double-check all row/column references against the actual spreadsheet layout.

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