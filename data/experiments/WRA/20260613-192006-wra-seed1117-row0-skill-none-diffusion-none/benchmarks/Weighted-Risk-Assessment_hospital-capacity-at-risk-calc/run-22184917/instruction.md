# Task Instruction

## Task: Update hospital capacity workbook with formulas

### Overview
You need to read, understand, and update `/root/data/workbook.xlsx` by populating specific cells with spreadsheet formulas, then save the result to `/root/output/result.xlsx`.

### Step 0: Inspect the workbook structure
1. `mkdir -p /root/output`
2. Use `openpyxl` to open `/root/data/workbook.xlsx` and inspect:
   - Sheet `Task`: Print rows 1-55 with all values and formulas. Pay special attention to:
     - Column D (series codes) for rows 12-17, 19-24, 26-31
     - Row 10 (years in columns H-L)
     - The yellow cell ranges: H12:L17, H19:L24, H26:L31
     - Rows 35-40 (Net capacity headroom), 42-47 (statistics), 50 (weighted mean)
     - Any labels in columns A-G that describe what each row block represents
   - Sheet `Data`: Print rows 21-38 to understand the data layout (column headers, row structure, what's in each column)
   - Determine the exact column layout of the Data sheet (which column has the series code, which columns have years/values)
3. Print cell fills/colors for a sample yellow cell to confirm which cells need formulas.

### Step 1: Populate lookup formulas in H12:L17, H19:L24, H26:L31
For each cell in these ranges, write a formula that:
- Takes the series code from column D of that row on sheet `Task`
- Takes the year from row 10 of that column on sheet `Task`
- Looks up the matching value from sheet `Data` rows 21:38
- Uses one of the allowed patterns: INDEX/MATCH, VLOOKUP/MATCH, HLOOKUP/MATCH, or XLOOKUP/MATCH

**Important**: Before writing formulas, determine the Data sheet's layout:
- If Data is organized with series codes in a column and years across columns (or vice versa), choose the appropriate lookup pattern.
- A common robust pattern is `INDEX(Data!$A$21:$Z$38, MATCH(D12, Data!$A$21:$A$38, 0), MATCH(H$10, Data!$A$20:$Z$20, 0))` — but adjust column/row references based on actual data layout.
- Use absolute references for the data range and mixed references (e.g., `$D12` for series code, `H$10` for year) so formulas can be consistent across the block.
- Make sure MATCH is used as part of the formula (required by the task).

### Step 2: Net capacity headroom and statistics

**H35:L40 — Net capacity headroom per cluster:**
Formula: `(Available Care Slots - Occupied Care Slots) / Staffed Bed Capacity * 100`
- Determine which row blocks correspond to "Available Care Slots" (likely H12:L17), "Occupied Care Slots" (likely H19:L24), and "Staffed Bed Capacity" (likely H26:L31) based on the labels you found in Step 0.
- For each cell, e.g., H35: `=(H12-H19)/H26*100` (adjust row references based on actual mapping of the 6 clusters across the 3 blocks).

**H42:L47 — Column-wise statistics over H35:L40:**
- H42: `=MIN(H35:H40)` (minimum)
- H43: `=MAX(H35:H40)` (maximum)  
- H44: `=MEDIAN(H35:H40)` (median)
- H45: `=AVERAGE(H35:H40)` (simple mean)
- H46: `=PERCENTILE(H35:H40,0.25)` (25th percentile)
- H47: `=PERCENTILE(H35:H40,0.75)` (75th percentile)
- **Check the row labels** (column A-G, rows 42-47) to confirm the correct order of min/max/median/mean/p25/p75. Match the formula to the label, not to my assumed order.

### Step 3: Weighted mean in H50:L50
Formula using SUMPRODUCT:
`=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)`
- This computes the weighted mean of the Net capacity headroom percentages (H35:H40) weighted by Staffed Bed Capacity (H26:L31).
- Apply across columns H through L.

### Step 4: Save and verify
1. Save to `/root/output/result.xlsx` using openpyxl, preserving all existing formatting.
2. Re-open the saved file and verify:
   - All yellow cells in H12:L17, H19:L24, H26:L31 contain formulas (not plain values)
   - Each formula includes MATCH
   - H35:L40 contain formulas referencing the three data blocks
   - H42:L47 contain statistical formulas
   - H50:L50 contain SUMPRODUCT formulas
   - No new sheets were added
   - Print a sample of formulas from each block to confirm correctness

### Critical constraints
- Do NOT add sheets, macros, VBA, external links, or helper tabs
- Do NOT change existing formatting (fonts, colors, borders, column widths)
- Use `openpyxl` with `keep_vba=False` (default) and be careful to preserve styles
- When loading, do NOT use `data_only=True` — you need to write formulas, not values
- The formulas must be Excel formulas written as strings starting with `=`

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