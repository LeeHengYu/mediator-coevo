# Task Instruction

Complete the following task to update an Excel workbook with formulas.

## Phase 0: Inspect the workbook structure

1. Copy the workbook: `cp /root/data/workbook.xlsx /root/output/result.xlsx`
2. Create and run a Python script using `openpyxl` to inspect the workbook structure. Print:
   - Sheet `Task`: All cell values and any fill colors for rows 1-55, columns A-M. Pay special attention to:
     - Column D rows 12-17, 19-24, 26-31 (series codes)
     - Row 10 columns H-L (years)
     - Row 35-40 labels/structure
     - Row 42-47 labels (should be min, max, median, mean, 25th percentile, 75th percentile)
     - Row 50 label
     - The yellow-highlighted cells in H12:L17, H19:L24, H26:L31
   - Sheet `Data`: All cell values for rows 1-40, especially rows 21-38. Print the full grid so we can see the data layout (which column has series codes, which rows/columns have years, etc.)

Print everything clearly with row/column labels so we can understand the exact structure.

## Phase 1: Write formulas (after inspecting output from Phase 0)

Using `openpyxl`, open `/root/output/result.xlsx` and write spreadsheet formulas (NOT computed values) into the cells. Use `openpyxl`'s formula writing capability (just assign formula strings starting with `=` to cells).

### Step 1: Lookup formulas in H12:L17, H19:L24, H26:L31

For each cell in these ranges, write an INDEX+MATCH formula (or VLOOKUP+MATCH) that:
- Uses the series code from column D of that row as one lookup key
- Uses the year from row 10 of that column as the second lookup key
- Looks up the value from the `Data` sheet rows 21:38

The exact formula structure depends on the Data sheet layout discovered in Phase 0. For example, if Data has series codes in column A and years in a header row, use something like:
`=INDEX(Data!$B$21:$XX$38, MATCH($D12, Data!$A$21:$A$38, 0), MATCH(H$10, Data!$B$20:$XX$20, 0))`

Adjust column/row references based on actual inspection. Make sure:
- The series code reference locks the column ($D12) so it doesn't shift horizontally
- The year reference locks the row (H$10) so it doesn't shift vertically
- The Data ranges are fully absolute ($-locked)

### Step 2: Net production slack in H35:L40

Based on the task description, the three blocks are likely:
- H12:L17 = Finished Output
- H19:L24 = Scrap And Rework  
- H26:L31 = Rated Production Capacity

(Verify this from the row labels seen in Phase 0 inspection)

For H35:L40, write formulas: `=(H12-H19)/H26*100` pattern, mapping each of the 6 plants across 5 years. Adjust row references to match the correct corresponding plant rows in each block.

For H42:L47, write:
- Row 42: `=MIN(H35:H40)` (column-wise MIN)
- Row 43: `=MAX(H35:H40)` (column-wise MAX)
- Row 44: `=MEDIAN(H35:H40)` (column-wise MEDIAN)
- Row 45: `=AVERAGE(H35:H40)` (column-wise AVERAGE/mean)
- Row 46: `=PERCENTILE(H35:H40,0.25)` (25th percentile)
- Row 47: `=PERCENTILE(H35:H40,0.75)` (75th percentile)

(Verify the order from row labels in Phase 0)

### Step 3: Weighted mean in H50:L50

For each column H-L: `=SUMPRODUCT(H35:H40, H26:H31)/SUM(H26:H31)`

## Phase 2: Validate

1. Re-open the saved file and verify all target cells contain formula strings (not None/empty).
2. Print all formulas in the target ranges to confirm correctness.
3. Verify no sheets were added or removed.
4. Verify the file saves without errors.

**CRITICAL NOTES:**
- Write FORMULAS, not computed values. Each cell must contain a string starting with `=`.
- Do NOT modify any existing cell formatting, values, or structure outside the target ranges.
- Do NOT add sheets, macros, or VBA.
- Run Phase 0 FIRST, then adapt the formulas based on what you discover about the actual layout before writing anything.

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