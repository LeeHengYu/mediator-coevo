# Task Instruction

## Task: Update `/root/data/workbook.xlsx` with formulas and save to `/root/output/result.xlsx`

### Phase 0: Inspect the workbook
1. Create `/root/output/` directory if it doesn't exist.
2. Use `openpyxl` (with `data_only=False`) to open `/root/data/workbook.xlsx` and inspect:
   - Sheet `Task`: Print the contents of rows 1–55, columns A–L (values AND any existing formulas). Pay special attention to:
     - Column D rows 12–17, 19–24, 26–31 (series codes)
     - Row 10, columns H–L (years)
     - Rows 35–40 (what labels/structure exists)
     - Rows 42–47 (labels for min/max/median/mean/percentiles)
     - Row 50 (Platform SLA Coalition label)
   - Sheet `Data`: Print rows 1–40, focusing on rows 21–38 to understand the data layout (column headers, row headers, how series codes and years map).
3. Print cell fill colors for a sample yellow cell (e.g., H12) to confirm the target ranges.

### Phase 1: Populate lookup formulas in H12:L17, H19:L24, H26:L31

For each cell in these three blocks, write an `INDEX(MATCH,MATCH)` formula that:
- Uses the series code from column D of the same row as one lookup key
- Uses the year from row 10 of the same column as the other lookup key
- Looks up data from sheet `Data` rows 21:38

Concrete approach:
- First determine the exact data range on `Data` sheet. Identify which column holds the series codes and which row holds the years. For example, if `Data` has series codes in column A rows 21:38 and years in some header row, the formula pattern would be:
  `=INDEX(Data!$B$21:$XX$38, MATCH($D12, Data!$A$21:$A$38, 0), MATCH(H$10, Data!$B$20:$XX$20, 0))`
  Adjust the exact ranges based on what you find in Phase 0.
- Use absolute references for the data range and lookup arrays; use mixed references ($D12 for row-relative series code, H$10 for column-relative year).
- Write the formula to every cell in H12:L17, H19:L24, H26:L31 (6 rows × 5 columns × 3 blocks = 90 cells).

### Phase 2: Net SLA Buffer in H35:L40

For each cell in H35:L40, write a formula:
`=(HXX - HYY) / HZZ * 100`

Where:
- HXX = corresponding cell from the "Latency Budget Preserved" block (one of H12:L17, H19:L24, H26:L31 — determine which block maps to which metric from the row labels)
- HYY = corresponding cell from the "Latency Budget Consumed" block
- HZZ = corresponding cell from the "Covered Request Capacity" block

IMPORTANT: Identify which of the three blocks (rows 12-17, 19-24, 26-31) corresponds to which metric by reading the labels in the Task sheet. The formula is:
`(Latency Budget Preserved - Latency Budget Consumed) / Covered Request Capacity * 100`

The six services in rows 35-40 should correspond to the same six services in each block. Match them by examining column D or other label columns.

### Phase 3: Summary statistics in H42:L47

For each column H through L, in rows 42–47, write formulas for the column-wise statistics over the Net SLA Buffer block (H35:L40 for column H, etc.):
- Row 42: `=MIN(H35:H40)` (or whichever row is labeled minimum)
- Row 43: `=MAX(H35:H40)`
- Row 44: `=MEDIAN(H35:H40)`
- Row 45: `=AVERAGE(H35:H40)`
- Row 46: `=PERCENTILE(H35:H40, 0.25)` (or `PERCENTILE.INC`)
- Row 47: `=PERCENTILE(H35:H40, 0.75)`

IMPORTANT: Match the actual row labels (read in Phase 0) to determine which statistic goes in which row. The labels might be in a different order.

### Phase 4: Weighted mean in H50:L50

For each column H through L:
`=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)`

This uses Net SLA Buffer percentages as values and Covered Request Capacity as weights. Adjust the Covered Request Capacity range reference (H26:H31) based on which block actually contains that metric.

### Phase 5: Save and validate
1. Save the workbook to `/root/output/result.xlsx` preserving all existing formatting (do NOT change fonts, fills, borders, number formats, column widths, etc.).
2. Reopen the saved file and verify:
   - All 90 lookup cells contain formulas (not hardcoded values)
   - H35:L40 contain formulas referencing the three metric blocks
   - H42:L47 contain statistical formulas
   - H50:L50 contain SUMPRODUCT formulas
   - No extra sheets were added
   - Print a sample of formulas from each section to confirm correctness
3. Optionally open with `data_only=True` to check that Excel would compute reasonable values (though openpyxl won't evaluate formulas, the structure should be sound).

### Critical constraints
- Do NOT add new sheets, macros, VBA, external links, or helper tabs
- Do NOT modify existing formatting
- Use `openpyxl` to read and write
- All formulas must use one of the approved lookup patterns: INDEX/MATCH is recommended
- Save to `/root/output/result.xlsx`

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