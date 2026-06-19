# Task Instruction

Execute the following steps to produce /root/output/result.xlsx from /root/data/workbook.xlsx.

## Phase 0 – Inspect
1. `mkdir -p /root/output`
2. Open /root/data/workbook.xlsx with openpyxl (data_only=False). List sheet names, then:
   - Print Task sheet rows 10-50 (columns D-L) so you can see: the year row (row 10), the series codes in column D for rows 12-17, 19-24, 26-31, the labels in rows 35-40 and 42-47, and row 50.
   - Print Data sheet rows 21-38 to understand the lookup table layout (which column holds series codes, which row holds years, etc.).
   - Note the exact column letters/numbers for series codes and years on Data sheet.
3. Identify the three blocks on Task sheet:
   - Block 1 (H12:L17) – e.g., Loaded Containers Inbound
   - Block 2 (H19:L24) – e.g., Loaded Containers Outbound
   - Block 3 (H26:L31) – e.g., Terminal Throughput Capacity
4. Identify which ports map to which rows in each block (rows 12-17, 19-24, 26-31 each cover 6 ports).

## Phase 1 – Lookup Formulas (H12:L17, H19:L24, H26:L31)
For each cell in these three 6×5 blocks, write an INDEX-MATCH or XLOOKUP formula that:
- Uses the series code from column D of the same row (use $D reference for column lock)
- Uses the year from row 10 (use row-locked reference like H$10)
- Looks up in Data sheet rows 21:38
- Pattern example: `=INDEX(Data!<data_range>,MATCH($D12,Data!<series_col>,0),MATCH(H$10,Data!<year_row>,0))`
- Adjust the exact ranges based on what you see in the Data sheet inspection.
- Apply the formula to all 90 cells (3 blocks × 6 rows × 5 columns).

## Phase 2 – Net Container Flow (H35:L40)
For each of the 6 ports (rows 35-40) and 5 years (columns H-L):
`=(H12-H19)/H26*100` (adjusted for the correct row offsets)
- Row 35 uses data from rows 12, 19, 26 (Port 1)
- Row 36 uses data from rows 13, 20, 27 (Port 2)
- ... and so on for all 6 ports.
Use cell references, not hardcoded values.

## Phase 3 – Summary Statistics (H42:L47)
For each year column (H through L), calculate column-wise stats over H35:L40 (the 6 net-flow values):
- Row 42: `=MIN(H35:H40)`
- Row 43: `=MAX(H35:H40)`
- Row 44: `=MEDIAN(H35:H40)`
- Row 45: `=AVERAGE(H35:H40)`
- Row 46: `=PERCENTILE(H35:H40,0.25)` — use PERCENTILE, NOT PERCENTILE.INC or PERCENTILE.EXC
- Row 47: `=PERCENTILE(H35:H40,0.75)` — use PERCENTILE, NOT PERCENTILE.INC or PERCENTILE.EXC
Check the existing row labels (42-47) to confirm which stat goes in which row. Adjust mapping if the labels differ from the order above.

## Phase 4 – Weighted Mean (H50:L50)
For each year column:
`=SUMPRODUCT(H35:H40,H26:H31)/SUM(H26:H31)`
This computes the CPA weighted mean using net-flow percentages as values and Terminal Throughput Capacity as weights.

## Phase 5 – Save and Validate
1. Save to /root/output/result.xlsx. Do NOT change formatting, do NOT add sheets.
2. Reopen the saved file with openpyxl (data_only=False) and print the formulas in a few sample cells from each block to confirm they are correctly written.
3. Optionally open with data_only=True (note: openpyxl won't compute formulas, but check no cells are None that shouldn't be).

## Critical Rules
- Use `PERCENTILE` not `PERCENTILE.INC` or `PERCENTILE.EXC` — the latter cause #NAME? errors in the verifier.
- Use mixed references ($D12 for series code column, H$10 for year row) so formulas work when filled across the grid.
- Do not add sheets, macros, VBA, external links, or helper tabs.
- Preserve all existing formatting.

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