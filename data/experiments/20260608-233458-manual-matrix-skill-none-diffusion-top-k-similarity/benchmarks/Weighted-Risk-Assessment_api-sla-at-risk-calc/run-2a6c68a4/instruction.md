# Task Instruction

Execute the following steps to produce /root/output/result.xlsx from /root/data/workbook.xlsx.

## Phase 0 – Inspect the workbook
1. Open /root/data/workbook.xlsx with openpyxl (data_only=False).
2. Print sheet names to confirm `Task` and `Data` exist.
3. On sheet `Task`:
   - Print rows 10-50 for columns D through L (values) so you can see:
     • The years in row 10 (columns H–L).
     • The series codes in column D for rows 12–17, 19–24, 26–31.
     • The labels in rows 35–40 (Net SLA buffer services), 42–47 (stats), and 50 (Platform SLA Coalition).
   - Note the exact block structure: which rows belong to which metric group.
4. On sheet `Data`:
   - Print rows 21–38 completely (all used columns) to see the lookup table layout: where series codes live, where years live, and where values are.
   - Determine whether the data is arranged with series codes in a column and years across a row, or vice-versa. Identify the exact column that holds series codes and the exact row that holds years.

## Phase 1 – Lookup formulas in H12:L17, H19:L24, H26:L31
For each of the three blocks (rows 12–17, 19–24, 26–31), for each cell in columns H–L:
- Write an INDEX/MATCH formula that:
  • Uses `$D<row>` (absolute column, relative row) as the series-code lookup value.
  • Uses `H$10` (relative column, absolute row) as the year lookup value.
  • References the Data sheet rows 21:38 appropriately.
- Use the inspection results to set the correct ranges:
  • The MATCH for the series code should search the series-code column on Data.
  • The MATCH for the year should search the year row on Data.
  • The INDEX should reference the full data block on Data.
- Use mixed references so formulas are consistent across the block.
- Example pattern (adjust ranges based on inspection):
  `=INDEX(Data!$B$22:$Z$38, MATCH($D12, Data!$A$22:$A$38, 0), MATCH(H$10, Data!$B$21:$Z$21, 0))`
  Adjust $A, $B, $Z, row numbers to match actual layout.

## Phase 2 – Net SLA buffer (H35:L40) and statistics (H42:L47)
For rows 35–40 (six services), for each column H–L:
- Identify which rows in the three blocks above correspond to "Latency Budget Preserved", "Latency Budget Consumed", and "Covered Request Capacity" for each service. The series codes or row labels will tell you. The three blocks likely correspond to these three metrics, with the same six services in each block.
- Write the formula: `=(H12 - H19) / H26 * 100` adjusting row references to map the correct service across blocks. Use the row offset pattern: if block 1 is rows 12–17, block 2 is rows 19–24, block 3 is rows 26–31, then for the first service: `=(H12-H19)/H26*100`, second service: `=(H13-H20)/H27*100`, etc.
- For statistics in H42:L47, write column-wise formulas over H35:H40 (adjust per column):
  • MIN: `=MIN(H35:H40)`
  • MAX: `=MAX(H35:H40)`
  • MEDIAN: `=MEDIAN(H35:H40)`
  • AVERAGE: `=AVERAGE(H35:H40)`
  • 25th percentile: `=PERCENTILE(H35:H40, 0.25)`
  • 75th percentile: `=PERCENTILE(H35:H40, 0.75)`
- Check the labels in column D (or nearby) for rows 42–47 to confirm which row gets which statistic.

## Phase 3 – Weighted mean (H50:L50)
For each column H–L in row 50:
- `=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)`
  This computes the weighted mean of Net SLA buffer percentages using Covered Request Capacity as weights.

## Phase 4 – Save and verify
1. Ensure /root/output/ directory exists (create if needed).
2. Save the workbook to /root/output/result.xlsx.
3. Re-open the saved file and print a sample of cells from each block (e.g., H12, H19, H26, H35, H42, H50) to confirm they contain formulas (not None or bare values).
4. Confirm no extra sheets were added.

## Important constraints
- Do NOT add sheets, macros, VBA, external links, or helper tabs.
- Do NOT change existing formatting.
- Use openpyxl throughout. Set formulas as strings (e.g., cell.value = '=INDEX(...)').
- Carefully verify the actual layout before writing any formula. If the data layout differs from assumptions, adapt accordingly.
- Double-check that the block labels (Latency Budget Preserved, Latency Budget Consumed, Covered Request Capacity) map correctly to the three row ranges.

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