# Task Instruction

Execute the following steps to produce /root/output/result.xlsx from /root/data/workbook.xlsx.

## 0 — Inspect
1. Open /root/data/workbook.xlsx with openpyxl (data_only=False) and inspect:
   - Sheet 'Task': print rows 10-50 for columns D-L (values and existing formulas). Pay special attention to:
     • Row 10 (year headers in H10:L10)
     • Column D rows 12-17, 19-24, 26-31 (series codes)
     • Rows 35-40 (port names / labels for Net container flow)
     • Rows 42-47 (statistic labels: min, max, median, mean, 25th, 75th percentile)
     • Row 50 (CPA weighted mean label)
   - Also check which cells have yellow fills (to confirm target ranges).
   - Sheet 'Data': print rows 21-38 to understand the layout — which row holds which series code, which columns hold which years.
2. Print the exact year values in H10:L10 and the exact series codes in D12:D17, D19:D24, D26:D31.
3. Print the Data sheet structure: row 21 headers, and for rows 22-38 print column A (or whichever column holds the series code) and the first few data columns.

## 1 — Step 1: Lookup formulas in H12:L17, H19:L24, H26:L31
For each cell in these three blocks, write a formula that looks up the value from Data!$21:$38 using:
- The series code from column D of the current row on Task sheet
- The year from row 10 of the current column on Task sheet

Use the XLOOKUP+MATCH pattern (preferred) or INDEX+MATCH+MATCH. The exact pattern depends on how the Data sheet is laid out — determine this from inspection.

Example pattern (adjust after inspection):
- If Data has series codes in column A and years across row 21:
  `=INDEX(Data!$B$22:$XX$38, MATCH($D12,Data!$A$22:$A$38,0), MATCH(H$10,Data!$B$21:$XX$21,0))`
  Adjust column/row references to match actual layout.

IMPORTANT: Use absolute references for the Data lookup range and mixed references ($D12 for series code column, H$10 for year row) so formulas can be applied across the block.

## 2 — Step 2: Net container flow (H35:L40) and statistics (H42:L47)
For H35:L40, each row corresponds to a port. The formula is:
  `=(H12 - H19) / H26 * 100`
where row 12 = Loaded Containers Inbound for that port, row 19 = Loaded Containers Outbound, row 26 = Terminal Throughput Capacity. Adjust row offsets so port 1 uses rows 12,19,26; port 2 uses 13,20,27; etc.

For H42:L47, write column-wise statistics over H35:L40:
- H42: `=MIN(H35:H40)`
- H43: `=MAX(H35:H40)`
- H44: `=MEDIAN(H35:H40)`
- H45: `=AVERAGE(H35:H40)`  (simple mean)
- H46: `=PERCENTILE(H35:H40,0.25)`  (25th percentile)
- H47: `=PERCENTILE(H35:H40,0.75)`  (75th percentile)

CRITICAL: Use PERCENTILE (not PERCENTILE.INC or PERCENTILE.EXC) to avoid #NAME? errors in the verifier. Confirm the exact statistic labels in column D/E of rows 42-47 to map the right function to the right row.

## 3 — Step 3: Weighted mean in H50:L50
For each column H through L:
  `=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)`
This computes the weighted mean of the net container flow percentages, weighted by Terminal Throughput Capacity.

## 4 — Save
- Create /root/output/ directory if it doesn't exist.
- Save the workbook to /root/output/result.xlsx.
- Do NOT change any formatting, do NOT add sheets or macros.

## 5 — Validate
- Reopen /root/output/result.xlsx with openpyxl (data_only=False).
- Print formulas in a sample of cells from each block (e.g., H12, L17, H35, L40, H42, H47, H50, L50) to confirm they are correctly written.
- Verify no cells in the target ranges are empty or contain plain values instead of formulas.

## Key Warnings
- From a failed sibling task: using PERCENTILE.INC or PERCENTILE.EXC caused #NAME? errors. Stick to PERCENTILE.
- Inspect before writing — do not assume layout. The exact row/column structure of the Data sheet and the mapping of ports to rows must come from inspection.
- Preserve all existing content and formatting.

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