# Task Instruction

Execute the following steps to produce /root/output/result.xlsx from /root/data/workbook.xlsx.

## Phase 0 – Inspect the workbook
1. `mkdir -p /root/output`
2. Open /root/data/workbook.xlsx with openpyxl (data_only=False) and print:
   - Sheet names.
   - Task sheet: values in D12:D17, D19:D24, D26:D31 (series codes), and H10:L10 (year headers).
   - Data sheet: the layout of rows 20-23 and columns A-F to confirm where series codes live (expected Column C) and where year headers live (expected Row 22, or possibly Row 20/21). Also print columns A-Z of row 22 (or whichever row has years) so we know exact column positions.
   - Print a few sample data cells so we can verify the INDEX/MATCH coordinate mapping.

## Phase 1 – Write lookup formulas in H12:L31
Using the inspection results, write INDEX/MATCH formulas into each yellow cell in H12:L17, H19:L24, and H26:L31.

The pattern for cell Hrow (column 8) should be:
```
=INDEX(Data!<data_range>, MATCH($Drow, Data!$C$21:$C$38, 0), MATCH(H$10, Data!<year_header_range>, 0))
```
Adjust the data range and year header range based on what the inspection reveals. Use absolute row references for the year header row ($10) and absolute column reference for the series code column ($D). The data range should cover the numeric block on the Data sheet corresponding to rows 21:38 and the columns holding year data.

Fill all 18 rows × 5 columns = 90 cells with this formula pattern (adjusting column letters H through L).

## Phase 2 – Net container flow formulas in H35:L40
For each of the 6 ports (rows 35-40) and each year column (H-L), write:
```
=(H12 - H19) / H26 * 100
```
where H12 corresponds to the Loaded Containers Inbound row for that port, H19 to Loaded Containers Outbound, and H26 to Terminal Throughput Capacity. Adjust row references per port:
- Row 35: uses rows 12, 19, 26
- Row 36: uses rows 13, 20, 27
- Row 37: uses rows 14, 21, 28
- Row 38: uses rows 15, 22, 29
- Row 39: uses rows 16, 23, 30
- Row 40: uses rows 17, 24, 31

## Phase 3 – Summary statistics in H42:L47
For each year column (H through L):
- Row 42 (Min): `=MIN(H35:H40)`
- Row 43 (Max): `=MAX(H35:H40)`
- Row 44 (Median): `=MEDIAN(H35:H40)`
- Row 45 (Mean): `=AVERAGE(H35:H40)`
- Row 46 (25th percentile): `=PERCENTILE(H35:H40, 0.25)`
- Row 47 (75th percentile): `=PERCENTILE(H35:H40, 0.75)`

Verify the row labels (check cells in column B or C around rows 42-47) to confirm which row is min, max, median, mean, 25th, 75th. Assign formulas accordingly.

## Phase 4 – Weighted mean in H50:L50
For each year column:
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```

## Phase 5 – Save and verify
1. Save as /root/output/result.xlsx, preserving all formatting.
2. Re-open the saved file with data_only=False and print formulas in a sample of cells (e.g., H12, L17, H19, H26, H35, H40, H42, H47, H50) to confirm they are present and correctly structured.
3. Also open with data_only=True (after a quick check) to see if openpyxl cached values look reasonable (they may be None since no calc engine ran, which is fine—the formulas are what matter).

## Important constraints
- Do NOT add new sheets, macros, VBA, external links, or helper columns.
- Do NOT alter existing formatting, merged cells, or other content.
- Use openpyxl only; do not use xlsxwriter or other libraries that would recreate the file from scratch.
- If the Data sheet year headers or series code column differ from the expected positions, adapt all formulas accordingly based on the Phase 0 inspection.

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