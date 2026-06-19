# Task Instruction

Execute the following two-phase plan to populate formulas in /root/data/workbook.xlsx and save to /root/output/result.xlsx.

## Phase 1: Inspect the workbook structure

Run a Python script using openpyxl in read-only/data-only=False mode to print:
1. Sheet 'Task': rows 10-50, columns A-L (print cell values for every cell, especially column D for series codes and row 10 for years)
2. Sheet 'Task': identify exactly which cells in column D correspond to rows 12-17, 19-24, 26-31, 35-40, 42-47, 50
3. Sheet 'Task': row 10 values in columns H-L (the year headers)
4. Sheet 'Data': rows 1-5 to see headers, then rows 21-38 to see the lookup data (print columns A through at least Z)
5. Sheet 'Data': identify the structure - which column has the series codes, which row has years, what the data range looks like

Print everything clearly so the exact cell references can be determined.

## Phase 2: Write formulas based on inspection results

Using the inspection output, write a Python script (openpyxl, data_only=False, no read-only) that:

### Step 1: Lookup formulas in H12:L17, H19:L24, H26:L31
For each of the three blocks (rows 12-17, 19-24, 26-31), populate cells H through L with INDEX/MATCH/MATCH formulas. The pattern should be:
```
=INDEX(Data!$[startcol]$21:$[endcol]$38, MATCH($D12, Data!$[codecol]$21:$[codecol]$38, 0), MATCH(H$10, Data!$[startcol]$[yearrow]:$[endcol]$[yearrow], 0))
```
where:
- The series code reference uses absolute column ($D12, $D19, $D26 etc.) so it locks to column D but the row is relative
- The year reference uses H$10 (column relative, row absolute) so it shifts across columns H-L
- The Data sheet ranges are determined from inspection

Make sure to use the EXACT column letters and row numbers found during inspection. Do NOT guess.

### Step 2: Net production slack in H35:L40
For each cell in H35:L40, write a formula:
```
=(H12-H19)/H26*100
```
where H12 references the Finished Output block (rows 12-17), H19 references the Scrap And Rework block (rows 19-24), and H26 references the Rated Production Capacity block (rows 26-31). The row offsets between blocks should be consistent (e.g., row 35 uses rows 12, 19, 26; row 36 uses rows 13, 20, 27; etc.).

### Step 3: Summary statistics in H42:L47
For each column H through L, write formulas in rows 42-47:
- Row 42 (minimum): =MIN(H35:H40)
- Row 43 (maximum): =MAX(H35:H40)
- Row 44 (median): =MEDIAN(H35:H40)
- Row 45 (simple mean): =AVERAGE(H35:H40)
- Row 46 (25th percentile): =PERCENTILE(H35:H40,0.25)
- Row 47 (75th percentile): =PERCENTILE(H35:H40,0.75)

Verify from the inspection output which row is which statistic. The labels in column D/E/F will tell you the order.

### Step 4: Weighted mean in H50:L50
For each column H through L:
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```
This uses the Net production slack percentages as values and Rated Production Capacity as weights.

### Save
- Ensure mkdir -p /root/output
- Save to /root/output/result.xlsx
- Do NOT change formatting, do NOT add sheets, macros, VBA, external links, or helper tabs

## Validation
After saving, reopen the file with openpyxl (data_only=False) and print the formula content of a sample of cells (e.g., H12, L17, H19, L24, H26, L31, H35, L40, H42, H47, H50, L50) to confirm formulas were written correctly.

## Critical Notes
- The row-to-row mapping between blocks MUST be verified from inspection (the 6 plants in rows 12-17 must correspond to the same 6 plants in rows 19-24, 26-31, and 35-40)
- Column D series codes must match what's in the Data sheet
- Row 10 year values must match what's in the Data sheet header row
- Use PERCENTILE (not PERCENTILE.INC or PERCENTILE.EXC) for compatibility

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