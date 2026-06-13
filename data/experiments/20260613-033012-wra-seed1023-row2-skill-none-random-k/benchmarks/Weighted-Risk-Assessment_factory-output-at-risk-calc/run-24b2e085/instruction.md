# Task Instruction

Execute the following steps to produce /root/output/result.xlsx from /root/data/workbook.xlsx.

## Step 0 – Inspect the workbook
1. `mkdir -p /root/output`
2. Open /root/data/workbook.xlsx with openpyxl (data_only=False) and print:
   - Sheet names.
   - Task sheet: values in D12:D17, D19:D24, D26:D31 (series codes), row 10 columns H–L (years), rows 35–40 col D (plant names or codes), row 42–47 col D–G (stat labels), row 50 col D–G (Regional Output Council label).
   - Data sheet: column A rows 21–38 (series codes), row 21 columns A–beyond (year headers), and a sample of the data block so you understand the layout.
   Record the exact series codes, years, row/column positions. This is critical for building correct MATCH references.

## Step 1 – Lookup formulas in H12:L17, H19:L24, H26:L31
For every cell in those three blocks, write an INDEX-MATCH formula of the form:

```
=INDEX(Data!$B$22:$XX$38, MATCH($D12,Data!$A$22:$A$38,0), MATCH(H$10,Data!$B$21:$XX$21,0))
```

Adjust the absolute references to match the actual data extent you discovered in Step 0. Key points:
- Row anchor: the series code in column D of the current row (use $D with relative row).
- Column anchor: the year in row 10 (use relative column with $10).
- The MATCH ranges must cover exactly the series-code column and the year header row on the Data sheet.
- Use 0 (exact match) for both MATCH calls.

## Step 2a – Net production slack (H35:L40)
For each of the six plants (rows 35–40) and each year column (H–L), write:

```
=(H12 - H19) / H26 * 100
```

where H12 is the Finished Output cell, H19 is the Scrap And Rework cell, and H26 is the Rated Production Capacity cell for the same plant and year. Adjust row references so each plant row maps correctly (35→12/19/26, 36→13/20/27, … 40→17/24/31).

## Step 2b – Summary statistics (H42:L47)
For each year column (H–L), in the six stat rows write:
- Row 42 (Min):    =MIN(H35:H40)
- Row 43 (Max):    =MAX(H35:H40)
- Row 44 (Median): =MEDIAN(H35:H40)
- Row 45 (Mean):   =AVERAGE(H35:H40)
- Row 46 (25th):   =PERCENTILE(H35:H40,0.25)
- Row 47 (75th):   =PERCENTILE(H35:H40,0.75)

**IMPORTANT**: Use `PERCENTILE` (not `PERCENTILE.INC` or `PERCENTILE.EXC`) to avoid #NAME? errors in older Excel engines and in the verifier. Confirm the stat label order by checking column D/G of rows 42–47; if the order differs from Min/Max/Median/Mean/25th/75th, match the labels you see.

## Step 3 – Weighted mean (H50:L50)
For each year column (H–L):
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```
This computes the weighted mean of Net production slack using Rated Production Capacity as weights.

## Step 4 – Save
Save the workbook to /root/output/result.xlsx. Do NOT change any formatting, do NOT add sheets or macros.

## Step 5 – Validate
1. Reopen /root/output/result.xlsx with openpyxl (data_only=False).
2. Print formulas in a sample of cells: H12, L17, H19, L24, H26, L31, H35, L40, H42, H47, H50, L50.
3. Confirm every formula cell is a string starting with '=', contains the expected function names (INDEX, MATCH, MIN, MAX, MEDIAN, AVERAGE, PERCENTILE, SUMPRODUCT), and references the correct ranges.
4. Verify no cells in the target ranges are empty or contain plain values where formulas are expected.

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