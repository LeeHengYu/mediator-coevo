# Task Instruction

Execute the following steps precisely to produce /root/output/result.xlsx.

## 0. Inspect the workbook
```bash
mkdir -p /root/output
```
Open /root/data/workbook.xlsx with openpyxl (NOT data_only) and inspect:
- Sheet 'Task': read the series codes in D12:D17, D19:D24, D26:D31 (these are the 6 ports × 3 blocks). Read the years in H10:L10. Read the port names in D35:D40. Read any labels in column D for rows 42:47 and row 50.
- Sheet 'Data': read rows 21:38 to understand the layout — identify which row holds which series code, and which columns hold which years. Print the first column (series codes/labels) and the header row so you understand the data orientation.

Print all of this so you can design correct formulas.

## 1. Populate lookup formulas in H12:L17, H19:L24, H26:L31

For each cell in these three blocks, write an INDEX-MATCH formula that:
- Uses the series code from column D of that row
- Uses the year from row 10 of that column
- Looks up in sheet Data rows 21:38

The exact formula pattern (adjust column references after inspection):
```
=INDEX(Data!$B$21:$XX$38, MATCH($D12,Data!$A$21:$A$38,0), MATCH(H$10,Data!$B$20:$XX$20,0))
```
Adjust the column range boundaries based on what you see in the Data sheet. The key is:
- Row lookup: MATCH the series code in column D against the first column of the Data range (rows 21:38)
- Column lookup: MATCH the year in row 10 against the header row of the Data range (likely row 20 or whichever row contains years)
- INDEX into the data block accordingly

Write these as string formulas in openpyxl (e.g., cell.value = '=INDEX(...)').

## 2. Net container flow in H35:L40

For each of the 6 ports (rows 35-40) and 5 year columns (H-L), write:
```
=(H12-H19)/H26*100
```
adjusted so that:
- H12 block = Loaded Containers Inbound (rows 12:17)
- H19 block = Loaded Containers Outbound (rows 19:24)  
- H26 block = Terminal Throughput Capacity (rows 26:31)

The row offset for each port should match: port 1 uses rows 12,19,26; port 2 uses rows 13,20,27; etc.

So for cell H35: =(H12-H19)/H26*100
For cell H36: =(H13-H20)/H27*100
... and so on through row 40 and columns H through L.

## 3. Summary statistics in H42:L47

For each column H through L:
- Row 42 (MIN): =MIN(H35:H40)
- Row 43 (MAX): =MAX(H35:H40)
- Row 44 (MEDIAN): =MEDIAN(H35:H40)
- Row 45 (MEAN): =AVERAGE(H35:H40)
- Row 46 (25th percentile): =PERCENTILE(H35:H40,0.25)
- Row 47 (75th percentile): =PERCENTILE(H35:H40,0.75)

IMPORTANT: Use PERCENTILE (legacy), NOT PERCENTILE.INC. The previous run failed because PERCENTILE.INC was not recognized.

Verify the row labels in column D for rows 42:47 to confirm the order (min, max, median, mean, 25th, 75th). If the order differs from what I assumed, adjust accordingly.

## 4. Weighted mean in H50:L50

For each column H through L:
```
=SUMPRODUCT(H35:H40,H26:H31)/SUM(H26:H31)
```
This computes the weighted mean of net container flow percentages using Terminal Throughput Capacity as weights.

## 5. Save and force evaluation

Save the workbook with openpyxl to /root/output/result.xlsx.

Then force formula evaluation by running:
```bash
cd /root/output
libreoffice --headless --calc --convert-to xlsx result.xlsx --outdir /root/output/
```
This ensures cached values are populated for any verifier that reads with data_only=True.

After conversion, verify the output file exists and open it with openpyxl(data_only=True) to check that cells H12, H35, H42, H46, H47, and H50 have numeric cached values (not None).

## Critical constraints
- Do NOT add new sheets, macros, VBA, external links, or helper tabs.
- Do NOT change existing formatting.
- Use openpyxl to write formulas; preserve all existing content.
- The Data sheet column/row references MUST be verified by inspection before writing formulas.

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