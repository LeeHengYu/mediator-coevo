# Task Instruction

Execute the following steps to complete the weighted port throughput calculation task.

## Step 0: Inspect the workbook
1. Copy the workbook: `cp /root/data/workbook.xlsx /root/output/result.xlsx`
2. Open `/root/output/result.xlsx` with openpyxl and inspect:
   - Sheet `Task`: Read row 10 to find the year headers in columns H–L. Read column D rows 12–17, 19–24, 26–31 to find the series codes. Read row 35–40 column D or B to find the port names/identifiers. Read rows 42–47 column D/E/F/G to find the stat labels (min, max, median, mean, 25th, 75th percentile). Read row 50 to understand the CPA weighted mean row. Read H26:L31 to understand the Terminal Throughput Capacity block location. Note any existing content, formatting, or merged cells.
   - Sheet `Data`: Read rows 21–38 to understand the data layout — identify which row holds the series codes, which column holds the years, and the orientation (whether series codes are in a column and years across a row, or vice versa). Print a representative sample.
3. Print all findings so you have a complete map before writing any formulas.

## Step 1: Populate lookup formulas in H12:L17, H19:L24, H26:L31
For each cell in these three blocks, write a spreadsheet formula (not a computed value) that uses INDEX/MATCH (preferred) or another approved lookup pattern. The formula must:
- Reference the series code from column D of the same row (use a relative row reference or absolute as appropriate).
- Reference the year from row 10 of the same column (use a relative column reference or absolute as appropriate).
- Look up the value from sheet `Data` rows 21:38.
- Use appropriate absolute references (with `$`) so that the series-code column and year row references are correctly anchored.

Use openpyxl to write the formula string into each cell. Do NOT set `data_only=True`. Make sure the formula starts with `=`.

IMPORTANT: openpyxl does not evaluate formulas. Write formula strings (e.g., `=INDEX(Data!$B$21:$B$38,MATCH(...))`) so Excel or the verifier can evaluate them.

## Step 2: Net container flow in H35:L40
For each cell in H35:L40, write a formula that computes:
`(Loaded Containers Inbound - Loaded Containers Outbound) / Terminal Throughput Capacity * 100`
where:
- Loaded Containers Inbound values are in H12:L17
- Loaded Containers Outbound values are in H19:L24
- Terminal Throughput Capacity values are in H26:L31

The formula for cell H35 should be something like: `=(H12-H19)/H26*100` (adjust row references to match the correct port alignment).

## Step 2b: Summary statistics in H42:L47
For each column H through L, write formulas in rows 42–47:
- Row 42 (MIN): `=MIN(H35:H40)` (adjust column)
- Row 43 (MAX): `=MAX(H35:H40)`
- Row 44 (MEDIAN): `=MEDIAN(H35:H40)`
- Row 45 (MEAN): `=AVERAGE(H35:H40)`
- Row 46 (25th percentile): `=PERCENTILE(H35:H40,0.25)` or `=PERCENTILE.INC(H35:H40,0.25)`
- Row 47 (75th percentile): `=PERCENTILE(H35:H40,0.75)` or `=PERCENTILE.INC(H35:H40,0.75)`

Verify the stat labels in column D/E/F/G of rows 42–47 to confirm the correct order of min/max/median/mean/25th/75th. Adjust row assignments accordingly.

## Step 3: Weighted mean in H50:L50
For each column H through L, write a SUMPRODUCT formula:
`=SUMPRODUCT(H35:H40,H26:H31)/SUM(H26:H31)`
This uses the net container flow percentages as values and Terminal Throughput Capacity as weights.

## Step 4: Save and verify
1. Save the workbook to `/root/output/result.xlsx`.
2. Re-open the saved file and print the formula strings in a sample of cells (e.g., H12, L17, H35, L40, H42, H47, H50, L50) to confirm they are correctly written as formula strings (starting with `=`).
3. If any cell contains None or a raw number instead of a formula string (for the lookup and derived cells), investigate and fix.
4. Run any available test script: `cd /root && python -m pytest test_outputs.py -v` or similar. Report results.

## Critical Reminders
- Do NOT use `data_only=True` when opening the workbook for writing.
- Do NOT add new sheets, macros, VBA, or external links.
- Do NOT change existing formatting.
- Write FORMULA STRINGS, not computed Python values.
- The previous failed run on a similar task had cells returning None because formulas were not properly written — double-check every block after saving.
- Inspect the actual Data sheet layout carefully before constructing INDEX/MATCH formulas. The exact column/row structure of Data!21:38 determines whether you need MATCH on a row vs. a column.

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