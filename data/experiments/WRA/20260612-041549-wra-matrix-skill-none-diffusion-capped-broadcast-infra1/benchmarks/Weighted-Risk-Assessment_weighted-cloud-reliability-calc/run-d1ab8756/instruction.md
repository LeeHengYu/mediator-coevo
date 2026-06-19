# Task Instruction

Execute the following steps in order to produce `/root/output/result.xlsx`.

## Phase 0 – Setup
```bash
mkdir -p /root/output
```

## Phase 1 – Inspect the workbook
Open `/root/data/workbook.xlsx` with openpyxl (data_only=False). On sheet **Data**:
- Print rows 21–39 (columns A–V or so) to identify where series codes live (expected: B22:B39) and where years live (expected: row 21, starting around column C).
- Note the exact column letters for the first and last year.

On sheet **Task**:
- Print rows 10–50, columns D–L, to see:
  - The year headers in row 10 (columns H–L).
  - The series codes in column D for rows 12–17, 19–24, 26–31.
  - The labels/structure in rows 35–50.
- Confirm the three indicator blocks: H12:L17, H19:L24, H26:L31.
- Confirm the six region rows for Net reliability gap (H35:L40), stats rows (H42:L47), and weighted-mean row (H50:L50).

Print everything clearly before proceeding.

## Phase 2 – Populate lookup formulas (H12:L17, H19:L24, H26:L31)
For each cell in these three blocks, write an `INDEX/MATCH` formula that:
- Looks up the series code from column D of that row against `Data!$B$22:$B$39` (exact match).
- Looks up the year from row 10 of that column against `Data!$C$21:$V$21` (or whatever the actual year header range is – adjust based on Phase 1 inspection).
- Returns the value from `Data!$C$22:$V$39` (adjust to actual data range).

Formula pattern (adjust cell refs based on inspection):
```
=INDEX(Data!$C$22:$V$39, MATCH($D12, Data!$B$22:$B$39, 0), MATCH(H$10, Data!$C$21:$V$21, 0))
```

Use `$D12` style (absolute column, relative row) and `H$10` style (relative column, absolute row) so the formula can be written per-cell correctly. Write each formula as a string into the cell.

**Important:** Make sure to use the exact row/column references discovered in Phase 1. Do NOT assume – verify.

## Phase 3 – Net reliability gap (H35:L40)
For each of the six region rows (rows 35–40) and each year column (H–L), write a formula:
```
=(H12 - H19) / H26 * 100
```
where H12 is the Successful API Requests cell, H19 is the Failed API Requests cell, and H26 is the Compute Capacity cell for the same region and year. Adjust row offsets: row 35 corresponds to row 12/19/26, row 36→13/20/27, etc.

**Verify the mapping:** The region in row 35 of the Task sheet should match the region in row 12, row 19, and row 26. Print the D-column values for rows 12–17, 19–24, 26–31, and 35–40 to confirm the series codes or region labels align correctly. If they don't align by simple offset, match by region name.

## Phase 4 – Column-wise statistics (H42:L47)
For each year column (H–L), write these formulas referencing the Net reliability gap block (rows 35–40):
- Row 42 (Minimum): `=MIN(H35:H40)`
- Row 43 (Maximum): `=MAX(H35:H40)`
- Row 44 (Median): `=MEDIAN(H35:H40)`
- Row 45 (Mean): `=AVERAGE(H35:H40)`
- Row 46 (25th percentile): `=PERCENTILE(H35:H40, 0.25)`
- Row 47 (75th percentile): `=PERCENTILE(H35:H40, 0.75)`

**Check the row labels** in column D or nearby to confirm which row is min, max, median, mean, 25th, 75th. Adjust if the order differs from what's listed above.

## Phase 5 – Weighted mean (H50:L50)
For each year column (H–L):
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```
This uses the Net reliability gap values as the "values" and Compute Capacity as weights.

## Phase 6 – Save and validate
1. Save the workbook to `/root/output/result.xlsx`.
2. Re-open the saved file and print cells from each block to confirm formulas were written (not None/empty).
3. Also open with data_only=True (after a quick check) to see if openpyxl shows the formula strings.
4. Run any test script if present: `cd /root && python -m pytest test_output.py -v` (or similar).

## Critical notes
- Do NOT add new sheets, macros, VBA, or external links.
- Do NOT change any existing formatting.
- Use `openpyxl` for all Excel operations.
- All formulas must be written as strings starting with `=`.
- Double-check every range reference against the actual workbook structure found in Phase 1.
- If the region ordering in the derived blocks (rows 35–40) doesn't match the lookup blocks (rows 12–17) by simple offset, explicitly map them by matching region names from column D.

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
Task metadata: author_email=catpaw@meituan.com, author_name=CatPaw Task Engineer, category=spreadsheet-formula-reuse, difficulty=easy, tags=[excel, formulas, lookup, statistics, weighted-mean].
Verifier config: timeout_sec=600.0.