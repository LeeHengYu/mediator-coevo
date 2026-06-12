# Task Instruction

Execute the following steps precisely to complete the task.

## 0. Setup
```bash
mkdir -p /root/output
pip install openpyxl
```

## 1. Inspect the workbook structure
Open `/root/data/workbook.xlsx` with openpyxl and inspect:
- Sheet names
- Sheet `Task`: print rows 10-50 for columns D through L (values and formulas). Pay special attention to:
  - Row 10 (year headers in H10:L10)
  - Column D rows 12-17, 19-24, 26-31 (series codes)
  - What is already in H12:L17, H19:L24, H26:L31 (should be empty/yellow)
  - Row 35-40 column D (port names or references for Net container flow)
  - Rows 42-47 column G or nearby (labels: min, max, median, mean, 25th, 75th percentile)
  - Row 50 (CPA weighted mean)
- Sheet `Data`: print rows 21-38 to understand the data layout. Identify:
  - Which row contains headers (likely row 21 or nearby)
  - Which column contains series codes
  - How years are arranged (likely in a header row)
  - The exact range structure

Print all of this before making any edits.

## 2. Populate lookup formulas in H12:L17, H19:L24, H26:L31

Using openpyxl, write formulas into the yellow cells. For each cell at position (row, col) where col corresponds to H=8, I=9, J=10, K=11, L=12:

Use an INDEX/MATCH/MATCH pattern like:
```
=INDEX(Data!$B$21:$Z$38, MATCH($D{row}, Data!$A$21:$A$38, 0), MATCH(H$10, Data!$B$20:$Z$20, 0))
```

**IMPORTANT**: Before writing formulas, carefully determine from your inspection:
- The exact column in `Data` that contains the series codes (likely column A or B)
- The exact row in `Data` that contains the year headers (likely row 20 or 21)
- The exact data range boundaries
- Adjust the formula references accordingly

The formula must use the series code from column D of the current row (`$D12`, `$D13`, etc.) and the year from row 10 (`H$10`, `I$10`, etc.).

Apply this formula pattern to all 54 cells (6 rows × 5 columns × 3 blocks).

## 3. Calculate Net container flow in H35:L40

The formula for each cell should be:
```
=(H12 - H19) / H26 * 100
```
where the row offsets correspond to the same port:
- H35 = (H12 - H19) / H26 * 100
- H36 = (H13 - H20) / H27 * 100
- H37 = (H14 - H21) / H28 * 100
- H38 = (H15 - H22) / H29 * 100
- H39 = (H16 - H23) / H30 * 100
- H40 = (H17 - H24) / H31 * 100

And similarly for columns I through L.

**IMPORTANT**: Verify the actual row mapping by checking which ports appear in rows 12-17 vs 19-24 vs 26-31 vs 35-40. The ports must align. If rows 19-24 correspond to Loaded Containers Outbound, rows 12-17 to Loaded Containers Inbound, and rows 26-31 to Terminal Throughput Capacity, then the formula offsets above should be correct. Adjust if the inspection reveals different row mappings.

## 4. Summary statistics in H42:L47

For each column (H through L), write these formulas:
- Row 42 (MIN): `=MIN(H35:H40)`
- Row 43 (MAX): `=MAX(H35:H40)`
- Row 44 (MEDIAN): `=MEDIAN(H35:H40)`
- Row 45 (MEAN): `=AVERAGE(H35:H40)`
- Row 46 (25th percentile): `=PERCENTILE(H35:H40, 0.25)`
- Row 47 (75th percentile): `=PERCENTILE(H35:H40, 0.75)`

**IMPORTANT**: Check the labels in column G (or nearby) for rows 42-47 to confirm the exact order (min, max, median, mean, 25th, 75th). Adjust row assignments to match the actual labels.

## 5. Weighted mean for CPA in H50:L50

For each column (H through L):
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```
This computes the weighted mean of Net container flow percentages weighted by Terminal Throughput Capacity.

## 6. Save

Save the workbook to `/root/output/result.xlsx`. Do NOT change any formatting, do NOT add sheets, macros, VBA, external links, or helper tabs.

## 7. Verify

Reopen the saved file and print the formulas in a few sample cells (e.g., H12, L17, H35, H42, H47, H50) to confirm they were written correctly. Also confirm sheet names are unchanged and no extra sheets were added.

## Critical Notes
- Use `openpyxl` and write **string formulas** (not computed values). When opening the workbook, do NOT use `data_only=True`.
- The `$` signs in formulas matter: lock the column for `$D{row}` (so it doesn't shift across columns) and lock the row for `H$10` (so it doesn't shift down rows).
- When writing formulas with openpyxl, the cell value should be a string starting with `=`.
- Preserve all existing content and formatting. Only write into the specified cells.

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