# Task Instruction

Execute the following steps precisely to complete the task.

## 0. Setup
```bash
mkdir -p /root/output
cp /root/data/workbook.xlsx /root/output/result.xlsx
```

## 1. Inspect the workbook structure
Open `/root/output/result.xlsx` using openpyxl. Inspect:
- Sheet names (should be `Task` and `Data`).
- On sheet `Task`: read row 10 to find the years in columns H–L. Read column D rows 12–17, 19–24, 26–31 to find the series codes. Read row 35–40 labels/column D to understand the six services. Read rows 42–47 column G (or nearby) to find which stat is expected (min, max, median, mean, 25th pctl, 75th pctl). Read row 50 label.
- On sheet `Data`: inspect rows 21–38 to understand the data layout — identify which column holds series codes, which row holds years, and where the numeric data lives. Print the header row of the data block and a few sample rows.

Print all of this so we understand the exact layout before writing any formulas.

## 2. Populate lookup formulas in H12:L17, H19:L24, H26:L31

Using openpyxl, write Excel formulas (not computed values) into each yellow cell. For each cell at row `r`, column `c` (H=8, I=9, J=10, K=11, L=12):

- The series code is in `$D{r}` (column D of the same row).
- The year is in the corresponding column of row 10, e.g., `H$10`, `I$10`, etc.
- The data source is on sheet `Data`, rows 21:38.

Based on the data layout discovered in step 1, construct an INDEX/MATCH/MATCH formula or XLOOKUP with MATCH. The most likely pattern (adjust based on actual layout):

```
=INDEX(Data!<data_range>, MATCH($D{r}, Data!<series_code_column>, 0), MATCH({col}$10, Data!<year_header_row>, 0))
```

Make sure:
- The series code reference uses `$D` (absolute column) so it doesn't shift across columns.
- The year reference uses `$10` (absolute row) so it doesn't shift across rows.
- The Data range references are absolute.
- Use the exact range discovered in step 1.

Write these formulas for all 54 cells (3 blocks × 6 rows × 5 columns).

## 3. Net SLA Buffer formulas in H35:L40

For each cell at row `r` in 35–40 and column `c` in H–L:
- Identify which rows correspond to `Latency Budget Preserved` (likely rows 12–17), `Latency Budget Consumed` (likely rows 19–24), and `Covered Request Capacity` (likely rows 26–31).
- The service in row `r` of the 35–40 block corresponds to the same relative position (1st–6th) in each of the three blocks above.
- Formula: `=({col}{preserved_row} - {col}{consumed_row}) / {col}{capacity_row} * 100`
  For example if row 35 corresponds to the 1st service: `=(H12-H19)/H26*100`

Verify the mapping by checking that column D labels in rows 35–40 match the service names in rows 12–17 (or are in the same order). Adjust row references if the order differs.

## 4. Summary statistics in H42:L47

For each column `c` (H–L), write these formulas referencing the Net SLA buffer range `{c}35:{c}40`:
- Row 42 (check label — likely MIN): `=MIN({c}35:{c}40)`
- Row 43 (MAX): `=MAX({c}35:{c}40)`
- Row 44 (MEDIAN): `=MEDIAN({c}35:{c}40)`
- Row 45 (MEAN): `=AVERAGE({c}35:{c}40)`
- Row 46 (25th percentile): `=PERCENTILE({c}35:{c}40, 0.25)`
- Row 47 (75th percentile): `=PERCENTILE({c}35:{c}40, 0.75)`

**Important**: Match each formula to the actual label in column G (or nearby) for that row. Print the labels first and assign formulas accordingly. The order may differ from what I listed.

## 5. Weighted mean in H50:L50

For each column `c` (H–L):
```
=SUMPRODUCT({c}35:{c}40, {c}26:{c}31) / SUM({c}26:{c}31)
```
This computes the weighted mean of Net SLA Buffer percentages weighted by Covered Request Capacity.

## 6. Save and verify
- Save the workbook (it's already at `/root/output/result.xlsx`).
- Re-open it and print a sample of the formulas written to confirm they are formula strings (start with `=`), not computed values.
- Verify no new sheets were added.
- Verify the file is valid by loading it without errors.

## Key constraints
- Use openpyxl in Python. Do NOT use data_only mode when writing.
- Write Excel formula strings, not Python-computed values, for all cells.
- Do not modify any existing formatting, do not add sheets, macros, VBA, or external links.
- If any cell reference or range doesn't match what you discover in step 1, adjust accordingly — the inspection step is critical.
- Do not use PERCENTILE.INC or PERCENTILE.EXC — use plain PERCENTILE (which is equivalent to PERCENTILE.INC in Excel).

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