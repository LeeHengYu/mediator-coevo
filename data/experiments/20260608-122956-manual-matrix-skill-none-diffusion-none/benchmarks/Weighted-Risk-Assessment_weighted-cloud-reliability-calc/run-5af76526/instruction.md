# Task Instruction

Execute the following steps to produce /root/output/result.xlsx from /root/data/workbook.xlsx.

## Phase 0 – Inspect the workbook
1. `mkdir -p /root/output`
2. Open `/root/data/workbook.xlsx` with openpyxl (data_only=False).
3. Print sheet names.
4. For sheet **Task**:
   - Print cells D12:D17, D19:D24, D26:D31 (series codes).
   - Print row 10, columns H–L (years).
   - Print cells H35:H40 labels or D35:D40 region labels.
   - Print H42:H47 labels (min/max/median/mean/25th/75th).
   - Print H50 label area.
5. For sheet **Data**:
   - Print row 21 through row 38, focusing on the first column (series codes) and the header row that contains years. Identify which column holds the series codes and which row holds the year headers.
   - Determine the exact column-letter range for the data values and the exact row for year headers.

Record all coordinates precisely before writing any formulas.

## Phase 1 – Lookup formulas (H12:L17, H19:L24, H26:L31)
For each block, write an `INDEX/MATCH` formula into every cell in the range. The pattern for cell `HN` (row N, column H) should be:

```
=INDEX(Data!<data_value_range>, MATCH($D{N}, Data!<series_code_column>, 0), MATCH(H$10, Data!<year_header_row>, 0))
```

- `$D{N}` uses a dollar sign on the column so it stays fixed when copied across columns.
- `H$10` uses a dollar sign on the row so it stays fixed when copied down rows.
- `<data_value_range>` is the rectangular block of numeric values on the Data sheet (rows 21–38, columns with year data).
- `<series_code_column>` is the single column on the Data sheet holding series codes (same rows 21–38).
- `<year_header_row>` is the single row on the Data sheet holding years (same columns as data values).

**Critical**: Verify that the series codes in column D of Task sheet match exactly (character-for-character) with those in the Data sheet. Print both side-by-side to confirm. If there are whitespace differences, note them but still use the exact cell reference `$D{N}` so MATCH compares cell-to-cell.

## Phase 2 – Net reliability gap (H35:L40)
Formula for each cell (e.g., H35):
```
=(H12 - H19) / H26 * 100
```
Where H12 = Successful API Requests, H19 = Failed API Requests, H26 = Compute Capacity for the corresponding region and year. Adjust row references for each of the 6 regions (rows 35–40 map to offsets 0–5 from the three blocks).

## Phase 3 – Summary statistics (H42:L47)
For each column (H through L):
- Row 42: `=MIN(H35:H40)`
- Row 43: `=MAX(H35:H40)`
- Row 44: `=MEDIAN(H35:H40)`
- Row 45: `=AVERAGE(H35:H40)`
- Row 46: `=PERCENTILE(H35:H40, 0.25)`  ← use legacy `PERCENTILE`, NOT `PERCENTILE.INC`
- Row 47: `=PERCENTILE(H35:H40, 0.75)`  ← use legacy `PERCENTILE`, NOT `PERCENTILE.INC`

**Important**: Cross-task feedback shows `PERCENTILE.INC` causes `#NAME?` errors. Use `PERCENTILE` only.

## Phase 4 – Weighted mean (H50:L50)
For each column (H through L):
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```
This computes the weighted mean of Net reliability gap percentages weighted by Compute Capacity.

## Phase 5 – Save and verify
1. Save the workbook to `/root/output/result.xlsx`.
2. Re-open it with openpyxl (data_only=False) and print all formula cells in the key ranges to confirm formulas are present and correctly structured.
3. Optionally, re-open with data_only=True (note: openpyxl won't evaluate, but at least confirm no obvious errors in formula strings).

## Constraints
- Do NOT add new sheets, macros, VBA, external links, or helper tabs.
- Do NOT change existing formatting.
- Do NOT use XLOOKUP (may cause #NAME? in some evaluators). INDEX/MATCH is safest.
- Use `PERCENTILE` not `PERCENTILE.INC`.
- All formulas must be Excel formulas (strings starting with `=`), not Python-computed values.

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