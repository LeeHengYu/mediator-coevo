# Task Instruction

You must update the workbook `/root/data/workbook.xlsx` by injecting Excel formulas into specific cells on the `Task` sheet, then save the result to `/root/output/result.xlsx`. Follow these steps precisely:

## Preliminary Inspection
1. Open `/root/data/workbook.xlsx` with `openpyxl` (with `data_only=False` so formulas are preserved).
2. On sheet `Task`, read:
   - The series codes in column D for rows 12–17, 19–24, 26–31 (these identify which data series to look up).
   - The year headers in H10:L10 (these are the column keys for the lookup).
   - The campus names in column C or B for rows 35–40 (for context; the Net renewable balance rows).
3. On sheet `Data`, inspect rows 21–38 to understand the layout:
   - Identify which column holds the series codes (likely column A or B on Data).
   - Identify which row holds the year headers (likely a header row above row 21, or row 20, or within the block).
   - Determine the exact range for the lookup table so you can build correct INDEX-MATCH or similar formulas.

## Step 1: Populate H12:L17, H19:L24, H26:L31 with lookup formulas
For each cell in these ranges, create a formula that looks up the value using:
- The series code from column D of the same row on `Task`.
- The year from row 10 of the same column on `Task`.
- The data table on `Data!` rows 21:38.

Use an `INDEX`/`MATCH` pattern like:
```
=INDEX(Data!<data_range>, MATCH($D12, Data!<series_code_column>, 0), MATCH(H$10, Data!<year_header_row>, 0))
```
Adjust the exact ranges based on your inspection of the Data sheet. Make sure:
- The series code column reference uses `$D12` (absolute column, relative row).
- The year reference uses `H$10` (relative column, absolute row).
- The INDEX range, MATCH lookup arrays are correctly sized and referenced.

## Step 2: Net Renewable Balance in H35:L40
For each cell in H35:L40, write a formula computing:
```
=(H12 - H19) / H26 * 100
```
where row 12 corresponds to Renewable Generation, row 19 to Grid Consumption, and row 26 to Baseline Energy Demand for the same campus. Adjust row references so that:
- Row 35 uses rows 12, 19, 26
- Row 36 uses rows 13, 20, 27
- Row 37 uses rows 14, 21, 28
- Row 38 uses rows 15, 22, 29
- Row 39 uses rows 16, 23, 30
- Row 40 uses rows 17, 24, 31

Verify this mapping by checking that the campus names in the balance block match the order in the three data blocks above.

## Step 2 (cont): Statistics in H42:L47
For each column H through L, compute column-wise statistics over H35:L40:
- Row 42: `=MIN(H35:H40)` (minimum)
- Row 43: `=MAX(H35:H40)` (maximum)
- Row 44: `=MEDIAN(H35:H40)` (median)
- Row 45: `=AVERAGE(H35:H40)` (simple mean)
- Row 46: `=PERCENTILE(H35:H40, 0.25)` (25th percentile)
- Row 47: `=PERCENTILE(H35:H40, 0.75)` (75th percentile)

**CRITICAL**: Verify which labels are in rows 42–47 by reading column B or C. The order of MIN/MAX/MEDIAN/AVERAGE/PERCENTILE must match the row labels. Do NOT assume the order above—read the labels first and assign formulas accordingly.

**CRITICAL**: Use `PERCENTILE` (not `PERCENTILE.INC` or `PERCENTILE.EXC`) to avoid #NAME? errors in older Excel engines. Similarly, use `AVERAGE` not `MEAN`.

Actually, check: `PERCENTILE.INC` is valid in modern Excel. But the failed hospital-capacity task got #NAME? errors in the percentile rows. To be safe, use `PERCENTILE(range, k)` which is universally supported. If you see the labels say '25th Percentile' and '75th Percentile', use `=PERCENTILE(H35:H40,0.25)` and `=PERCENTILE(H35:H40,0.75)`.

## Step 3: Weighted Mean in H50:L50
For each column H through L:
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```
This computes the weighted mean of the Net renewable balance percentages using Baseline Energy Demand as weights.

## Saving
- Create `/root/output/` directory if it doesn't exist.
- Save the workbook to `/root/output/result.xlsx`.
- Do NOT add any new sheets, macros, VBA, or external links.
- Do NOT change existing formatting.

## Validation
After saving, re-open `/root/output/result.xlsx` and spot-check:
1. That cells in H12:L17 contain formula strings starting with `=INDEX(` or similar.
2. That cells in H35:L40 contain arithmetic formulas.
3. That cells in H42:L47 contain MIN/MAX/MEDIAN/AVERAGE/PERCENTILE formulas.
4. That cells in H50:L50 contain SUMPRODUCT formulas.
5. Print a few formula strings to confirm correctness.

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