# Task Instruction

You must update the workbook `/root/data/workbook.xlsx` and save the result to `/root/output/result.xlsx`. Work only inside the existing sheets `Task` and `Data`. Do not add sheets, macros, VBA, external links, or helper tabs. Preserve all existing formatting.

## Preliminary inspection

1. Open `/root/data/workbook.xlsx` with openpyxl (keep formulas: `data_only=False`).
2. Print the sheet names to confirm `Task` and `Data` exist.
3. On sheet `Data`, print rows 19–40 (all columns up to ~column M) so you can see the header row, series codes, and the data layout. Pay special attention to:
   - Which row contains the year headers (likely row 20 or 21).
   - Which column contains the series codes.
   - The range of data columns.
4. On sheet `Task`, print:
   - Row 10 (to see the year headers in columns H–L).
   - Column D, rows 12–31 (to see the series codes for lookups).
   - Rows 26–31 columns D and H–L (Baseline Energy Demand block — you'll need these as weights later).
   - Rows 35–40 column D (to see campus names for Net renewable balance).
   - Row 50 (to see the MCEC row).

Print everything before writing any formulas.

## Step 1 — Lookup formulas in yellow cells

Populate `H12:L17`, `H19:L24`, and `H26:L31` on sheet `Task` with formulas. Each formula must use the series code from column D of that row and the year from row 10 of that column. The data source is sheet `Data` rows 21:38.

Use the `INDEX`/`MATCH` pattern. For each cell (e.g., H12):
```
=INDEX(Data!$B$21:$Z$38, MATCH($D12, Data!$A$21:$A$38, 0), MATCH(H$10, Data!$B$20:$Z$20, 0))
```
Adjust the exact column/row references based on what you observed in the inspection step:
- The first argument to INDEX should be the data body (excluding the series-code column and the header row).
- The first MATCH finds the series code in the series-code column of Data.
- The second MATCH finds the year in the header row of Data.

**Important**: Verify the exact layout before writing formulas. The column letters and row numbers must match the actual Data sheet structure. Double-check by reading a few cells.

After writing all lookup formulas, read back a few cells (e.g., H12, L17, H26, L31) to confirm they contain formula strings (not None).

## Step 2 — Net renewable balance and statistics

Rows 35–40 correspond to six campuses. For each cell in `H35:L40`, write a formula:
```
=(HXX - HYY) / HZZ * 100
```
where:
- `HXX` = the corresponding Renewable Generation cell (from the H12:L17 block)
- `HYY` = the corresponding Grid Consumption cell (from the H19:L24 block)
- `HZZ` = the corresponding Baseline Energy Demand cell (from the H26:L31 block)

For example, H35 = `(H12 - H19) / H26 * 100`, H36 = `(H13 - H20) / H27 * 100`, etc. Adjust column letters for each column H–L.

Then in `H42:L47`, write column-wise statistics formulas over the corresponding column in rows 35:40:
- Row 42: `=MIN(H35:H40)` (minimum)
- Row 43: `=MAX(H35:H40)` (maximum)
- Row 44: `=MEDIAN(H35:H40)` (median)
- Row 45: `=AVERAGE(H35:H40)` (simple mean)
- Row 46: `=PERCENTILE(H35:H40, 0.25)` (25th percentile)  — **use PERCENTILE, not PERCENTILE.INC**
- Row 47: `=PERCENTILE(H35:H40, 0.75)` (75th percentile)  — **use PERCENTILE, not PERCENTILE.INC**

**Critical**: The cross-task feedback shows that `#NAME?` errors occurred from unrecognized function names. openpyxl does not validate Excel function names. Make sure you use `PERCENTILE` (which is universally recognized), not `PERCENTILE.INC` or `PERCENTILE.EXC`. Similarly use `MEDIAN`, `MIN`, `MAX`, `AVERAGE` — all standard names.

However, first check the Task sheet rows 42–47 column A–D to see if labels indicate a different order (e.g., maybe row 42 is min, row 43 is max, etc.). Follow whatever order the labels indicate.

## Step 3 — Weighted mean with SUMPRODUCT

In `H50:L50`, for each column write:
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```
This computes the weighted mean of the Net renewable balance percentages using Baseline Energy Demand as weights.

## Saving

1. Create `/root/output/` directory if it doesn't exist.
2. Save the workbook to `/root/output/result.xlsx`.
3. Re-open the saved file and spot-check a few cells (H12, H35, H42, H46, H50) to confirm formulas are present and not None.

## Key cautions
- Do NOT use `data_only=True` when loading — you need to preserve and write formulas.
- Do NOT use dot-style function names like `PERCENTILE.INC` — use `PERCENTILE`.
- Do NOT modify any existing formatting, sheets, or structure.
- Make sure every formula starts with `=`.
- Adjust all cell references based on actual inspection of the workbook layout.

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