# Task Instruction

Execute the following phases to complete the workbook task.

## Phase 0 — Structural Inspection

1. Copy the source workbook:
   ```
   cp /root/data/workbook.xlsx /root/output/result.xlsx
   ```
2. Open `/root/output/result.xlsx` with openpyxl and inspect:
   - **Task sheet**: Print the contents of cells D12:D17, D19:D24, D26:D31 (series codes), and H10:L10 (year headers). Also print H35:H40 labels or D35:D40 if present, and any labels in column D or G for rows 35-50. Print the exact content of cells in rows 10-11 around columns D-L to understand headers.
   - **Data sheet**: Print rows 1-5 to understand the header structure, then print rows 21-38 completely (all non-empty columns). Identify: which column holds series codes, which row holds year headers, and the exact data range dimensions.
   - Print the exact cell values (with repr() to reveal hidden spaces or type differences) for a sample series code from Task!D12 and compare it against the matching value in the Data sheet.
   - Print cell types (string vs number) for year values in Task!H10 and the corresponding year header row in the Data sheet.

Do NOT write any formulas yet. Just print all this information.

## Phase 1 — Write Lookup Formulas in H12:L31

Based on Phase 0 findings, write INDEX/MATCH formulas for all three blocks (H12:L17, H19:L24, H26:L31). Use this pattern (adjust ranges based on Phase 0 findings):

```
=INDEX(Data!<data_range>, MATCH(D12, Data!<series_code_column>, 0), MATCH(H$10, Data!<year_header_row>, 0))
```

Critical details:
- The series code reference should use `$D12` (absolute column, relative row) so it locks to column D when dragged across columns H-L.
- The year reference should use `H$10` (relative column, absolute row) so it picks up each year when dragged across.
- Ensure the data_range, series_code_column, and year_header_row exactly match what was found in Phase 0.
- If year values are stored as different types (string vs number), wrap the MATCH argument in `TEXT()` or `VALUE()` as needed, or verify they match naturally.
- After writing formulas, read back a few cells to confirm they are formula strings (not None).

## Phase 2 — Net Production Slack (H35:L40)

For each cell in H35:L40, write a formula that computes:
```
=(H12 - H19) / H26 * 100
```
where:
- Row 12-17 block = Finished Output (first block)
- Row 19-24 block = Scrap And Rework (second block)
- Row 26-31 block = Rated Production Capacity (third block)

So H35 references H12, H19, H26; H36 references H13, H20, H27; etc. Adjust row references accordingly for each of the 6 plants.

Verify by reading back a sample cell.

## Phase 3 — Summary Statistics (H42:L47)

For each column H through L, write these formulas in rows 42-47:
- Row 42: `=MIN(H35:H40)` (minimum)
- Row 43: `=MAX(H35:H40)` (maximum)
- Row 44: `=MEDIAN(H35:H40)` (median)
- Row 45: `=AVERAGE(H35:H40)` (simple mean)
- Row 46: `=PERCENTILE(H35:H40, 0.25)` (25th percentile)
- Row 47: `=PERCENTILE(H35:H40, 0.75)` (75th percentile)

Check the labels in column D/G for rows 42-47 to confirm the correct order (min, max, median, mean, 25th, 75th). If the order differs from what's labeled, match the labels.

## Phase 4 — Weighted Mean (H50:L50)

For each column H through L:
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```
This computes the weighted mean of Net Production Slack using Rated Production Capacity as weights.

## Phase 5 — Save and Validate

1. Save the workbook (it's already at `/root/output/result.xlsx`).
2. Re-open it and spot-check:
   - Read H12, I15, L17 — should be formulas, not None.
   - Read H35, L40 — should be formulas.
   - Read H42, H47, H50 — should be formulas.
3. Confirm no extra sheets were added.
4. Print confirmation that all phases completed successfully.

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