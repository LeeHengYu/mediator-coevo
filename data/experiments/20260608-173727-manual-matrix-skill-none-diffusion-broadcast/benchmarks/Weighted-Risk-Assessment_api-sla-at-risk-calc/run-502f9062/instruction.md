# Task Instruction

Execute the following steps to produce `/root/output/result.xlsx`:

## 1. Inspect the workbook
- Open `/root/data/workbook.xlsx` using openpyxl.
- Read sheet `Task`: inspect column D (series codes) in rows 12-17, 19-24, 26-31 to understand the six service series codes per block. Inspect row 10 columns H-L to get the five year headers. Print these values.
- Read sheet `Data` rows 21-38 to understand the data layout (column headers, row labels, structure). Print a summary.
- Inspect cells H35:H40 area, H42:H47 area, H50 area to see what labels exist in columns D-G for those rows.

## 2. Populate lookup formulas in H12:L17, H19:L24, H26:L31
For each cell in these three blocks, write an INDEX/MATCH formula that:
- Uses the series code from column D of that row (e.g., `$D12` for row 12)
- Uses the year from row 10 of that column (e.g., `H$10` for column H)
- Looks up in sheet `Data` rows 21:38

Use this pattern (adjust exact Data range based on inspection):
```
=INDEX(Data!$B$21:$XX$38, MATCH($D12,Data!$A$21:$A$38,0), MATCH(H$10,Data!$B$20:$XX$20,0))
```
Adjust the column/row references based on actual Data sheet layout discovered in step 1. The key is:
- Row match: series code in column D against the label column in Data
- Column match: year in row 10 against the header row in Data

## 3. Calculate Net SLA buffer in H35:L40
Formula for each cell: `(LatencyBudgetPreserved - LatencyBudgetConsumed) / CoveredRequestCapacity * 100`

The three blocks correspond to:
- H12:L17 = first metric (e.g., Latency Budget Preserved)
- H19:L24 = second metric (e.g., Latency Budget Consumed)
- H26:L31 = third metric (e.g., Covered Request Capacity)

So for H35: `=(H12-H19)/H26*100` (adjust row offsets to align the same service across blocks).

## 4. Statistics in H42:L47
For each column (H through L), calculate over the 6 values in rows 35:40:
- Row 42: `=MIN(H35:H40)`
- Row 43: `=MAX(H35:H40)`
- Row 44: `=MEDIAN(H35:H40)`
- Row 45: `=AVERAGE(H35:H40)`
- Row 46: `=PERCENTILE(H35:H40,0.25)` — **Use `PERCENTILE` not `PERCENTILE.INC`** to avoid #NAME? errors
- Row 47: `=PERCENTILE(H35:H40,0.75)` — **Use `PERCENTILE` not `PERCENTILE.INC`**

## 5. Weighted mean in H50:L50
For each column: `=SUMPRODUCT(H35:H40,H26:H31)/SUM(H26:H31)`

## 6. Save
- Create `/root/output/` directory if needed.
- Save to `/root/output/result.xlsx`.
- Do NOT add sheets, macros, VBA, external links, or helper tabs.
- Preserve all existing formatting.

## Critical Notes
- **PERCENTILE vs PERCENTILE.INC**: Cross-task feedback confirms that `PERCENTILE.INC` causes `#NAME?` errors in the evaluation engine. Use the legacy `PERCENTILE` function.
- Before writing formulas, print the actual layout of the Data sheet (header row, label column, data range boundaries) to ensure INDEX/MATCH references are correct.
- After writing all formulas, re-read a sample of cells to confirm formulas were written (not just values).
- Use `data_only=False` when loading so existing formulas are preserved.
- When saving, do not modify any cells outside the specified ranges.

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