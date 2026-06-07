# Task Instruction

Complete the following multi-step Excel workbook task. Work carefully and inspect the workbook thoroughly before writing any formulas.

## Phase 0: Inspect the Workbook

1. Copy `/root/data/workbook.xlsx` to `/root/output/result.xlsx` first, then work on the copy.
2. Open and inspect the `Task` sheet thoroughly:
   - Read the contents of column D (especially rows 12-17, 19-24, 26-31) to identify the series codes.
   - Read row 10 (especially columns H through L) to identify the years used as column headers.
   - Read the existing content in rows 35-50 to understand labels and layout.
   - Note what is in H35:L40, H42:L47, and H50:L50 areas (row labels, any existing content).
3. Inspect the `Data` sheet:
   - Read rows 21 through 38 to understand the data layout: which row contains headers, how series codes and years are arranged, what the column structure is.
   - Determine whether the data is arranged with series codes in a column and years across columns, or vice versa. This is critical for choosing the right lookup approach.
   - Identify the exact row and column ranges needed for lookups.

Print all inspected content so you can reference it precisely when writing formulas.

## Phase 1: Populate Lookup Formulas (H12:L17, H19:L24, H26:L31)

For each yellow cell in these three blocks, write a spreadsheet formula that:
- Takes two inputs: the series code from column D of that row, and the year from row 10 of that column.
- Looks up the corresponding value from the `Data` sheet rows 21:38.
- Uses one of these patterns: INDEX/MATCH, VLOOKUP/MATCH, HLOOKUP/MATCH, or XLOOKUP/MATCH.

Choose the lookup pattern that best fits the data layout you observed. Use absolute references (with `$`) where appropriate so formulas can be filled across the range correctly. The series code reference should lock the column (e.g., `$D12`), and the year reference should lock the row (e.g., `H$10`).

Write formulas for all cells: H12:L17 (6 rows × 5 cols = 30 cells), H19:L24 (30 cells), H26:L31 (30 cells).

## Phase 2: Net Reliability Gap (H35:L40)

For each of the 6 regions (rows 35-40) and 5 year columns (H-L), calculate:
`(Successful API Requests - Failed API Requests) / Compute Capacity * 100`

The three data blocks from Step 1 correspond to three metrics. Based on the row labels, identify which block (H12:L17, H19:L24, H26:L31) corresponds to:
- Successful API Requests
- Failed API Requests  
- Compute Capacity

Then write the formula referencing the appropriate cells from those blocks. For example, if row 35 corresponds to the first region, and H12:L17 is Successful API Requests, H19:L24 is Failed API Requests, H26:L31 is Compute Capacity, then H35 = (H12-H19)/H26*100. Adjust based on actual layout.

## Phase 3: Summary Statistics (H42:L47)

For each year column (H through L), calculate column-wise statistics over the 6 Net reliability gap values (rows 35-40):
- Row 42: MIN
- Row 43: MAX
- Row 44: MEDIAN
- Row 45: AVERAGE (simple mean)
- Row 46: PERCENTILE (25th) — use PERCENTILE.INC or PERCENTILE
- Row 47: PERCENTILE (75th)

Check the row labels to confirm which statistic goes in which row. Adjust accordingly.

## Phase 4: Weighted Mean (H50:L50)

For each year column, calculate:
`=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)`
or equivalently use SUMPRODUCT with weights. The values are the Net reliability gap percentages (H35:H40) and weights are the Compute Capacity values (H26:H31). Adjust column references per column.

## Phase 5: Validate

1. Re-read the result file to confirm formulas are in all required cells.
2. Verify no extra sheets were added.
3. Verify formatting is preserved.
4. Save the final file as `/root/output/result.xlsx`.

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