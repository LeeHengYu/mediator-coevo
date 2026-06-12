# Task Instruction

Execute the following multi-step plan to populate formulas in `/root/data/workbook.xlsx` and save the result to `/root/output/result.xlsx`.

## Phase 0: Setup
- `mkdir -p /root/output`

## Phase 1: Inspect the workbook structure thoroughly
Using openpyxl, open `/root/data/workbook.xlsx` and print:
1. **Task sheet:**
   - Row 10 (H10:L10) — the year headers
   - Column D for rows 12–17, 19–24, 26–31 — the series codes
   - Row 35–40 column D — labels for the Net SLA buffer block
   - Row 42–47 column D or whatever column has stat labels (min, max, median, mean, 25th, 75th percentile)
   - Row 50 column D — label for Platform SLA Coalition
   - Check what's in H12:L12 currently (should be empty/yellow)
   - Check H35:L35 currently
   - Check H42:L42 currently
   - Check H50:L50 currently
2. **Data sheet:**
   - Print rows 19–40 to see the full data block structure (all columns A through at least M)
   - Identify: which row contains year headers, which column contains series codes, and where the numeric data matrix starts
   - Print the exact content so we can determine the correct INDEX/MATCH ranges

Print everything clearly with row/column labels. Do NOT write any formulas yet.

## Phase 2: Construct and write formulas
Based on Phase 1 inspection, write a Python script using openpyxl to:

### Step 1: Lookup formulas in H12:L17, H19:L24, H26:L31
For each cell in these three 6×5 blocks, write an INDEX/MATCH formula that:
- Uses the series code from column D of the current row (e.g., `$D12`)
- Uses the year from row 10 (e.g., `H$10`)
- Looks up in the Data sheet rows 21:38 (or whatever the inspection reveals as the data range)
- Pattern: `=INDEX(Data!<data_range>, MATCH($D12, Data!<series_code_column>, 0), MATCH(H$10, Data!<year_header_row>, 0))`
- Adjust the exact ranges based on Phase 1 findings

### Step 2: Net SLA buffer in H35:L40
Formula: `=(H12 - H19) / H26 * 100` (adjusted for actual row positions)
- H12:L17 = Latency Budget Preserved
- H19:L24 = Latency Budget Consumed  
- H26:L31 = Covered Request Capacity
So H35 = `=(H12-H19)/H26*100`, H36 = `=(H13-H20)/H27*100`, etc.

### Step 2b: Statistics in H42:L47
For each column H through L:
- Row 42: `=MIN(H35:H40)`
- Row 43: `=MAX(H35:H40)`
- Row 44: `=MEDIAN(H35:H40)`
- Row 45: `=AVERAGE(H35:H40)`
- Row 46: `=PERCENTILE(H35:H40,0.25)` — use legacy PERCENTILE, NOT PERCENTILE.INC
- Row 47: `=PERCENTILE(H35:H40,0.75)` — use legacy PERCENTILE, NOT PERCENTILE.INC

Verify from Phase 1 which row is which stat label and map accordingly.

### Step 3: Weighted mean in H50:L50
`=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)` for each column H through L.

### Save
Save to `/root/output/result.xlsx`.

## Phase 3: Verify
Reopen `/root/output/result.xlsx` with openpyxl and:
1. Print the formulas (not values) in H12, H19, H26 to confirm INDEX/MATCH structure
2. Print the formula in H35 to confirm Net SLA buffer calculation
3. Print the formulas in H42:H47 to confirm stats
4. Print the formula in H50 to confirm SUMPRODUCT
5. Confirm no new sheets were added
6. Confirm the file exists and is non-empty

## Critical Notes
- Do NOT use PERCENTILE.INC — it causes #NAME? errors. Use PERCENTILE only.
- Do NOT add sheets, macros, VBA, or helper tabs.
- Do NOT change existing formatting.
- The order of stats (min/max/median/mean/25th/75th) must match the row labels found in Phase 1. Inspect carefully.
- Use `$D12` (absolute column) and `H$10` (absolute row) references so formulas can be applied across the grid.
- Execute Phase 1 FIRST and read its output before proceeding to Phase 2.

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