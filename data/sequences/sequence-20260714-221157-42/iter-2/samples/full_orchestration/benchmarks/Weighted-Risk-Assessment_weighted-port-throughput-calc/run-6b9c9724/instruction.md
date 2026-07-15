# Task Instruction

## Task: Populate formulas and calculations in /root/data/workbook.xlsx and save to /root/output/result.xlsx

### Pre-work: Inspect the workbook
1. Create `/root/output/` directory if it doesn't exist.
2. Using openpyxl, open `/root/data/workbook.xlsx` and inspect:
   - **Sheet `Data`**: Print rows 21–38 completely (all columns with data). Pay special attention to column A/B to see the series codes and how data is organized. Also print row 1 or any header row to understand column layout.
   - **Sheet `Task`**: Print the following regions:
     - Column D, rows 12–17 (series codes for block 1)
     - Column D, rows 19–24 (series codes for block 2)
     - Column D, rows 26–31 (series codes for block 3)
     - Row 10 (header row with years in columns H–L)
     - Column D or labels for rows 35–40 (port names for Net container flow)
     - Rows 42–47 column D/E/F/G labels (min, max, median, mean, percentiles)
     - Row 50 labels
     - Also check columns A–G for rows 12–31 to understand the full layout.
   - Print cell fills/colors if possible to confirm yellow cells match H12:L17, H19:L24, H26:L31.

### Step 1: Populate lookup formulas in H12:L17, H19:L24, H26:L31

Based on the inspection, write an openpyxl script that:
- For each cell in ranges H12:L17, H19:L24, and H26:L31, inserts an Excel formula.
- The formula must use one of the allowed patterns: INDEX/MATCH, VLOOKUP/MATCH, HLOOKUP/MATCH, or XLOOKUP/MATCH.
- Each formula uses TWO inputs: (1) the series code from column D of the same row, and (2) the year from row 10 of the same column.
- The lookup source is sheet `Data` rows 21:38.

**CRITICAL**: Before writing formulas, verify:
- The exact column in `Data` that contains the series codes (likely column A or B).
- The exact row in `Data` that contains the year headers.
- Whether the data is arranged with series codes in rows and years in columns, or vice versa.
- Match the series codes in Task!D12:D17 etc. against the actual values in the Data sheet. Print them side by side to confirm they match exactly (watch for whitespace, case differences).

Recommended formula pattern (adjust column/row references based on actual inspection):
- `=INDEX(Data!$B$21:$Z$38, MATCH(D12, Data!$A$21:$A$38, 0), MATCH(H$10, Data!$B$20:$Z$20, 0))`
- Adjust the ranges based on actual data layout discovered during inspection.

### Step 2: Net container flow in H35:L40 and statistics in H42:L47

For H35:L40 (6 ports × 5 years):
- Formula: `(Loaded Containers Inbound - Loaded Containers Outbound) / Terminal Throughput Capacity * 100`
- The three blocks H12:L17, H19:L24, H26:L31 correspond to three different metrics. Determine which block is which by checking the labels in column D or nearby.
- Assuming block 1 (rows 12–17) = Loaded Containers Inbound, block 2 (rows 19–24) = Loaded Containers Outbound, block 3 (rows 26–31) = Terminal Throughput Capacity (verify from labels!).
- Formula for H35: `=(H12-H19)/H26*100` (adjust row references to match the correct port in each block).

For H42:L47 (column-wise statistics):
- H42: `=MIN(H35:H40)`
- H43: `=MAX(H35:H40)`
- H44: `=MEDIAN(H35:H40)`
- H45: `=AVERAGE(H35:H40)`
- H46: `=PERCENTILE(H35:H40,0.25)`
- H47: `=PERCENTILE(H35:H40,0.75)`
- Verify the order of statistics by checking row labels in column D/E for rows 42–47.

### Step 3: Weighted mean in H50:L50
- Use SUMPRODUCT with the net container flow percentages (H35:H40) as values and Terminal Throughput Capacity (H26:H31) as weights.
- Formula for H50: `=SUMPRODUCT(H35:H40,H26:H31)/SUM(H26:H31)`
- Apply across columns H through L.

### Final Steps
- Do NOT change formatting, do NOT add sheets, macros, VBA, external links, or helper tabs.
- Save the workbook to `/root/output/result.xlsx`.
- After saving, reopen the file and verify that formulas are present in the expected cells (print a sample of cells to confirm they contain formula strings, not None or empty values).
- If any formula references seem wrong based on the data layout, fix them before final save.

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