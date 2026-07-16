# Task Instruction

## Objective

Populate formula cells in `/root/data/workbook.xlsx` sheet `Task` and save the result to `/root/output/result.xlsx`. Do NOT add sheets, macros, VBA, external links, or helper tabs. Preserve all existing formatting.

## Step 0 — Inspect the workbook thoroughly

1. `mkdir -p /root/output`
2. Using openpyxl (with `data_only=False`), open `/root/data/workbook.xlsx` and inspect:
   - **Sheet `Task`**:
     a. Print the exact contents of row 10 columns A–L (these are year headers). Note their types (int, float, string).
     b. Print column D for rows 12–17, 19–24, 26–31 (these are series codes). Note exact strings.
     c. Print any existing content/formulas in H12:L17, H19:L24, H26:L31, H35:L40, H42:L47, H50:L50.
     d. Print rows 33–50 columns A–L to understand the labels for Step 2 and Step 3 blocks.
   - **Sheet `Data`**:
     a. Print rows 19–40 columns A–Z (or however far data extends) to see the full lookup table structure.
     b. Identify: Which row contains years/headers? Which column contains series codes? What is the data layout (series codes in a column, years in a row, values in the body)?
     c. Note the exact data types of the year headers in the Data sheet (int, float, string) and compare with Task row 10.

**Critical**: Print everything literally. Do not assume. The #N/A failure in a sibling task was caused by data-type mismatches between year values in Task row 10 and Data sheet headers.

## Step 1 — Lookup formulas in H12:L17, H19:L24, H26:L31

Based on Step 0 findings, write formulas using INDEX/MATCH (or another allowed pattern) into the yellow cells.

For each cell at row `r`, column `c` (H=8, I=9, J=10, K=11, L=12):
- The series code is in `Task!D{r}`
- The year is in `Task!{col_letter}10` (e.g., H10, I10, etc.)
- The lookup table is on `Data` sheet rows 21:38

Formula pattern (adapt based on actual Data sheet layout discovered in Step 0):
- If Data has series codes in column A and years in a header row (say row 20), and values in the body:
  `=INDEX(Data!B21:??38, MATCH(D{r}, Data!A21:A38, 0), MATCH({col}10, Data!B20:??20, 0))`
- Adjust column ranges and header row based on what you actually find.

**Data type matching**: If Task row 10 has numbers but Data headers are strings (or vice versa), wrap the MATCH lookup_value to convert types. For example, if Task row 10 has numbers and Data headers have numbers, no conversion needed. If there's a mismatch, use `VALUE()` or `TEXT()` or `1*` to coerce. Verify by checking types carefully in Step 0.

Use absolute references for the data range and relative/mixed references appropriately so formulas are consistent across the block. Write formulas cell by cell or use a loop in Python.

## Step 2 — Net budget buffer (H35:L40) and summary statistics (H42:L47)

### H35:L40 — Net budget buffer
Identify which blocks correspond to:
- **Committed Funding**: one of H12:L17, H19:L24, H26:L31
- **Operating Spend**: another of those blocks
- **Approved Budget Base**: the third block

Use the labels from column A/B/C near rows 12, 19, 26 (printed in Step 0) to determine which block is which.

Formula for each cell in H35:L40:
`= (CommittedFunding_cell - OperatingSpend_cell) / ApprovedBudgetBase_cell * 100`

Use cell references (e.g., `=(H12-H19)/H26*100` — adjust row references based on actual block assignments).

### H42:L47 — Summary statistics
For each column (H through L), compute over the 6 values in rows 35:40:
- Row 42: `=MIN(H35:H40)` (or whichever row is minimum per labels)
- Row 43: `=MAX(H35:H40)`
- Row 44: `=MEDIAN(H35:H40)`
- Row 45: `=AVERAGE(H35:H40)`
- Row 46: `=PERCENTILE(H35:H40, 0.25)`
- Row 47: `=PERCENTILE(H35:H40, 0.75)`

**Check the labels in column A/B/C for rows 42–47 to confirm which row gets which function.** Assign formulas according to the actual labels, not the order above.

## Step 3 — Weighted mean in H50:L50

For each column `c` (H through L):
`=SUMPRODUCT({c}35:{c}40, {c}26:{c}31) / SUM({c}26:{c}31)`

Here `{c}35:{c}40` are the Net budget buffer percentages and `{c}26:{c}31` are the Approved Budget Base values (adjust block reference if Approved Budget Base is in a different row range).

The instruction says to use SUMPRODUCT. A valid formula:
`=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)`

## Step 4 — Save and verify

1. Save the workbook to `/root/output/result.xlsx`.
2. Reopen the saved file with `data_only=False` and print:
   - A sample of formulas from each block (H12, L17, H19, L24, H26, L31, H35, L40, H42, H47, H50, L50)
   - Confirm no cells are empty or contain raw values where formulas should be.
3. Optionally open with `data_only=True` (note: openpyxl won't evaluate formulas, so just confirm formulas are present).

## Critical Reminders
- The #N/A failure in the sibling task was due to lookup mismatches. Triple-check that the MATCH arguments reference the correct ranges and data types.
- Do NOT hardcode values. Every yellow cell must contain a formula.
- Do NOT modify any existing formatting, sheets, or non-yellow cells.
- All formulas must use cell references to Task and Data sheets as described.

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