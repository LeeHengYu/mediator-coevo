# Task Instruction

## Task: Populate formulas in `/root/data/workbook.xlsx` and save to `/root/output/result.xlsx`

### Preliminary Investigation

1. **Read the workbook** using `openpyxl` (with `data_only=False` to preserve formulas). Inspect:
   - Sheet names (confirm `Task` and `Data` exist).
   - Sheet `Task`: Print rows 10-50, columns D through L, to understand the layout — especially:
     - Row 10: the year headers in columns H:L.
     - Column D rows 12-17, 19-24, 26-31: the series codes.
     - The structure of rows 35-40 (departments), 42-47 (statistics), and row 50.
   - Sheet `Data`: Print rows 21-38 to understand the data layout — identify which row holds headers (series codes or years), which column holds series codes, and how data is arranged (is it a vertical table with series codes in a column, or a horizontal table with years in a row?).
   - Also check row 1-20 of `Data` if needed to find column headers.
   - Print cell fill colors for a few yellow cells (H12, etc.) to confirm target range.

2. **Determine the lookup orientation** on `Data` sheet rows 21-38:
   - If series codes are in a column and years are in a row header, use INDEX(MATCH, MATCH) or VLOOKUP+MATCH or similar.
   - If the layout is transposed, adjust accordingly.
   - Note the exact column letter and row numbers for the data range on `Data`.

### Step 1: Populate H12:L17, H19:L24, H26:L31 with lookup formulas

For each cell in these three blocks, write a formula that:
- Takes the **series code** from column D of the same row on `Task`.
- Takes the **year** from row 10 of the same column on `Task`.
- Looks up the value from `Data` rows 21:38.

Use one consistent lookup pattern. Preferred: `INDEX(MATCH,MATCH)` since it's the most flexible. The formula pattern should be something like:
```
=INDEX(Data!<data_range>, MATCH($D12, Data!<series_code_column>, 0), MATCH(H$10, Data!<year_row>, 0))
```
Adjust the ranges based on what you find in the investigation step. Use absolute references for the data range and mixed references ($D for column lock, $10 for row lock) so the formula can be consistent across the block.

**Important**: Write actual Excel formula strings into the cells (e.g., `cell.value = '=INDEX(...)'`). Do NOT compute values in Python.

### Step 2: Net budget buffer in H35:L40 and statistics in H42:L47

**H35:L40 — Net budget buffer:**
The formula is: `(Committed Funding - Operating Spend) / Approved Budget Base * 100`

From the layout:
- H12:L17 = one block (likely Committed Funding, Operating Spend, or Approved Budget Base)
- H19:L24 = second block
- H26:L31 = third block

Identify which block is which by reading labels in the Task sheet (likely in column B or C near rows 12, 19, 26). Then for each cell in H35:L40, write a formula referencing the appropriate cells from the three blocks. For example, if row 12 and row 35 correspond to the same department:
```
=(H12 - H19) / H26 * 100
```
Adjust row references based on which block maps to Committed Funding, Operating Spend, and Approved Budget Base.

**H42:L47 — Summary statistics (column-wise):**
For each column H through L:
- H42: `=MIN(H35:H40)`
- H43: `=MAX(H35:H40)`
- H44: `=MEDIAN(H35:H40)`
- H45: `=AVERAGE(H35:H40)`
- H46: `=PERCENTILE(H35:H40, 0.25)`
- H47: `=PERCENTILE(H35:H40, 0.75)`

**Check**: Verify the row labels in column B/C/D near rows 42-47 to confirm the order (min, max, median, mean, 25th, 75th). Adjust the row assignments if the order differs.

### Step 3: Weighted mean in H50:L50

For each column (e.g., H):
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```
This computes the weighted mean of the net budget buffer percentages (H35:H40) weighted by Approved Budget Base (H26:H31).

### Saving

1. Create `/root/output/` directory if it doesn't exist.
2. Save the workbook to `/root/output/result.xlsx` using `openpyxl`.
3. Do NOT change formatting, do NOT add sheets, macros, VBA, or external links.

### Validation

After saving, re-open `/root/output/result.xlsx` and:
- Print the formula in H12, H19, H26, H35, H42, H46, H47, H50 to confirm they are formula strings (start with `=`).
- Confirm no new sheets were added.
- Confirm the file exists and is non-empty.

### Key Cautions
- Use `openpyxl` with `load_workbook(filename, data_only=False)` to preserve existing formulas.
- Write formulas as strings starting with `=`.
- Be very careful about the exact row/column mapping — investigate first, then write formulas.
- The three blocks (rows 12-17, 19-24, 26-31) each have 6 rows for 6 departments. The block in rows 35-40 also has 6 rows. Make sure department ordering matches.
- For SUMPRODUCT weighted mean, use the standard formula `SUMPRODUCT(values, weights)/SUM(weights)`.

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