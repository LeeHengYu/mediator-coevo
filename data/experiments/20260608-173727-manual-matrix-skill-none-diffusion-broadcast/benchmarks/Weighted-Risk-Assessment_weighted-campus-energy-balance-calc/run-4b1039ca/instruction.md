# Task Instruction

Execute the following steps in order.

## 1. Inspect the source workbook

Open `/root/data/workbook.xlsx` with openpyxl (data_only=False) and print:

1a. **Task sheet – series codes**: For every row 12–17, 19–24, 26–31, print the exact value in column D (`Task!D12`, `Task!D13`, …, `Task!D31`). Use `repr()` so whitespace/type is visible.

1b. **Task sheet – year headers**: Print `repr()` of cells `Task!H10`, `Task!I10`, `Task!J10`, `Task!K10`, `Task!L10`.

1c. **Data sheet – row keys**: Print `repr()` of every cell in `Data!A21:A38`.

1d. **Data sheet – column headers**: Print `repr()` of every cell in `Data!B20` through the last non-empty cell in row 20 (scan up to column Z).

1e. **Task sheet – campus names and any weights**: Print `repr()` of cells in column C or D for rows 35–40 (the Net renewable balance block) and row 50 (MCEC row). Also print any existing content in `H26:L31` (Baseline Energy Demand block) to see if those are also lookup cells or already filled.

1f. **Data sheet – a sample data row**: Print `repr()` of `Data!A21:Z21` (first data row) so we can see the layout.

Do NOT edit anything yet.

## 2. Build the formulas

After inspecting, write a Python script that opens the workbook, inserts formulas, and saves to `/root/output/result.xlsx`. Follow these rules:

### 2a. Lookup formulas (Step 1)

For every cell in the three blocks `H12:L17`, `H19:L24`, `H26:L31`:
- Use an `INDEX/MATCH` pattern.
- The row-match key is the series code in column D of the **same row** on the Task sheet.
- The column-match key is the year in row 10 of the **same column** on the Task sheet.
- The data source is `Data!` rows 21:38.
- **Critical**: Make the MATCH lookup references match the actual data layout you discovered in step 1. If the series codes are in column A of the Data sheet and the years are in row 20, use those ranges. If the data values span columns B–? and rows 21–38, use that rectangle for INDEX.
- Use exact match (0) for both MATCH calls.
- Example pattern (adjust ranges based on inspection):
  `=INDEX(Data!$B$21:$Z$38, MATCH($D12, Data!$A$21:$A$38, 0), MATCH(H$10, Data!$B$20:$Z$20, 0))`
- If the year headers in `Task!H10:L10` are numbers but `Data!` row 20 stores them as text (or vice versa), wrap the lookup value so types match. For example, if Task years are numbers but Data headers are text, use `MATCH(TEXT(H$10,"0"), …)`. If both are the same type, use a plain reference.

### 2b. Net renewable balance (Step 2 – rows 35:40)

For each cell in `H35:L40`:
`= (H12 - H19) / H26 * 100`  (adjust row references per campus: row 35 uses rows 12,19,26; row 36 uses 13,20,27; etc.)

So for row `r` in 35–40, the offset `k = r - 35`, and the formula references rows `12+k`, `19+k`, `26+k`.

### 2c. Statistics (Step 2 – rows 42:47)

For each column H–L:
- Row 42 (Min): `=MIN(H35:H40)`
- Row 43 (Max): `=MAX(H35:H40)`
- Row 44 (Median): `=MEDIAN(H35:H40)`
- Row 45 (Mean): `=AVERAGE(H35:H40)`
- Row 46 (25th pctl): `=PERCENTILE(H35:H40,0.25)`  ← use legacy `PERCENTILE`, NOT `PERCENTILE.INC`
- Row 47 (75th pctl): `=PERCENTILE(H35:H40,0.75)`

### 2d. Weighted mean (Step 3 – row 50)

For each column H–L:
`=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)`

### 2e. Formatting & saving

- Do NOT modify any existing formatting, sheet names, or other cells.
- Ensure `/root/output/` directory exists (`os.makedirs('/root/output', exist_ok=True)`).
- Save to `/root/output/result.xlsx`.

## 3. Validate

After saving, reopen `/root/output/result.xlsx` with openpyxl (data_only=False) and print:
- The formula strings in cells H12, L17, H19, L24, H26, L31, H35, H40, H42, H47, H50, L50.
- Confirm none are None or empty.

Then reopen with data_only=True and print the cached values of those same cells (they may be None since openpyxl doesn't evaluate, but at least confirm formulas were written).

## 4. Important notes

- Do the full inspection FIRST, then adapt formulas to match the actual data layout.
- If series codes have leading/trailing spaces in one sheet but not the other, use TRIM() in the MATCH formula.
- If year headers are stored as different types, add a type-conversion wrapper.
- Do NOT add sheets, macros, VBA, external links, or helper tabs.
- Do NOT delete or modify any existing cell content outside the specified target ranges.

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