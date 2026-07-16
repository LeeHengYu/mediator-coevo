# Task Instruction

## Task: Populate formulas in /root/data/workbook.xlsx and save to /root/output/result.xlsx

### Preparation
1. `mkdir -p /root/output`
2. Install openpyxl if not already available: `pip install openpyxl`
3. Open and inspect `/root/data/workbook.xlsx` thoroughly before making any changes:
   - Read sheet names (expect `Task` and `Data`).
   - On sheet `Task`: print rows 10-50 (columns D through L) to understand the layout — identify series codes in column D, years in row 10, the three lookup blocks (H12:L17, H19:L24, H26:L31), the Net SLA buffer block (H35:L40), the stats block (H42:L47), and the weighted-mean row (H50:L50).
   - On sheet `Data`: print rows 21-38 to understand the source data layout — identify how series codes and years map to values (row headers, column headers, data orientation).
   - Print the exact content of row 10 on `Task` (columns H through L) to see the year values.
   - Print column D rows 12-17, 19-24, 26-31 on `Task` to see the series codes.
   - Identify the orientation of `Data` rows 21:38 — are years in a row and series codes in a column, or vice versa?

### Step 1: Lookup Formulas in H12:L17, H19:L24, H26:L31
For each cell in these three blocks, write a formula that looks up the value from `Data!$21:$38` using:
- The series code from column D of the current row on `Task`
- The year from row 10 of the current column on `Task`

Use one of the allowed patterns: `INDEX(MATCH,MATCH)` is recommended as the most flexible.
- Example pattern (adjust after inspecting Data layout): `=INDEX(Data!<data_range>, MATCH($D12, Data!<series_code_column>, 0), MATCH(H$10, Data!<year_row>, 0))`
- Use absolute references for the lookup arrays and mixed references ($D for column, $10 for row) so formulas copy correctly across the block.
- The exact ranges depend on the Data sheet layout — determine them from inspection.

### Step 2: Net SLA Buffer (H35:L40) and Statistics (H42:L47)

**H35:L40 — Net SLA Buffer:**
The formula is: `(Latency Budget Preserved - Latency Budget Consumed) / Covered Request Capacity * 100`
- Identify which of the three lookup blocks (H12:L17, H19:L24, H26:L31) corresponds to each of these three metrics. The series codes in column D for rows 12-17, 19-24, and 26-31 will tell you.
- For each cell in H35:L40, reference the corresponding cells from the appropriate blocks. For example, if block 1 is Latency Budget Preserved, block 2 is Latency Budget Consumed, and block 3 is Covered Request Capacity, then: `=(H12-H19)/H26*100` for H35, etc. Adjust row offsets to match the six services.
- Use the actual block positions after inspection.

**H42:L47 — Column-wise statistics:**
For each column (H through L), compute over the 6 values in that column of H35:L40:
- Row 42: `=MIN(H35:H40)` (or whichever row is labeled minimum)
- Row 43: `=MAX(H35:H40)`
- Row 44: `=MEDIAN(H35:H40)`
- Row 45: `=AVERAGE(H35:H40)`
- Row 46: `=PERCENTILE(H35:H40, 0.25)` (25th percentile)
- Row 47: `=PERCENTILE(H35:H40, 0.75)` (75th percentile)
- **Check the labels in column D or nearby columns for rows 42-47 to assign the correct function to the correct row.**

### Step 3: Weighted Mean (H50:L50)
For each column H through L:
`=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)`
This uses the Net SLA buffer percentages as values and the Covered Request Capacity as weights. The task says to use `SUMPRODUCT`, so write it in that form.

### Writing Formulas
Use openpyxl to write formulas as strings (e.g., `ws['H12'] = '=INDEX(...)'`). Do NOT compute values in Python — write Excel formula strings so they evaluate in Excel.

### Preservation
- Do NOT modify any existing formatting, styles, or cell values outside the target ranges.
- Do NOT add or remove sheets.
- Do NOT add macros, VBA, external links, or helper tabs.
- Load the workbook with `data_only=False` to preserve existing formulas.
- When saving, keep the same engine defaults to preserve formatting.

### Validation
After writing formulas:
1. Re-read the saved file and print the formula strings in a sample of cells (e.g., H12, L17, H35, H42, H50) to confirm they are correctly written.
2. Verify no extra sheets were added.
3. Save to `/root/output/result.xlsx`.

### Important Notes
- Inspect FIRST, code SECOND. The exact row/column mapping in the Data sheet is critical.
- If Data rows 21-38 have a header row within them, account for that in INDEX/MATCH ranges.
- Double-check that series codes in column D match exactly (including case and spacing) with what's in the Data sheet.

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