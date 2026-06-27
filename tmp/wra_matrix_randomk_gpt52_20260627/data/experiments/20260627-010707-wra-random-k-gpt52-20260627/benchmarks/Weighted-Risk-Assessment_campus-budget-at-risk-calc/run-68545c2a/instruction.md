# Task Instruction

## Task: Populate formulas in `/root/data/workbook.xlsx` and save to `/root/output/result.xlsx`

### Preparation
1. Create `/root/output/` directory if it doesn't exist.
2. Open `/root/data/workbook.xlsx` and inspect both sheets (`Task` and `Data`) to understand the layout:
   - On `Task` sheet: read row 10 to see the year headers in columns H–L. Read column D rows 12–17, 19–24, 26–31 to see the series codes. Read the labels for rows 35–40, 42–47, and 50.
   - On `Data` sheet: read rows 21–38 to understand the data layout (which row has headers, how series codes and years are arranged, whether data is organized with series codes in a column and years across rows, or vice versa).
3. Print out the exact cell values so you understand the lookup structure before writing any formulas.

### Step 1: Lookup Formulas in H12:L17, H19:L24, H26:L31
For each cell in these three blocks, write a spreadsheet formula that looks up data from `Data` sheet rows 21–38. The formula must use:
- The series code from column D of the current row on `Task` sheet
- The year from row 10 of `Task` sheet (columns H–L)
- One of these patterns: `INDEX/MATCH`, `VLOOKUP/MATCH`, `HLOOKUP/MATCH`, or `XLOOKUP/MATCH`

IMPORTANT: Before writing formulas, determine the exact data layout on `Data` sheet rows 21–38:
- Identify which column contains the series codes (lookup keys)
- Identify which row contains the year headers
- Choose the appropriate lookup pattern based on the data orientation
- Use appropriate absolute/mixed references so formulas can span the block correctly (lock the lookup column for series codes, lock the lookup row for years, etc.)

Write formulas into all 30 cells (6 rows × 5 columns) in each of the three blocks (90 formulas total).

### Step 2: Net Budget Buffer and Summary Statistics in H35:L40 and H42:L47

**H35:L40 — Net Budget Buffer per department:**
The formula is: `(Committed Funding - Operating Spend) / Approved Budget Base * 100`

Based on the three blocks from Step 1:
- One block (H12:L17 or H19:L24 or H26:L31) contains "Committed Funding"
- Another contains "Operating Spend"
- Another contains "Approved Budget Base"

Identify which block is which by reading the labels on the `Task` sheet (likely labels in column A or nearby). Then for each cell in H35:L40, write a formula referencing the corresponding cells from those three blocks. For example, if H12:L17 = Committed Funding, H19:L24 = Operating Spend, H26:L31 = Approved Budget Base, then H35 = (H12 - H19) / H26 * 100. Adjust based on actual layout.

**H42:L47 — Summary statistics (column-wise over H35:L40):**
- Row 42: MIN of H35:H40 (and similarly for columns I–L)
- Row 43: MAX of H35:H40
- Row 44: MEDIAN of H35:H40
- Row 45: AVERAGE of H35:H40
- Row 46: PERCENTILE (or PERCENTILE.INC) of H35:H40 with 0.25
- Row 47: PERCENTILE (or PERCENTILE.INC) of H35:H40 with 0.75

Check the row labels on the `Task` sheet to confirm which row is which statistic, and assign formulas accordingly. Use column-wise ranges (e.g., H35:H40 for column H).

### Step 3: Weighted Mean in H50:L50
For each column (H through L), calculate:
`=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)`

This uses the Net Budget Buffer percentages (H35:H40) as values and the Approved Budget Base block (H26:L31) as weights. Adjust the Approved Budget Base range reference if it's a different block.

Alternatively, if the task says use SUMPRODUCT specifically, write it as:
`=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)`

### Final Steps
1. Save the workbook to `/root/output/result.xlsx`.
2. Reopen the saved file and verify:
   - Sheets `Task` and `Data` still exist (no extra sheets)
   - Cells H12:L17, H19:L24, H26:L31 contain formulas (not hardcoded values)
   - Cells H35:L40 contain formulas
   - Cells H42:L47 contain formulas
   - Cells H50:L50 contain formulas
   - No macros, VBA, or external links were added
3. Print sample computed values from a few cells to sanity-check (values should be reasonable numbers, not errors).

### Important Constraints
- Do NOT add new sheets, macros, VBA, external links, or helper tabs
- Do NOT change existing formatting
- Use `openpyxl` to read and write the workbook
- When writing formulas, write them as strings starting with `=` so they are stored as formulas, not computed values
- Preserve all existing content in the workbook

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