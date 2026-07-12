# Task Instruction

## Task: Update /root/data/workbook.xlsx with formulas and save to /root/output/result.xlsx

### Phase 0: Inspect the workbook structure
1. Create `/root/output/` directory if it doesn't exist.
2. Use `openpyxl` to open `/root/data/workbook.xlsx` and inspect:
   - Sheet names (confirm `Task` and `Data` exist).
   - On sheet `Task`: read row 10 (the year headers in columns H–L), read column D rows 12–17, 19–24, 26–31 (the series codes), read row 35–40 column D (region names), read rows 42–47 column D or G (stat labels: min, max, median, mean, 25th, 75th percentile), read row 50 (GCM weighted mean row).
   - On sheet `Data`: read rows 21–38 to understand the data layout (column headers, row structure, where series codes and years appear).
   - Print all of this so you understand the exact cell references, series codes, year values, and data layout before writing any formulas.

### Phase 1: Populate lookup formulas in H12:L17, H19:L24, H26:L31

For each cell in these three blocks, write a spreadsheet **formula** (not a computed value). The formula must use one of the allowed lookup patterns: `INDEX/MATCH`, `VLOOKUP/MATCH`, `HLOOKUP/MATCH`, or `XLOOKUP/MATCH`.

The two inputs are:
- The series code from column D of the **current row** on sheet `Task`.
- The year from row 10 of the **current column** on sheet `Task`.

The data source is sheet `Data` rows 21:38. Inspect that range carefully to determine:
- Which column contains the series codes (the match lookup array for series).
- Which row contains the years (the match lookup array for years).
- The data array for the values.

Then construct the formula accordingly. Use absolute references (`$`) where appropriate so the formula correctly references the fixed lookup arrays while the series code (row-dependent) and year (column-dependent) vary.

For example, if Data has series codes in column A rows 21:38 and years in row 20 columns B onward, an INDEX/MATCH/MATCH formula might look like:
`=INDEX(Data!$B$21:$Z$38, MATCH($D12, Data!$A$21:$A$38, 0), MATCH(H$10, Data!$B$20:$Z$20, 0))`

Adjust the exact ranges based on what you observe in the actual data. **Do not guess ranges — read them first.**

IMPORTANT: Use `openpyxl` to write these as string formulas (e.g., `cell.value = '=INDEX(...)'`). Make sure the formula string starts with `=`.

### Phase 2: Net reliability gap formulas in H35:L40

The formula for each cell is:
`(Successful API Requests - Failed API Requests) / Compute Capacity * 100`

The three blocks are:
- H12:L17 — first block (check what series these correspond to)
- H19:L24 — second block
- H26:L31 — third block

You need to identify which block corresponds to "Successful API Requests", which to "Failed API Requests", and which to "Compute Capacity" by reading the labels/headers on the Task sheet (likely in rows 11, 18, 25 or nearby). The six regions in rows 35–40 should correspond to the six rows in each block (rows 12–17, 19–24, 26–31).

Write a formula in each cell H35:L40 that references the appropriate cells from the three blocks. For example, if Successful API Requests is in rows 12–17, Failed is in rows 19–24, and Compute Capacity is in rows 26–31, then H35 would be:
`=(H12-H19)/H26*100`

Adjust row offsets for rows 36–40 accordingly.

### Phase 3: Summary statistics in H42:L47

For each column H through L, compute column-wise statistics over the Net reliability gap values (H35:H40 for column H, etc.):
- Row 42: `=MIN(H35:H40)`
- Row 43: `=MAX(H35:H40)`
- Row 44: `=MEDIAN(H35:H40)`
- Row 45: `=AVERAGE(H35:H40)` (simple mean)
- Row 46: `=PERCENTILE(H35:H40, 0.25)` (25th percentile)
- Row 47: `=PERCENTILE(H35:H40, 0.75)` (75th percentile)

**IMPORTANT**: Check the actual labels in column D (or nearby) for rows 42–47 to confirm which row gets which statistic. Map them correctly — do not assume the order above. Read the labels first, then assign formulas.

### Phase 4: Weighted mean in H50:L50

Use SUMPRODUCT for the weighted mean. The values are the Net reliability gap percentages (H35:H40), and the weights are the Compute Capacity values (H26:H31):
`=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)`

Write this formula for each column H through L in row 50.

### Phase 5: Save and validate
1. Save the workbook to `/root/output/result.xlsx`. When saving with openpyxl, do NOT use `data_only=True` when opening (that strips formulas). Open normally, write formulas, save.
2. Re-open the saved file and verify:
   - All formula cells in H12:L17, H19:L24, H26:L31 contain formula strings (start with `=`).
   - All cells in H35:L40 contain formula strings.
   - All cells in H42:L47 contain formula strings.
   - All cells in H50:L50 contain formula strings.
   - No extra sheets were added.
   - Print a sample of formulas from each block to confirm correctness.
3. Confirm the file exists at `/root/output/result.xlsx`.

### Critical constraints
- Do NOT add sheets, macros, VBA, external links, or helper tabs.
- Do NOT change existing formatting (do not set fonts, fills, borders, etc.).
- Write **formulas**, not hardcoded values.
- Use `openpyxl` for all Excel operations.
- Read the actual workbook structure before writing anything.

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