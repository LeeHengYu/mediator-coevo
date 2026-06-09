# Task Instruction

## Task: Update /root/data/workbook.xlsx with formulas and save to /root/output/result.xlsx

### Phase 0: Inspect the workbook structure
1. Create `/root/output/` directory if it doesn't exist.
2. Open `/root/data/workbook.xlsx` using openpyxl and inspect:
   - Sheet names (confirm `Task` and `Data` exist).
   - On sheet `Task`: read row 10 (especially H10:L10) to see the year headers. Read column D rows 12-17, 19-24, 26-31 to see the series codes. Read rows 35-40 to understand the service labels. Read row 42-47 labels (min, max, median, mean, 25th, 75th percentile). Read row 50 label.
   - On sheet `Data`: read rows 21-38 to understand the data layout — identify which row contains headers, which column has series codes, and how years are arranged.
   - Print all of this information so you understand the exact cell references, labels, and data layout before writing any formulas.

### Phase 1: Populate lookup formulas in H12:L17, H19:L24, H26:L31

For each yellow cell in these three blocks, write a spreadsheet formula (not a Python-computed value) that looks up data from sheet `Data` rows 21:38. The formula must use two inputs:
- The series code from column D of the current row on sheet `Task`
- The year from row 10 of the current column on sheet `Task`

Use an `INDEX(MATCH, MATCH)` pattern (or `VLOOKUP` with `MATCH`, `HLOOKUP` with `MATCH`, or `XLOOKUP` with `MATCH`). For example, an INDEX/MATCH pattern might look like:
```
=INDEX(Data!$B$21:$XX$38, MATCH($D12, Data!$A$21:$A$38, 0), MATCH(H$10, Data!$B$20:$XX$20, 0))
```
But you MUST adjust the exact ranges based on what you observe in the actual workbook. Key things to verify:
- Which column on `Data` contains the series codes (could be column A, B, etc.)
- Which row on `Data` contains the year headers
- The exact extent of the data range

Write these as string formulas in openpyxl (e.g., `cell.value = '=INDEX(...)'`). Make sure to use absolute references where needed ($ signs) so that the series code reference locks the column and the year reference locks the row.

### Phase 2: Net SLA Buffer formulas in H35:L40

For each cell in H35:L40, write a formula that computes:
```
(Latency Budget Preserved - Latency Budget Consumed) / Covered Request Capacity * 100
```
where:
- "Latency Budget Preserved" values are in one of the three blocks (H12:L17, H19:L24, H26:L31)
- "Latency Budget Consumed" values are in another block
- "Covered Request Capacity" values are in the third block (specifically H26:L31 based on the instruction)

Identify which block corresponds to which metric by reading the labels on the Task sheet (likely around rows 11, 18, 25). The formula should reference the corresponding cells from those blocks. For example, if row 12 and row 35 correspond to the same service, then H35 might be:
```
=(H12 - H19) / H26 * 100
```
Adjust based on actual layout.

### Phase 3: Summary statistics in H42:L47

For each column H through L, write formulas for the six statistics over the Net SLA Buffer values (H35:H40 for column H, etc.):
- Row 42: `=MIN(H35:H40)`
- Row 43: `=MAX(H35:H40)`
- Row 44: `=MEDIAN(H35:H40)`
- Row 45: `=AVERAGE(H35:H40)`
- Row 46: `=PERCENTILE(H35:H40, 0.25)` (or `PERCENTILE.INC`)
- Row 47: `=PERCENTILE(H35:H40, 0.75)` (or `PERCENTILE.INC`)

Verify the row assignments by checking the labels in column D or nearby columns for rows 42-47.

### Phase 4: Weighted mean in H50:L50

For each column H through L, write a SUMPRODUCT formula:
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```
This computes the weighted mean of Net SLA Buffer percentages weighted by Covered Request Capacity.

### Phase 5: Save and validate
1. Save the workbook to `/root/output/result.xlsx` preserving all existing formatting. When opening the workbook use `keep_vba=False` and do NOT use `data_only=True` (we want formulas preserved).
2. Re-open the saved file and verify:
   - Sheets are still only `Task` and `Data` (no extra sheets added).
   - Spot-check that cells H12, L17, H19, L24, H26, L31 contain formula strings (start with '=').
   - Spot-check that H35, H42, H47, H50 contain formula strings.
   - Print several formula values to confirm they look correct.
3. Do NOT add any macros, VBA, external links, or helper sheets.

### Critical Notes
- You MUST inspect the actual workbook structure before writing any formulas. Do not assume cell positions.
- Use openpyxl for all operations.
- All values in the yellow cells must be Excel formulas, not hardcoded numbers.
- Preserve all existing formatting (do not clear styles, fonts, fills, borders, etc.).
- The row-to-statistic mapping for rows 42-47 must match the actual labels in the workbook. Read them first.
- For the lookup formulas, verify the exact data range on the Data sheet by inspecting it.

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