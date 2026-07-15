# Task Instruction

## Task: Populate formulas in /root/data/workbook.xlsx and save to /root/output/result.xlsx

### Phase 0 — Inspect the workbook thoroughly

1. Create the output directory: `mkdir -p /root/output`
2. Open `/root/data/workbook.xlsx` with openpyxl and inspect **both** sheets:
   - **Sheet `Task`:**
     - Print the exact contents of row 10, columns A–L (these are the year headers for H–L).
     - Print column D for rows 12–31 (these are the series codes used for lookup).
     - Print the labels in column A or B for rows 12–31 to understand the three blocks.
     - Print rows 35–50, columns A–L to understand the layout of the derived sections.
     - Note the exact data types of the year values in row 10 (int, float, string?).
   - **Sheet `Data`:**
     - Print rows 21–38 fully (all columns with data) to see the lookup table structure.
     - Identify: Which column contains the series codes? Which row contains the year headers? What are the exact data types of the year headers (int, float, string?)?
     - Print the first row of the Data sheet (row 1) and any header rows to understand the column layout.
3. **Critical check:** Compare the data types of years in `Task` row 10 vs the year headers in the `Data` sheet. If one is numeric and the other is string, the MATCH/lookup will fail with #N/A. Document any mismatch.

### Phase 1 — Write lookup formulas in H12:L17, H19:L24, H26:L31

Based on the inspection, write formulas using `INDEX`/`MATCH` (recommended for reliability). For each cell in the yellow ranges:

- The formula should look up the series code from column D of the **current row** and the year from row 10 of the **current column** in the Data sheet rows 21–38.
- Use the pattern: `=INDEX(Data!<value_range>, MATCH(<series_code>, Data!<series_code_column>, 0), MATCH(<year>, Data!<year_header_row>, 0))`
- **Data type alignment is critical:** If the years in the Data sheet header row are stored as numbers but Task row 10 has them as text (or vice versa), you must either:
  - Fix the Task row 10 values to match the Data sheet type, OR
  - Wrap the MATCH year argument with `VALUE()` or `TEXT()` or multiply by 1 to coerce types.
  - Similarly check if series codes have leading/trailing spaces or case differences.
- The value range and series code column must correspond to the actual Data sheet layout discovered in Phase 0.
- Apply the formula to all 6 rows × 5 columns in each of the three blocks (H12:L17, H19:L24, H26:L31).

### Phase 2 — Net reliability gap (H35:L40)

For each of the 6 regions (rows 35–40) and 5 year columns (H–L):
- Formula: `=(H12 - H19) / H26 * 100` (adjusted for actual row references)
  - H12 block = Successful API Requests
  - H19 block = Failed API Requests  
  - H26 block = Compute Capacity
- Verify the row mapping: row 35 corresponds to the first region (row 12, 19, 26), row 36 to second (row 13, 20, 27), etc.

### Phase 3 — Summary statistics (H42:L47)

For each year column H–L, calculate column-wise stats over H35:L40:
- Row 42: `=MIN(H35:H40)`
- Row 43: `=MAX(H35:H40)`
- Row 44: `=MEDIAN(H35:H40)`
- Row 45: `=AVERAGE(H35:H40)` (simple mean)
- Row 46: `=PERCENTILE(H35:H40, 0.25)` (25th percentile)
- Row 47: `=PERCENTILE(H35:H40, 0.75)` (75th percentile)

**Important:** Check the labels in column A/B for rows 42–47 to confirm the correct order (min, max, median, mean, 25th, 75th). Adjust row assignments if the labels indicate a different order.

### Phase 4 — Weighted mean (H50:L50)

For each year column H–L:
- `=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)`
- This uses Net reliability gap values as the values and Compute Capacity as weights.

### Phase 5 — Save and validate

1. Save the workbook to `/root/output/result.xlsx`.
2. Reopen the saved file with openpyxl (data_only=False) and verify:
   - Cells H12, L17, H19, L24, H26, L31 contain formula strings (not None).
   - Cells H35, L40 contain formula strings.
   - Cells H42, L47 contain formula strings.
   - Cell H50 contains a formula string.
3. Also open with data_only=True (or use xlcalc/formulas library if available) to spot-check that formulas don't obviously resolve to errors. If openpyxl data_only shows None (cached value not available), that's expected and acceptable.
4. Do NOT add any new sheets, macros, VBA, external links, or helper tabs.
5. Do NOT alter any existing formatting.

### Key warnings from cross-task feedback
- **#N/A errors** were the primary failure mode in a similar task. Root cause was data type mismatch between year values in the Task sheet row 10 and the Data sheet header row. Inspect and handle this explicitly.
- Always verify the exact cell coordinates and layout before writing formulas. Do not assume the layout matches the description without checking.

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