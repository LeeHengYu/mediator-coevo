# Task Instruction

## Task: Update /root/data/workbook.xlsx with formulas and save to /root/output/result.xlsx

### Phase 0: Inspect the workbook structure
1. Create `/root/output/` directory if it doesn't exist.
2. Use `openpyxl` to open `/root/data/workbook.xlsx` and inspect:
   - Sheet names (confirm `Task` and `Data` exist).
   - On sheet `Task`: read row 10 (the year headers in columns H–L), read column D rows 12–17, 19–24, 26–31 (the series codes), read the labels in rows 35–40 (region names), row 42–47 (stat labels), row 50 (GCM label). Print all of these so you understand the layout.
   - On sheet `Data`: read rows 21–38 to understand the data table structure — identify which row holds headers, which column holds series codes, how years map to columns. Print the first few cells of each row.
3. Based on inspection, determine:
   - The exact column letter/number in `Data` that contains the series codes (for VLOOKUP's table_array or INDEX/MATCH).
   - The exact row in `Data` that contains the year headers (for MATCH on years).
   - The range of the data table on `Data` sheet (e.g., `Data!A21:XX38` or similar).

### Phase 1: Populate lookup formulas in H12:L17, H19:L24, H26:L31

For each cell in these three blocks, write a spreadsheet formula (not a Python-computed value). Use the `INDEX(MATCH, MATCH)` pattern or `VLOOKUP` with `MATCH`. The formula must:
- Reference the series code from column D of the **same row** on the `Task` sheet.
- Reference the year from row 10 of the **same column** on the `Task` sheet.
- Look up the value from the `Data` sheet rows 21–38.

Concrete approach (adapt column/row references based on Phase 0 inspection):
- If `Data` has series codes in column A and year headers in some row (say row 21), use something like:
  `=INDEX(Data!$B$22:$ZZ$38, MATCH($D12, Data!$A$22:$A$38, 0), MATCH(H$10, Data!$B$21:$ZZ$21, 0))`
- Adjust the ranges based on actual inspection. Lock references appropriately with `$` so the formula can be applied across the H–L columns and down the rows correctly.
- Use `openpyxl` to write these formula strings into cells H12:L17, H19:L24, H26:L31.

**IMPORTANT**: When writing formulas with openpyxl, the formula string must start with `=`. Use English function names. Do NOT use `data_only` mode — open normally so formulas are preserved.

### Phase 2: Net reliability gap in H35:L40 and statistics in H42:L47

For H35:L40 (6 regions × 5 years):
- The formula is: `(Successful API Requests - Failed API Requests) / Compute Capacity * 100`
- The three blocks from Step 1 correspond to three different indicators. Determine from the series codes or labels which block is "Successful API Requests" (likely H12:L17), which is "Failed API Requests" (likely H19:L24), and which is "Compute Capacity" (likely H26:L31). Verify by reading the labels/series codes.
- For cell H35: `=(H12-H19)/H26*100` (adjust row references based on which block is which — the region in row 35 should correspond to row 12, row 19, row 26 for the same region). Confirm the region ordering is the same across all blocks.
- Apply this pattern for all 30 cells in H35:L40.

For H42:L47 (column-wise statistics over H35:L40):
- Row 42 (minimum): `=MIN(H35:H40)` for each column H–L
- Row 43 (maximum): `=MAX(H35:H40)` for each column H–L  
- Row 44 (median): `=MEDIAN(H35:H40)` for each column H–L
- Row 45 (simple mean): `=AVERAGE(H35:H40)` for each column H–L
- Row 46 (25th percentile): `=PERCENTILE(H35:H40,0.25)` for each column H–L
- Row 47 (75th percentile): `=PERCENTILE(H35:H40,0.75)` for each column H–L
- **Check the labels in column D/G for rows 42–47 to confirm the correct order of min/max/median/mean/25th/75th. Assign formulas matching the actual label in each row.**

### Phase 3: Weighted mean in H50:L50
- For each column (H–L): `=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)`
- This uses the Net reliability gap percentages as values and Compute Capacity as weights.

### Phase 4: Save and verify
1. Save the workbook to `/root/output/result.xlsx`.
2. Reopen the saved file and verify:
   - Sheets are still only `Task` and `Data` (no extra sheets).
   - Cells H12, L17, H19, L24, H26, L31 contain formula strings (start with `=`).
   - Cells H35, L40 contain formula strings.
   - Cells H42, L47 contain formula strings.
   - Cell H50 and L50 contain formula strings.
   - Print a sample of formulas from each block to confirm correctness.
3. Confirm no macros, VBA, or external links were added.

### Critical Notes
- Do NOT use `data_only=True` when opening — that strips formulas.
- Do NOT add or remove any sheets.
- Do NOT change formatting (fonts, colors, borders, etc.). Only write formulas into the specified cells.
- All formulas must be Excel-compatible spreadsheet formulas, not computed Python values.
- When constructing INDEX/MATCH formulas referencing the Data sheet, use the syntax `Data!` prefix.
- Double-check that the row/column references in your formulas align with the actual workbook layout discovered in Phase 0. Do not assume — inspect first.

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