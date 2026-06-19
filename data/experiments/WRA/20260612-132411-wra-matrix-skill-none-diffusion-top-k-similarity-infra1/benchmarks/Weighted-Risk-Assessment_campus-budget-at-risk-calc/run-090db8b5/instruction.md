# Task Instruction

Execute the following steps to produce `/root/output/result.xlsx`:

1. **Inspect the workbook** – Open `/root/data/workbook.xlsx` with openpyxl (data_only=False). Print:
   - Sheet names.
   - Sheet `Task`: cells D12:D17, D19:D24, D26:D31 (series codes), row 10 columns H–L (years), cells H35:H40 row labels or D35:D40, and any existing content/formatting in the target ranges.
   - Sheet `Data`: rows 21–38 to understand the lookup table layout (which row holds what, which column holds series codes, where year headers are).
   This inspection is critical — do NOT skip it.

2. **Determine lookup geometry** – From the Data sheet inspection, identify:
   - Whether series codes are in a column (for VLOOKUP/INDEX) or row (for HLOOKUP).
   - Where year headers sit.
   - The exact range boundaries for the MATCH and INDEX references.

3. **Write formulas in Python using openpyxl** – For each yellow cell in H12:L17, H19:L24, H26:L31 on sheet `Task`, write an Excel formula using the `INDEX(…, MATCH(…), MATCH(…))` pattern. Use:
   - One MATCH on the series code (column D of the current row) against the series-code column/row in Data!rows 21:38.
   - One MATCH on the year (row 10 of the current column) against the year row/column in Data!rows 21:38.
   - Anchor references with `$` signs appropriately so the formula can be written per-cell correctly.
   Example pattern: `=INDEX(Data!$B$21:$Z$38, MATCH($D12, Data!$A$21:$A$38, 0), MATCH(H$10, Data!$B$20:$Z$20, 0))` — adjust ranges based on actual inspection.

4. **Step 2 – Net budget buffer (H35:L40):**
   Write formulas computing `(Committed Funding - Operating Spend) / Approved Budget Base * 100`.
   - Committed Funding = H12:L17 block.
   - Operating Spend = H19:L24 block.
   - Approved Budget Base = H26:L31 block.
   - So H35 = `=(H12-H19)/H26*100`, etc. Confirm row alignment (department 1 in row 12, 19, 26, 35 etc.).

5. **Step 2 – Summary statistics (H42:L47):**
   For each column (H through L):
   - Row 42: `=MIN(H35:H40)`
   - Row 43: `=MAX(H35:H40)`
   - Row 44: `=MEDIAN(H35:H40)`
   - Row 45: `=AVERAGE(H35:H40)`
   - Row 46: `=PERCENTILE(H35:H40,0.25)` — use `PERCENTILE` not `PERCENTILE.INC` or `PERCENTILE.EXC` (openpyxl and Excel both support plain PERCENTILE; the failed artifact from a sibling task got #NAME? errors likely from using an unrecognized function name).
   - Row 47: `=PERCENTILE(H35:H40,0.75)`
   **IMPORTANT**: Verify by inspecting any existing labels in column D/G rows 42-47 to confirm which row is which statistic. Adjust mapping accordingly.

6. **Step 3 – Weighted mean (H50:L50):**
   For each column col in H–L:
   `=SUMPRODUCT(H35:H40,H26:H31)/SUM(H26:H31)` (adjusting column letter).

7. **Save** – Create `/root/output/` directory if needed. Save the workbook to `/root/output/result.xlsx`. Do NOT change formatting, do NOT add sheets.

8. **Validate** – Reopen the saved file with openpyxl (data_only=False) and print formulas in a sample of cells (e.g., H12, L17, H35, L40, H42, H47, H50, L50) to confirm they are well-formed Excel formulas. Also confirm no new sheets were added and the original sheets are intact.

**Key cautions:**
- Use `PERCENTILE` (not `PERCENTILE.INC` or `PERCENTILE.EXC`) to avoid #NAME? errors.
- Do not use `SUMPRODUCT` with division inside — use `SUMPRODUCT(values,weights)/SUM(weights)` form.
- Inspect before writing. The exact row/column layout of the Data sheet determines all formula references.
- Preserve all existing formatting and content outside the target cells.

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