# Task Instruction

Execute the following steps to produce `/root/output/result.xlsx`:

1. **Inspect the workbook** – Open `/root/data/workbook.xlsx` with openpyxl (data_only=False). Print:
   - Sheet names.
   - On sheet `Task`: cells D12:D17, D19:D24, D26:D31 (series codes), row 10 columns H–L (years), cells H35:H40 labels or D35:D40 labels, cells in column D or G for rows 42–47 (stat labels), and row 50 label.
   - On sheet `Data`: rows 21–38, focusing on the header row structure (which row holds series codes, which row/column holds years) so you understand the 2D layout.
   Print enough to determine the exact data range boundaries (first row, last row, first col, last col) of the Data block.

2. **Step 1 – Populate H12:L17, H19:L24, H26:L31 with lookup formulas**
   For each cell in those ranges, write an `INDEX/MATCH` formula that:
   - Looks up the series code from column D of that row (use mixed ref `$D12` so the column is locked).
   - Looks up the year from row 10 of that column (use mixed ref `H$10` so the row is locked).
   - References the Data sheet block you identified (lock with `$` signs).
   - Pattern: `=INDEX(Data!$<data_range>,MATCH($D12,Data!$<series_code_column>,0),MATCH(H$10,Data!$<year_row>,0))`
   Adjust the exact ranges based on what you found in step 1.

3. **Step 2a – Net renewable balance in H35:L40**
   The formula for each campus (rows 35–40, columns H–L) is:
   `=(H12 - H19) / H26 * 100`  (adjusted per row offset)
   Specifically, row 35 uses the data from row 12 (Renewable Gen), row 19 (Grid Consumption), row 26 (Baseline Energy Demand), etc. Map each campus row in 35–40 to the corresponding rows in the three blocks (12–17, 19–24, 26–31). Use the same column.

4. **Step 2b – Summary statistics in H42:L47**
   Read the labels in column D (or wherever they are) for rows 42–47 to determine which statistic each row expects. Then for each column H–L:
   - Minimum → `=MIN(H35:H40)`
   - Maximum → `=MAX(H35:H40)`
   - Median → `=MEDIAN(H35:H40)`
   - Mean → `=AVERAGE(H35:H40)`
   - 25th percentile → `=PERCENTILE(H35:H40,0.25)`
   - 75th percentile → `=PERCENTILE(H35:H40,0.75)`
   Match each label to the correct function.

5. **Step 3 – Weighted mean in H50:L50**
   For each column H–L:
   `=SUMPRODUCT(H35:H40,H26:H31)/SUM(H26:H31)`
   This uses the net renewable balance percentages as values and Baseline Energy Demand as weights.

6. **Save** – Create `/root/output/` if needed. Save the workbook to `/root/output/result.xlsx`. Do NOT change formatting, add sheets, macros, VBA, or external links.

7. **Verify** – Reopen the saved file with openpyxl, print a sample of the formula cells (e.g., H12, H19, H26, H35, H42, H50) to confirm they contain the expected formula strings (not values). Confirm no extra sheets were added.

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