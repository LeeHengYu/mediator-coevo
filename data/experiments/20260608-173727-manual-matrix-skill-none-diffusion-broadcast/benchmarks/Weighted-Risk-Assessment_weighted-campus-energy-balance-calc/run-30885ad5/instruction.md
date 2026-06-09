# Task Instruction

Execute the following steps to produce `/root/output/result.xlsx`:

1. **Inspect the workbook** – Open `/root/data/workbook.xlsx` with `openpyxl` (data_only=False). On the `Task` sheet, print:
   - Row 10 (headers/years) columns A–L
   - Rows 12–17 columns A–L (Generation block; note series codes in column D)
   - Rows 19–24 columns A–L (Consumption block)
   - Rows 26–31 columns A–L (Demand block)
   - Rows 35–50 columns A–L (Net balance, stats, weighted mean)
   On the `Data` sheet, print:
   - Row 20 columns A–Z (year headers)
   - Column A rows 21–38 (series codes)
   - A few sample data cells to confirm layout

2. **Step 1 – Lookup formulas (H12:L17, H19:L24, H26:L31)**
   For every cell in those three blocks, write an INDEX/MATCH formula string:
   ```
   =INDEX(Data!$B$21:$Z$38, MATCH($D{row}, Data!$A$21:$A$38, 0), MATCH({col}$10, Data!$B$20:$Z$20, 0))
   ```
   where `{row}` is the current row number and `{col}` is the column letter (H–L). Use `$D{row}` (mixed reference: column absolute, row relative) and `{col}$10` (column relative, row absolute).

3. **Step 2a – Net renewable balance (H35:L40)**
   For each of the six campus rows (rows 35–40 corresponding to rows 12–17 / 19–24 / 26–31), write:
   ```
   =({gen_cell} - {con_cell}) / {dem_cell} * 100
   ```
   where `gen_cell` is the Generation value (e.g., H12), `con_cell` is Consumption (e.g., H19), and `dem_cell` is Demand (e.g., H26), for the matching campus row and year column.

4. **Step 2b – Column-wise statistics (H42:L47)**
   For each column (H–L):
   - Row 42: `=MIN({col}35:{col}40)`
   - Row 43: `=MAX({col}35:{col}40)`
   - Row 44: `=MEDIAN({col}35:{col}40)`
   - Row 45: `=AVERAGE({col}35:{col}40)`
   - Row 46: `=PERCENTILE.INC({col}35:{col}40,0.25)`
   - Row 47: `=PERCENTILE.INC({col}35:{col}40,0.75)`

   **IMPORTANT**: Use `PERCENTILE.INC` (not `PERCENTILE`). The cross-task feedback shows that `#NAME?` errors occurred in percentile rows on a sibling task. Verify after writing that the function name is exactly `PERCENTILE.INC` with no typos.

5. **Step 3 – Weighted mean (H50:L50)**
   For each column (H–L):
   ```
   =SUMPRODUCT({col}35:{col}40, {col}26:{col}31) / SUM({col}26:{col}31)
   ```

6. **Save** – Create `/root/output/` if needed. Save the workbook to `/root/output/result.xlsx`. Do NOT change any existing formatting, do NOT add sheets, macros, VBA, external links, or helper tabs.

7. **Validate** – Re-open the saved file with openpyxl (data_only=False) and print the formula content of:
   - A sample lookup cell (e.g., H12)
   - A sample net balance cell (e.g., H35)
   - All stat cells H42:H47
   - A weighted mean cell (e.g., H50)
   Confirm formulas look correct and contain no typos. Also confirm the file has exactly the original sheets (`Task` and `Data`).

Key reminders:
- Use `openpyxl` to load and save. Write formula strings (not computed values).
- Preserve all existing cell formatting by not touching cells outside the specified ranges.
- The previous successful run used exactly the INDEX/MATCH pattern above and it scored 1.0. Replicate that approach.
- Watch for the PERCENTILE.INC pitfall from the cross-task feedback.

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