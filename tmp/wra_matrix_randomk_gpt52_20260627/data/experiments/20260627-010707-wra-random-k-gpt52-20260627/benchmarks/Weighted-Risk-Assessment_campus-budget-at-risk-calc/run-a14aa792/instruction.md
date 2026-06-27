# Task Instruction

Execute the following steps exactly:

1. **Inspect the workbook** 
 ```python
 import openpyxl
 wb = openpyxl.load_workbook('/root/data/workbook.xlsx')
 ```
 - Print sheet names to confirm `Task` and `Data` exist.
 - On sheet `Data`: print rows 20-38, focusing on column A (series codes) and row 20 (years). Identify the exact layout: which column holds series codes, which row holds year headers, and where the numeric data starts.
 - On sheet `Task`: print rows 10-50 for columns D and H-L. Identify:
   - Row 10: year headers in H10:L10
   - Column D rows 12-17, 19-24, 26-31: series codes
   - The three lookup blocks (H12:L17, H19:L24, H26:L31)
   - The formula block H35:L40 and stats block H42:L47
   - The weighted mean row H50:L50
   - Also print rows 12-17 columns B-G, rows 19-24 columns B-G, rows 26-31 columns B-G to understand block labels (Committed Funding, Operating Spend, Approved Budget Base).

2. **Determine exact Data sheet offsets** 
 From the inspection, note:
 - The column number (1-indexed) of the series-code column on `Data` (likely column A = 1).
 - The row number of the year header row on `Data` (likely row 20).
 - The first and last data columns on `Data`.
 - The row range of data (rows 21:38).

3. **Fill lookup formulas in H12:L17, H19:L24, H26:L31** 
 Use `INDEX/MATCH` with the `Data` sheet. For each cell at (row, col) in these blocks, write a formula like:
 ```
 =INDEX(Data!$B$21:$<lastcol>$38,MATCH($D<row>,Data!$A$21:$A$38,0),MATCH(<colref>$10,Data!$B$20:$<lastcol>$20,0))
 ```
 Adjust the column letters and row numbers based on what you found in step 1-2. Use `$D<row>` (mixed reference: column absolute, row relative within the block) and `<col>$10` (row absolute, column relative) so the formula can be filled across the block correctly.

   Write each cell individually with the correct references. Do NOT use fill; write each formula explicitly or use a loop.

4. **Fill Net budget buffer formulas in H35:L40** 
 For each cell (r, c) in H35:L40, the formula should reference the corresponding cells from the three lookup blocks:
 - Committed Funding block: rows 12-17 → row offset = r - 35 + 12
 - Operating Spend block: rows 19-24 → row offset = r - 35 + 19  
 - Approved Budget Base block: rows 26-31 → row offset = r - 35 + 26

   Formula pattern for cell H35:
   ```
   =(H12-H19)/H26*100
   ```
   Generalize for each cell in the 6×5 block.

5. **Fill statistics in H42:L47** 
 For each column (H through L):
 - Row 42: `=MIN(H35:H40)` (adjust column letter)
 - Row 43: `=MAX(H35:H40)`
 - Row 44: `=MEDIAN(H35:H40)`
 - Row 45: `=AVERAGE(H35:H40)`
 - Row 46: `=PERCENTILE(H35:H40,0.25)`
 - Row 47: `=PERCENTILE(H35:H40,0.75)`

   **Important**: Use `PERCENTILE` exactly (not `PERCENTILE.INC` or `PERCENTILE.EXC`) — the verifier recognizes this form.

   Check the Task sheet to see if the order of min/max/median/mean/25th/75th matches the row labels. If the labels differ, adjust the row assignments accordingly. Print the labels in column B or C for rows 42-47 to confirm.

6. **Fill weighted mean in H50:L50** 
 For each column (H through L):
 ```
 =SUMPRODUCT(H35:H40,H26:H31)/SUM(H26:H31)
 ```
   This computes the weighted mean of the Net budget buffer percentages using Approved Budget Base as weights.

7. **Save the workbook** 
 ```python
 import os
 os.makedirs('/root/output', exist_ok=True)
 wb.save('/root/output/result.xlsx')
 ```

8. **Verify** 
 - Reload the saved workbook with `data_only=False` and spot-check that formulas are present (not None) in representative cells: H12, L17, H19, L24, H26, L31, H35, L40, H42, H47, H50, L50.
 - Confirm no extra sheets were added.
 - Run any available test script: `cd /root && python -m pytest tests/ -v` or similar. Report the output.

**Critical reminders:**
- Do NOT add sheets, macros, VBA, or external links.
- Do NOT change existing formatting.
- Inspect actual cell contents before writing; re-read after writing to confirm.
- If the Data sheet layout differs from assumptions, adapt all formulas accordingly before writing.

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