# Task Instruction

Execute the following steps to produce `/root/output/result.xlsx`:

1. **Inspect the workbook** – Open `/root/data/workbook.xlsx` with openpyxl (data_only=False). Print:
   - Sheet names.
   - `Task` sheet: cells D12:D17 (series codes block 1), D19:D24 (block 2), D26:D31 (block 3), row 10 columns H–L (years), rows 35–40 col D (department labels for Net budget buffer), rows 42–47 col D or E (stat labels), row 50 col D (Campus Budget Council label).
   - `Data` sheet: rows 21–38 — print the first column (series codes) and the header row to understand the data layout (which row holds headers, which column holds series codes, how years are arranged).
   - Note the exact row/column structure so formulas reference correctly.

2. **Populate lookup formulas in H12:L17, H19:L24, H26:L31** using `INDEX-MATCH-MATCH`:
   - Pattern for each cell, e.g. H12:
     ```
     =INDEX(Data!$A$21:$Z$38,MATCH($D12,Data!$A$21:$A$38,0),MATCH(H$10,Data!$A$21:$Z$21,0))
     ```
     Adjust the range boundaries after inspecting the Data sheet. Use the actual extent of the data (e.g., if data goes from column A to column F on the Data sheet, use that range). The key references:
     - `$D12` (absolute column, relative row) = series code from column D of the current row.
     - `H$10` (relative column, absolute row) = year from row 10.
     - The lookup arrays must cover the full data range on the Data sheet rows 21–38.
   - Fill all 18 cells in each of the three blocks (6 rows × 5 columns each) using the same pattern with appropriate relative/absolute references.

3. **Net budget buffer in H35:L40** – For each cell (e.g., H35):
   ```
   =(H19-H12)/H26*100
   ```
   Where:
   - H19:L24 = Committed Funding (block 2)
   - H12:L17 = Operating Spend (block 1)
   - H26:L31 = Approved Budget Base (block 3)
   
   **Verify this mapping by checking the labels** next to each block on the Task sheet. The formula is `(Committed Funding - Operating Spend) / Approved Budget Base * 100`. Adjust row references if the blocks map differently (e.g., if block 1 is Committed Funding instead of Operating Spend). Print the labels in column C or D for rows 11, 18, 25 to confirm which block is which.

4. **Statistics in H42:L47** – Column-wise stats over H35:L40. Use these formulas (example for column H):
   - H42 (minimum): `=MIN(H35:H40)`
   - H43 (maximum): `=MAX(H35:H40)`
   - H44 (median): `=MEDIAN(H35:H40)`
   - H45 (mean): `=AVERAGE(H35:H40)`
   - H46 (25th percentile): `=PERCENTILE.INC(H35:H40,0.25)`
   - H47 (75th percentile): `=PERCENTILE.INC(H35:H40,0.75)`
   
   **CRITICAL**: Verify the stat labels in column D/E for rows 42–47 to determine the correct order. Map each row to the correct function based on the label. Check that the label row ordering matches (it might be min, max, median, mean, 25th, 75th or a different order).
   
   **IMPORTANT about PERCENTILE.INC**: The cross-task context warns that `PERCENTILE.INC` caused `#NAME?` errors in a similar task. After writing the formulas, open the saved file with openpyxl and verify the formulas are stored correctly. If the environment's Excel engine doesn't support `PERCENTILE.INC`, try `PERCENTILE` instead. However, openpyxl just stores formula strings — `#NAME?` errors would only appear at evaluation time in Excel. Since the previous successful run of THIS task used the same approach and scored 1.0, proceed with `PERCENTILE.INC` but double-check the exact function name has no typos.

5. **Weighted mean in H50:L50** – For each column (e.g., H50):
   ```
   =SUMPRODUCT(H35:H40,H26:H31)/SUM(H26:H31)
   ```
   This computes the weighted mean of the Net budget buffer percentages weighted by Approved Budget Base.

6. **Save** to `/root/output/result.xlsx` — create the output directory if needed. Use `openpyxl` to save, preserving existing formatting.

7. **Validate** — Reopen the saved file and print a sample of formulas from each block (e.g., H12, L17, H35, L40, H42, H47, H50, L50) to confirm they are correctly written.

Do NOT add sheets, macros, VBA, external links, or helper tabs. Do NOT alter existing formatting.

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