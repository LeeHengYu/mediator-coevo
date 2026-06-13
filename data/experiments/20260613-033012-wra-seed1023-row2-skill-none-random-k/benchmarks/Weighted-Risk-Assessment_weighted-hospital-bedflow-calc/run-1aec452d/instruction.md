# Task Instruction

Implement a Python script that:

1. **Inspect the workbook first.** Open `/root/data/workbook.xlsx` with `openpyxl` (with `data_only=False` so you can see any existing formulas). Print out:
   - The sheet names.
   - The contents of `Task` sheet cells D12:D17, D19:D24, D26:D31 (series codes).
   - The contents of `Task` sheet row 10, columns H through L (years).
   - The contents of `Data` sheet rows 21:38 — print the first few columns to understand the layout (especially column A or B for series codes, and the header row for years).
   - The contents of `Task` cells H35:H40 area labels, H42:H47 area labels, H50 area label.
   This inspection is critical to understand the exact structure before writing formulas.

2. **After inspecting**, write formulas into the workbook. Use `openpyxl` to write formula strings (starting with `=`). Here is the approach:

   **Step 1 — Lookup formulas in H12:L17, H19:L24, H26:L31:**
   For each cell in these ranges, write an `INDEX(MATCH,MATCH)` formula that:
   - Looks up the series code from column D of the same row on sheet `Task`.
   - Looks up the year from row 10 of the same column on sheet `Task`.
   - Searches within the `Data` sheet rows 21:38.
   
   Based on inspection, determine the exact data range on `Data` sheet. The formula pattern should be something like:
   `=INDEX(Data!$B$21:$Z$38, MATCH($D12, Data!$A$21:$A$38, 0), MATCH(H$10, Data!$B$20:$Z$20, 0))`
   Adjust the column/row references based on what you see in the actual data layout. The key is:
   - The MATCH for the series code searches the series code column in Data rows 21:38.
   - The MATCH for the year searches the year header row in the Data sheet.
   - INDEX returns the intersection.

   **Step 2 — Net patient flow in H35:L40:**
   The formula for net patient flow is: `(Admissions - Discharges) / Effective Bed Capacity * 100`
   - Admissions are in H12:L17 (rows 12-17).
   - Discharges are in H19:L24 (rows 19-24).
   - Effective Bed Capacity is in H26:L31 (rows 26-31).
   So for cell H35: `=(H12-H19)/H26*100`, for H36: `=(H13-H20)/H27*100`, etc.
   Map row 35→12,19,26; row 36→13,20,27; row 37→14,21,28; row 38→15,22,29; row 39→16,23,30; row 40→17,24,31.

   **Step 2 — Statistics in H42:L47:**
   For each column (H through L):
   - H42 (min): `=MIN(H35:H40)`
   - H43 (max): `=MAX(H35:H40)`
   - H44 (median): `=MEDIAN(H35:H40)`
   - H45 (mean): `=AVERAGE(H35:H40)`
   - H46 (25th percentile): `=PERCENTILE(H35:H40,0.25)` — use `PERCENTILE` not `PERCENTILE.INC` to avoid #NAME? errors in openpyxl/verifier contexts. Actually, check: Excel supports both. Since the avoid-artifact warns about #NAME? errors from unsupported functions, prefer `PERCENTILE` (the legacy function name).
   - H47 (75th percentile): `=PERCENTILE(H35:H40,0.75)`

   **Step 3 — Weighted mean in H50:L50:**
   `=SUMPRODUCT(H35:H40,H26:H31)/SUM(H26:H31)` for each column H through L.

3. **Verify the order of statistics rows (42-47).** Before writing, inspect what labels exist in column D or G for rows 42-47 to confirm which row is min, max, median, mean, 25th pctl, 75th pctl. Adjust accordingly.

4. **Save** the workbook to `/root/output/result.xlsx`. Create the `/root/output/` directory if it doesn't exist.

5. **Post-save verification:** Reopen the saved file and print the formula content of a few cells (e.g., H12, H35, H42, H46, H50) to confirm formulas were written correctly.

**Critical details:**
- All formulas must be strings starting with `=`.
- Do NOT use `data_only=True` when writing — use default mode.
- Do NOT add new sheets, macros, or VBA.
- Do NOT change formatting.
- Use `PERCENTILE` (not `PERCENTILE.INC` or `PERCENTILE.EXC`) to avoid #NAME? errors.
- Use `AVERAGE` (not `MEAN`) for the simple mean.
- Ensure the Data sheet reference uses the exact sheet name as it appears in the workbook (case-sensitive in formulas).
- Run the inspection step FIRST before writing any formulas, and adapt all cell references based on what you actually observe in the data layout.

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