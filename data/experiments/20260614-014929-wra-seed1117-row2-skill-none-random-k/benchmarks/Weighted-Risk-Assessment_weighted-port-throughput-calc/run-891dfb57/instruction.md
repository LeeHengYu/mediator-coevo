# Task Instruction

Execute the following steps carefully to produce `/root/output/result.xlsx`.

## Phase 0 — Inspect the workbook

1. `mkdir -p /root/output`
2. Write and run a Python script that uses `openpyxl` (load with `data_only=False`) to inspect:
   - **Task sheet**: Print cells D12:D17, D19:D24, D26:D31 (series codes), row 10 columns H–L (year headers), cells H35:L40 labels if any, H42:L47 labels, H50:L50 labels, and the exact text in B12:B17, B19:B24, B26:B31 (block labels). Also print the exact content of every cell in rows 10–50, columns A–L, so you can see the full layout.
   - **Data sheet**: Print rows 1–5 (headers) and rows 18–40 completely (all columns A through at least Z). Pay special attention to: which column holds the series codes, which row holds the year headers, and the exact year values (type and content). Print `repr()` of every cell value so you can see types and whitespace.
3. From this inspection, determine:
   a. The lookup key column on Data (e.g., column A or B).
   b. The year header row on Data.
   c. Whether years are integers, floats, or strings on both sheets.
   d. The exact series codes on both sheets (check for whitespace mismatches).
   e. The layout: are series codes in rows (VLOOKUP) or columns (HLOOKUP)?

## Phase 1 — Write the formulas

Based on the inspection, write a Python script using `openpyxl` that:

1. Opens `/root/data/workbook.xlsx` (with `data_only=False`).
2. **Step 1 — Lookup formulas in H12:L17, H19:L24, H26:L31**:
   - For each cell in these ranges, write an Excel formula that combines the series code from column D of the same row with the year from row 10 of the same column.
   - Choose the correct lookup pattern based on the Data sheet layout:
     - If Data has series codes in a column and years across a row header: use `INDEX(Data!<data_range>, MATCH(Task!$D<row>, Data!<code_column>, 0), MATCH(Task!<col>$10, Data!<year_row>, 0))` or equivalent VLOOKUP/MATCH.
     - Make sure range references are correct and anchored properly (use `$` for fixed references).
   - **Critical**: Ensure the MATCH for years compares compatible types. If years on one sheet are numbers and on the other are text, wrap with `VALUE()` or `TEXT()` as needed. If both are numbers, use exact match (0).
   - **Critical**: Ensure the MATCH for series codes uses exact match (0) and the lookup range exactly covers the code column in Data rows 21:38.

3. **Step 2 — Net container flow in H35:L40**:
   - Formula: `(Loaded Containers Inbound - Loaded Containers Outbound) / Terminal Throughput Capacity * 100`
   - Map the three blocks: identify which block (H12:L17, H19:L24, H26:L31) corresponds to which metric by reading the labels in column B or C near rows 12, 19, 26.
   - For each cell in H35:L40, reference the corresponding cells from the three lookup blocks.

4. **Step 2 — Statistics in H42:L47**:
   - Row 42: `=MIN(H35:H40)` (column-wise MIN over the 6 ports) — but wait, it's only 6 rows H35:H40. Adjust if the range is H35:H40.
   - Row 43: `=MAX(H35:H40)`
   - Row 44: `=MEDIAN(H35:H40)`
   - Row 45: `=AVERAGE(H35:H40)`
   - Row 46: `=PERCENTILE(H35:H40, 0.25)` — **Use PERCENTILE or PERCENTILE.INC** (check which Excel supports; `PERCENTILE` is safest for openpyxl compatibility).
   - Row 47: `=PERCENTILE(H35:H40, 0.75)`
   - **Important**: Use `PERCENTILE` not `PERCENTILE.INC` or `PERCENTILE.EXC` to avoid #NAME? errors (learned from failed cloud-reliability task).
   - Actually, check: `PERCENTILE.INC` is valid in modern Excel. But openpyxl might not care — the formula is just a string. The issue is whether the verifier evaluates with a library that supports it. Use `PERCENTILE` to be safe.

5. **Step 3 — Weighted mean in H50:L50**:
   - `=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)` for each column H–L.
   - This computes the weighted mean using net container flow percentages as values and Terminal Throughput Capacity as weights.

6. Save to `/root/output/result.xlsx`. Do NOT change formatting, do NOT add sheets.

## Phase 2 — Validate

1. Re-open `/root/output/result.xlsx` with openpyxl and print all formula cells in the ranges H12:L17, H19:L24, H26:L31, H35:L40, H42:L47, H50:L50 to verify they contain formulas (not values, not None).
2. Also open with `data_only=True` to check if any cells show error values (though openpyxl may show None for formula cells when data_only=True without a cached value — that's OK).
3. If any formula looks wrong based on the Phase 0 inspection, fix it before finalizing.

## Key Pitfalls to Avoid
- Do NOT use `PERCENTILE.INC` or `PERCENTILE.EXC` — use `PERCENTILE` to avoid #NAME? errors.
- Do NOT assume the Data sheet layout — inspect it first.
- Ensure year type compatibility in MATCH functions.
- Ensure series code strings match exactly (no trailing spaces).
- Use absolute references where needed (e.g., `$D12` for the series code column, `H$10` for the year row).
- Do not modify any existing formatting, sheets, or structure.

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