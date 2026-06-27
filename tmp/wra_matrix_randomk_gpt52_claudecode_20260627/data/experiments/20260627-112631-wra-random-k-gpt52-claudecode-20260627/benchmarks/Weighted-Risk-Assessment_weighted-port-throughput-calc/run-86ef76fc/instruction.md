# Task Instruction

Execute the following steps to produce /root/output/result.xlsx from /root/data/workbook.xlsx.

## 0 – Inspect the workbook
1. `mkdir -p /root/output`
2. Write and run a Python script that opens `/root/data/workbook.xlsx` with openpyxl (data_only=False) and prints:
   - Sheet names.
   - From sheet `Task`: cells D12:D17, D19:D24, D26:D31 (series codes), row 10 columns H–L (years), cells H35:L40 labels/content, H42:L47 labels, H50:L50 label, and any existing content/formatting clues.
   - From sheet `Data`: row 21 through 38, columns A–Z (or however wide the data extends), so we can see the lookup table structure (headers, series codes, years, values).
   Print everything so we understand the exact layout before writing formulas.

## 1 – Populate lookup formulas (H12:L31)
Write a Python script using openpyxl that:
- Opens the workbook preserving styles/formatting.
- For each of the three blocks (rows 12–17, 19–24, 26–31), for each cell in columns H–L:
  - Reads the series code from column D of that row (e.g., `$D12`).
  - Reads the year from row 10 of that column (e.g., `H$10`).
  - Writes an `INDEX`/`MATCH` formula referencing sheet `Data` rows 21:38. Use the pattern:
    `=INDEX(Data!$B$21:$Z$38, MATCH($D12,Data!$A$21:$A$38,0), MATCH(H$10,Data!$B$20:$Z$20,0))`
    Adjust the exact column/row ranges after inspecting the Data sheet layout in step 0. The key contract: two MATCH calls (one for the series code in the leftmost column of the data range, one for the year in the header row above the data range), wrapped in INDEX over the value area.
  - Use absolute row references for the lookup ranges and mixed references ($D12 for series, H$10 for year) so formulas copy correctly.

## 2 – Net container flow (H35:L40)
For each cell in H35:L40, write a formula:
`=(H12-H19)/H26*100`
where row 12 = Loaded Containers Inbound, row 19 = Loaded Containers Outbound, row 26 = Terminal Throughput Capacity, adjusted per port (rows 12–17 map to 35–40, rows 19–24 map to 35–40, rows 26–31 map to 35–40). Specifically for row 35 col H: `=(H12-H19)/H26*100`, row 36: `=(H13-H20)/H27*100`, etc.

## 3 – Statistics (H42:L47)
For each column H–L, write these formulas in the six statistic rows:
- MIN:          `=MIN(H35:H40)`
- MAX:          `=MAX(H35:H40)`
- MEDIAN:       `=MEDIAN(H35:H40)`
- MEAN:         `=AVERAGE(H35:H40)`
- 25th pctile:  `=PERCENTILE(H35:H40,0.25)`
- 75th pctile:  `=PERCENTILE(H35:H40,0.75)`

**Important:** Use `PERCENTILE` (not `PERCENTILE.INC` or `PERCENTILE.EXC`) to avoid #NAME? errors in the verifier environment.

## 4 – Weighted mean (H50:L50)
For each column H–L, write:
`=SUMPRODUCT(H35:H40,H26:H31)/SUM(H26:H31)`
This computes the CPA weighted mean using Net container flow percentages as values and Terminal Throughput Capacity as weights.

## 5 – Save
Save to `/root/output/result.xlsx`. Do NOT create new sheets, macros, VBA, or external links.

## 6 – Validate
Re-open the saved file with openpyxl (data_only=False) and print all formula cells in the ranges above to confirm they are present and correctly structured. Check that no cell is None or empty where a formula is expected. Also verify sheet names are unchanged and no extra sheets exist.

## Key cautions
- After step 0 inspection, adjust all range references to match the actual Data sheet layout (header row for years, column for series codes, value area boundaries).
- Preserve all existing formatting; only write to the specified yellow cells.
- Use base function names (PERCENTILE, not PERCENTILE.INC) to avoid #NAME? errors.
- Double-check mixed vs absolute references in formulas.

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