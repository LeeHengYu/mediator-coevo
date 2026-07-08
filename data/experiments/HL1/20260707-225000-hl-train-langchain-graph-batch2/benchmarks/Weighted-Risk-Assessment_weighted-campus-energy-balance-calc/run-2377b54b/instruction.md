# Task Instruction

Execute the following steps precisely to complete the weighted campus energy balance workbook.

## Pre-work: Inspect the workbook

1. Copy the source file:
 ```bash
 cp /root/data/workbook.xlsx /root/output/result.xlsx
 ```
2. Use `openpyxl` (Python) to open `/root/output/result.xlsx` and inspect:
   - Sheet names (confirm `Task` and `Data` exist).
   - On sheet `Task`: read cells D12:D17, D19:D24, D26:D31 to see the series codes; read H10:L10 to see the years; read H35:H40 labels or D35:D40 for campus names; read H42:H47 or labels for the stat rows; read H50 row label.
   - On sheet `Data`: read row 21 (header row) and rows 22–38 to understand the data layout — identify which column holds the series code, which row holds years, and the data range.
   - Print all findings so you understand the exact layout before writing any formulas.

## Step 1: Populate H12:L17, H19:L24, H26:L31 with lookup formulas

For each cell in these three blocks, write a formula that:
- Looks up the series code from column D of that row against the Data sheet rows 21–38.
- Matches the year from row 10 of the current column against the Data sheet header row.
- Uses one of the allowed patterns: `INDEX(MATCH,MATCH)` is recommended for a 2D lookup.

Concretely, assuming Data!A21:A38 contains series codes and Data!B20:?20 (or row 21) contains years, adapt accordingly. A typical formula pattern (adjust references after inspection):

```
=INDEX(Data!$B$22:$??$38, MATCH($D12,Data!$A$22:$A$38,0), MATCH(H$10,Data!$B$21:$??$21,0))
```

Adjust the exact ranges based on what you find during inspection. The key constraints:
- Row anchor: series code from column D of the current row (use $D12 with absolute column).
- Column anchor: year from row 10 of the current column (use H$10 with absolute row).
- The lookup must use one of: VLOOKUP+MATCH, HLOOKUP+MATCH, XLOOKUP+MATCH, or INDEX+MATCH.
- Fill all 6 rows × 5 columns in each of the three blocks (90 cells total).

Use `openpyxl` to write these as formula strings (not computed values). Make sure to use the `data_only=False` mode when writing.

## Step 2: Net renewable balance (H35:L40) and statistics (H42:L47)

For H35:L40, write formulas computing:
```
=(RenewableGeneration - GridConsumption) / BaselineEnergyDemand * 100
```
where:
- Renewable Generation values are in H12:L17
- Grid Consumption values are in H19:L24  
- Baseline Energy Demand values are in H26:L31

So for cell H35: `=(H12-H19)/H26*100`, H36: `=(H13-H20)/H27*100`, etc. Map each campus row correctly (row 35↔row 12,19,26; row 36↔row 13,20,27; etc.).

For H42:L47, write column-wise statistical formulas over H35:L40:
- H42 (Min): `=MIN(H35:H40)`
- H43 (Max): `=MAX(H35:H40)`
- H44 (Median): `=MEDIAN(H35:H40)`
- H45 (Mean): `=AVERAGE(H35:H40)`
- H46 (25th percentile): `=PERCENTILE(H35:H40,0.25)` or `=PERCENTILE.INC(H35:H40,0.25)`
- H47 (75th percentile): `=PERCENTILE(H35:H40,0.75)` or `=PERCENTILE.INC(H35:H40,0.75)`

Repeat for columns I through L. Check the row labels to confirm the order (min, max, median, mean, 25th, 75th) and adjust if needed.

## Step 3: Weighted mean in H50:L50

For each column H through L, write a SUMPRODUCT formula:
```
=SUMPRODUCT(H35:H40,H26:H31)/SUM(H26:H31)
```
This computes the weighted mean of the net renewable balance percentages using baseline energy demand as weights.

## Post-work: Validation

1. Re-open the saved file and verify:
   - All target cells contain formula strings (not None or bare values).
   - Spot-check a few formulas for correctness.
   - No new sheets were added.
   - The file saves without error.
2. Print confirmation of completion.

## Critical constraints
- Use `openpyxl` only (no xlsxwriter for .xlsx with existing content).
- Do NOT use `data_only=True` when loading — preserve existing formulas.
- Do NOT add sheets, macros, VBA, external links, or helper tabs.
- Do NOT alter formatting (don't touch fonts, fills, borders, etc.).
- Save to `/root/output/result.xlsx`.
- Before writing formulas, always inspect the actual cell contents and layout to confirm assumptions.

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