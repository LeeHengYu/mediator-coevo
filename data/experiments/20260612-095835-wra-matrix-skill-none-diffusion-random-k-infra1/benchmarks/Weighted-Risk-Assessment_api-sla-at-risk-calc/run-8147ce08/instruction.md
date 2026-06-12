# Task Instruction

Execute the following steps to produce /root/output/result.xlsx from /root/data/workbook.xlsx.

## 0 – Inspect the workbook
1. Open /root/data/workbook.xlsx with openpyxl (data_only=False) and list every sheet name.
2. On sheet `Task`, print cells D12:D17, D19:D24, D26:D31 (series codes), row 10 columns H–L (years), and cells H35:L47, H50:L50 to understand current content.
3. On sheet `Data`, print rows 21–38 (all columns with data) so you know the layout: which column holds the series code, which row holds years, and where numeric values live.
4. Print any existing content / formatting notes in the yellow target ranges to confirm they are empty.

## 1 – Populate lookup formulas in H12:L17, H19:L24, H26:L31
For every cell in these three 6-row × 5-column blocks, write an Excel formula using the **INDEX/MATCH** pattern that:
- Uses the series code from column D of that row (e.g. $D12) as the row lookup key.
- Uses the year from row 10 of that column (e.g. H$10) as the column lookup key.
- Searches the Data sheet rows 21:38 for both keys.

Concretely, determine from your inspection:
- The column on `Data` that holds the series codes (call it col X, e.g. column B).
- The row on `Data` that holds the year headers (call it row Y, e.g. row 20 or row 21).
- The rectangular data range on `Data` that contains the numeric values.

Then write formulas like:
```
=INDEX(Data!<value_range>, MATCH($D12, Data!<series_code_column>, 0), MATCH(H$10, Data!<year_header_row>, 0))
```
Adjust references so they are correct for the actual layout. Use absolute references on the lookup vectors ($D12 locks column, H$10 locks row) and on the Data ranges.

## 2 – Net SLA buffer (H35:L40)
For each of the 6 services (rows 35–40) and 5 year-columns (H–L), write a formula:
```
=(H12 - H19) / H26 * 100
```
where H12 is from the Latency Budget Preserved block (rows 12–17), H19 from Latency Budget Consumed block (rows 19–24), and H26 from Covered Request Capacity block (rows 26–31). Adjust row references so each service row maps correctly (row 35↔rows 12,19,26; row 36↔rows 13,20,27; etc.).

## 3 – Summary statistics (H42:L47)
For each column H–L:
- Row 42: `=MIN(H35:H40)`
- Row 43: `=MAX(H35:H40)`
- Row 44: `=MEDIAN(H35:H40)`
- Row 45: `=AVERAGE(H35:H40)`
- Row 46: `=PERCENTILE(H35:H40, 0.25)`
- Row 47: `=PERCENTILE(H35:H40, 0.75)`

## 4 – Weighted mean (H50:L50)
For each column H–L:
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```

## 5 – Save
- Create /root/output/ if it doesn't exist.
- Save the workbook to /root/output/result.xlsx.
- Do NOT add sheets, macros, VBA, external links, or helper tabs.
- Do NOT alter existing formatting.

## 6 – Verify
- Reopen /root/output/result.xlsx with openpyxl (data_only=False).
- Print cells H12, L17, H19, L24, H26, L31 to confirm they contain formula strings (starting with '=').
- Print cells H35, L40, H42, L47, H50, L50 to confirm they contain formula strings.
- Confirm no extra sheets were added.

## Important notes
- Use openpyxl throughout. Load with data_only=False to preserve formulas.
- Read the actual file contents carefully before writing any formulas – the exact row/column layout on Data matters.
- If the Data sheet series codes or year headers are in different positions than assumed, adjust all formulas accordingly.
- The avoid-recheck artifact warns about cells returning None – this happens when formulas are not actually written. Double-check after writing that the cell.value is a string starting with '='.

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