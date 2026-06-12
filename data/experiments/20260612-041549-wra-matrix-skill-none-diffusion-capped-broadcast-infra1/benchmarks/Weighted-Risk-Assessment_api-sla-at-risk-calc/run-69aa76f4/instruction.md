# Task Instruction

Execute the following steps to produce /root/output/result.xlsx.

## 0 – Inspect the workbook
```bash
mkdir -p /root/output
```
Open /root/data/workbook.xlsx with openpyxl and inspect:
- Sheet `Task`: print rows 10-50 for columns D-L so you can see the series codes in column D, the years in row 10, the yellow target ranges, and the existing labels/layout.
- Sheet `Data`: print rows 21-38 to understand the lookup source structure (which column holds the series code, which row holds years, where values live).

Print cell coordinates, values, and any existing formulas. This is critical before writing anything.

## 1 – Populate lookup formulas in H12:L17, H19:L24, H26:L31

For every yellow cell in these three blocks, write a formula that looks up:
- The series code from column D of that row (e.g., $D12)
- The year from row 10 of that column (e.g., H$10)

against the Data sheet rows 21:38.

Choose INDEX/MATCH as the pattern because it is universally supported:
```
=INDEX(Data!<value_column_range>, MATCH($D12, Data!<series_code_column>, 0), MATCH(H$10, Data!<year_row>, 0))
```
Adjust the exact ranges after inspecting the Data sheet layout. The MATCH for the series code should search the column that contains series codes; the MATCH for the year should search the row that contains year headers. Lock references appropriately ($D12 for the row's series code, H$10 for the column's year).

## 2 – Net SLA buffer (H35:L40)

For each of the six services (rows 35-40) and each year column (H-L), write:
```
=(H12 - H19) / H26 * 100
```
adjusting row references so that:
- Row 12-17 → Latency Budget Preserved (first block)
- Row 19-24 → Latency Budget Consumed (second block)
- Row 26-31 → Covered Request Capacity (third block)

The mapping is: row 35↔rows 12,19,26; row 36↔rows 13,20,27; etc.

## 3 – Statistics block (H42:L47)

For each year column, calculate over H35:H40 (the six Net SLA buffer values):
- H42: =MIN(H35:H40)
- H43: =MAX(H35:H40)
- H44: =MEDIAN(H35:H40)
- H45: =AVERAGE(H35:H40)
- H46: 25th percentile
- H47: 75th percentile

**CRITICAL for percentiles:** openpyxl does not automatically add the `_xlfn.` prefix that xlsx files need for newer Excel functions. You MUST write the formulas with the prefix:
- H46: =_xlfn.PERCENTILE.INC(H35:H40,0.25)
- H47: =_xlfn.PERCENTILE.INC(H35:H40,0.75)

Alternatively, use the legacy `PERCENTILE` function but ALSO with the `_xlfn.` prefix to be safe:
- =_xlfn.PERCENTILE.INC(H35:H40,0.25)

If after inspection you find the verifier evaluates formulas via a Python engine (like formulas or xlcalc), check whether `_xlfn.PERCENTILE.INC` or plain `PERCENTILE` works. The previous failure was #NAME? errors on percentile cells, so this is the highest-risk area. Try `_xlfn.PERCENTILE.INC` first.

## 4 – Weighted mean (H50:L50)

For each year column:
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```
This computes the weighted mean of the Net SLA buffer percentages using Covered Request Capacity as weights.

## 5 – Save

Save the workbook to /root/output/result.xlsx. Do NOT change formatting, do NOT add sheets.

## 6 – Validate

After saving, re-open the file with openpyxl (data_only=False) and print all formula cells in the target ranges to confirm they contain formulas (not None). Then, if a test script exists at /root/test_output.py or similar, run it:
```bash
cd /root && python -m pytest test_output.py -v 2>&1 | head -80
```
If percentile cells fail with #NAME?, try switching to plain `PERCENTILE` (without `_xlfn.` prefix) or `_xlfn.PERCENTILE` and re-run. If that also fails, try the formulas library to check what function names it recognizes.

## Key Warnings
- The previous run failed because percentile formulas returned #NAME?. Fix this by using the correct function prefix.
- A sibling task (weighted-hospital-bedflow-calc) failed because cells were None — make sure every target cell gets a formula.
- Do not add sheets, macros, VBA, or external links.
- Preserve all existing formatting.

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