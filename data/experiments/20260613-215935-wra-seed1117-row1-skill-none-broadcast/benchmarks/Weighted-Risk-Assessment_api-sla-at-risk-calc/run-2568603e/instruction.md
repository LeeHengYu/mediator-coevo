# Task Instruction

Execute the following steps exactly, in order.

## 0 – Environment setup
```bash
pip install openpyxl 2>/dev/null
mkdir -p /root/output
```

## 1 – Inspect the workbook layout
Open `/root/data/workbook.xlsx` with openpyxl (data_only=False) and print:
- Sheet names.
- On sheet `Task`:
  - Row 10 (at least columns D–L) to see year headers.
  - Column D rows 12–31 to see series codes.
  - Rows 35–50 column D labels.
  - Any existing content/formulas in H12, H35, H42, H50.
- On sheet `Data`:
  - Row 20 or 21 headers (columns A–Z or so) and a few data rows (21–38) to understand the lookup table layout (which column holds the series code, which row holds years, where values sit).

Print everything clearly so you can design correct formulas.

## 2 – Design formulas from the inspection
Based on what you see:

### Step 1 – Lookup formulas (H12:L17, H19:L24, H26:L31)
For each cell use an INDEX/MATCH/MATCH pattern:
```
=INDEX(Data!<value_range>, MATCH($D<row>, Data!<series_code_column>, 0), MATCH(H$10, Data!<year_header_row>, 0))
```
Make sure:
- `$D<row>` is the series code in column D of the current row on `Task`.
- `H$10` (or I$10, J$10 …) is the year from row 10 on `Task`.
- The Data ranges are correctly identified from your inspection (rows 21:38 for values, the corresponding column for series codes, the corresponding row for year headers).
- Use absolute row/column references where needed so the formula copies correctly across the 5 columns and 6 rows of each block.

### Step 2 – Net SLA buffer (H35:L40)
Formula per cell:
```
=(H<latency_preserved_row> - H<latency_consumed_row>) / H<covered_request_row> * 100
```
Map each of the 6 services (rows 35–40) to the corresponding rows in the three lookup blocks. Identify which block is Latency Budget Preserved, which is Latency Budget Consumed, and which is Covered Request Capacity from the row labels in column D (or nearby).

### Step 2 continued – Statistics (H42:L47)
For each column (H through L), in rows 42–47 place:
- MIN(H35:H40)
- MAX(H35:H40)
- MEDIAN(H35:H40)
- AVERAGE(H35:H40)
- PERCENTILE(H35:H40, 0.25)  (or PERCENTILE.INC)
- PERCENTILE(H35:H40, 0.75)  (or PERCENTILE.INC)

Check the row labels in column D to confirm which statistic goes in which row.

### Step 3 – Weighted mean (H50:L50)
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```
(Weights are Covered Request Capacity from the third lookup block.)

## 3 – Write formulas with openpyxl
Using openpyxl, open the workbook, and for every cell described above, set `cell.value = '<formula_string>'`. Do NOT set data_only; write formula strings.

Critical rules:
- Do NOT delete or modify any existing formatting, sheets, or content outside the target cells.
- Do NOT add new sheets.
- Use the Translator or manual column-letter iteration to fill across H–L.
- Double-check that every formula references the correct sheet name (`Data`) with the `!` separator.

## 4 – Save
Save to `/root/output/result.xlsx`.

## 5 – Validate
Reopen `/root/output/result.xlsx` with openpyxl (data_only=False) and print:
- A sample of formulas from each block (e.g., H12, L17, H19, L24, H26, L31, H35, H40, H42, H47, H50, L50).
- Confirm none are None or empty.
- Confirm they reference `Data!` and use INDEX/MATCH (for lookups) or the expected statistical functions.

If any cell is None or wrong, fix and re-save before finishing.

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