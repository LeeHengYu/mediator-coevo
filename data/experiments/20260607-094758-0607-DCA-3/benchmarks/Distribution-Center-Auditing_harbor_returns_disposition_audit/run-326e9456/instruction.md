# Task Instruction

Build two deliverables for a returns-processing disposition audit using the three input files at `/root/Return_Plan.xlsx`, `/root/Disposition_Event_Log.xlsx`, and `/root/Disposition_Alias.xlsx`.

## Step 1: Inspect Source Files First
Before writing any code that produces output, load and print:
- All columns and the first ~15 rows of `Return_Plan.xlsx` (note exact column names, dtypes, and any literal strings like 'N/A').
- All columns and a sample of `Disposition_Event_Log.xlsx` (note Event Status values, timestamp column name, and how 'latest' is determined — likely a timestamp or sequence column).
- All rows of `Disposition_Alias.xlsx` (note which column is the alias vs. the standard disposition).
Confirm the join keys `(Return ID, Line ID)` exist in both Plan and Event Log with matching dtypes (cast both to string before joining to avoid type-mismatch drops).

## Step 2: Create `/root/Returns_Disposition_Audit.xlsx` with three sheets

### Sheet `RawData`
- Copy the Return_Plan table EXACTLY as read. Preserve string literals such as 'N/A' — do NOT let pandas/openpyxl coerce them to NaN/None. Load with `keep_default_na=False` (or equivalent) and write back the exact cell values. After writing, re-read the sheet and spot-check that any 'N/A' literal in the source is still 'N/A' in the output (not blank, not 'None').

### Sheet `Formatted Data`
- Same row order as RawData.
- Columns 1–8: Return ID, Line ID, Planned Disposition, Reason Code, Requested Qty, Warehouse, Carrier, Lane — copy exactly from RawData.
- Columns 9–12: Missing Final Event, Disposition Mismatch, Total Errors, Error Summary.

Derivation logic (implement carefully):
1. From `Disposition_Event_Log`, filter to rows with `Event Status == 'COMPLETED'` (case-insensitive match; ignore all other statuses).
2. For each `(Return ID, Line ID)` group, keep ONLY the latest row. Identify the timestamp/sequence column from inspection in Step 1 and sort descending by it; take the first row per group. Cast join keys to string on both sides.
3. Build alias map from `Disposition_Alias.xlsx`: lowercase(alias) -> standard disposition. When normalizing a kept event's `Final Disposition`, lowercase it and look up in the map; if no match, use the raw text. Compare to `Planned Disposition` case-insensitively.
4. Left-join Plan to the kept-events table on (Return ID, Line ID) as strings.
   - `Missing Final Event` = 1 if no kept COMPLETED event row joined, else 0.
   - `Disposition Mismatch` = 1 if a kept event exists AND normalized Final Disposition != Planned Disposition (case-insensitive), else 0. If Missing Final Event = 1, Disposition Mismatch = 0.
   - `Total Errors` = sum of the two.
   - `Error Summary` ∈ {`None`, `Missing Final Event`, `Disposition Mismatch`, `Missing Final Event, Disposition Mismatch`} based on which flags are set.
- Write concrete numeric/text values (no formulas).

### Sheet `Summary`
Headers exactly: Warehouse, Carrier, Missing Final Events, Disposition Mismatches, Total Errors.
- Aggregate Formatted Data by (Warehouse, Carrier), summing the three error counts.
- Include only groups where Total Errors > 0.
- Sort by Warehouse asc, then Carrier asc.
- Append final row: Warehouse='Grand Total', Carrier='-', and the three columns as dataset-wide totals (sum across the included groups must equal the totals computed from the full Formatted Data — verify).

## Step 3: Validation Before Writing Word Doc
Print and verify:
- Total rows in Formatted Data == rows in RawData == rows in Return_Plan.
- Sum of Missing Final Event column.
- Sum of Disposition Mismatch column.
- Sum of Total Errors column.
- Per-(Warehouse,Carrier) breakdown matches the Summary sheet (including the example cell ('WH-A','CarrierX') — if its total is unexpectedly small, re-inspect the join: confirm key dtypes match, confirm 'latest' selection is per-group not global, confirm COMPLETED filter is case-insensitive).
- Spot-check one row that should be Missing Final Event = 1 (a Plan row whose (Return ID, Line ID) has no COMPLETED event).

## Step 4: Create `/root/Returns_Disposition_Brief.docx`
A 3–6 sentence executive summary that MUST include:
- Plain-language definition of `Missing Final Event` (no completed disposition event was recorded for the planned return line) and `Disposition Mismatch` (the recorded final disposition differs from what was planned).
- The exact computed totals from Step 3: total Missing Final Events, total Disposition Mismatches, total Total Errors (write each number as a digit in the prose so it appears literally in the doc).
- At least one actionable recommendation (e.g., reconcile missing events with the warehouse team, retrain on disposition coding, etc.).
- At least two specific high-priority Return IDs — pick Return IDs from Formatted Data that have the most exceptions (group by Return ID, sum Total Errors, take top 2; include the literal ID strings in the doc).

## Constraints
- Exact filenames and sheet names as specified.
- Preserve source literals like 'N/A'; do not coerce to None/NaN.
- Cast join keys to string to avoid silent drops.
- Do not use formulas; write concrete values.
- After creating both files, re-open each and print key checks (sheet names present, headers exact, totals consistent between Summary Grand Total and Word doc numbers).

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
Task metadata: author_email=catpaw@meituan.com, author_name=CatPaw Benchmark Builder, category=spreadsheet-audit, difficulty=hard, tags=[excel, openpyxl, docx, audit, returns].
Verifier config: timeout_sec=900.0.