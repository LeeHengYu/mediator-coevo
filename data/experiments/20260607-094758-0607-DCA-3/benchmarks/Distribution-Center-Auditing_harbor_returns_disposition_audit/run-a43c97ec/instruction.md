# Task Instruction

Build two deliverables for a returns-disposition audit. Inputs are at `/root/Return_Plan.xlsx`, `/root/Disposition_Event_Log.xlsx`, and `/root/Disposition_Alias.xlsx`. Outputs must be `/root/Returns_Disposition_Audit.xlsx` and `/root/Returns_Disposition_Brief.docx`.

Step 0 — Inspect inputs first. Before writing any logic, load each input file with pandas and print: column names, dtypes, row counts, and 5-10 sample rows. Confirm the exact column header spellings for Return ID, Line ID, Planned Disposition, Reason Code, Requested Qty, Warehouse, Carrier, Lane in `Return_Plan.xlsx`; for Return ID, Line ID, Event Status, Final Disposition, and any timestamp/sequence column in `Disposition_Event_Log.xlsx`; and the alias→standard column pair in `Disposition_Alias.xlsx`. Do not assume header names — use what the files actually contain.

Step 1 — Build `RawData` sheet: copy the plan table from `Return_Plan.xlsx` exactly (all columns, original order, original values).

Step 2 — Derive the latest COMPLETED event per (Return ID, Line ID):
- Filter `Disposition_Event_Log.xlsx` to rows where `Event Status` equals `COMPLETED` (case-insensitive match, but preserve original).
- Sort by the event timestamp/sequence column (identify it during Step 0; common names: Event Time, Timestamp, Event Date, Sequence). Use the column that determines recency.
- For each (Return ID, Line ID), keep the latest row.
- Important: verify the dedup yields one row per key. Print count of unique (Return ID, Line ID) groups vs. plan rows so you can confirm coverage.

Step 3 — Build alias normalization map from `Disposition_Alias.xlsx`. Create a case-insensitive dict from alias → standard disposition. Define a normalize(text) function: if lower(text) matches an alias key, return the standard; else return text unchanged. Comparisons throughout must be case-insensitive.

Step 4 — Build `Formatted Data` sheet, preserving RawData row order:
- Columns 1-8 exactly: Return ID, Line ID, Planned Disposition, Reason Code, Requested Qty, Warehouse, Carrier, Lane.
- Column 9 `Missing Final Event`: 1 if no kept COMPLETED event exists for that (Return ID, Line ID), else 0.
- Column 10 `Disposition Mismatch`: 1 if a kept event exists AND normalize(Final Disposition).lower() != Planned Disposition.lower(); else 0. If Missing Final Event = 1, Disposition Mismatch must be 0 (no event to compare).
- Column 11 `Total Errors`: sum of cols 9 and 10.
- Column 12 `Error Summary`: exactly one of `None`, `Missing Final Event`, `Disposition Mismatch`, `Missing Final Event, Disposition Mismatch`.
- Write concrete numeric/text values (no formulas).
- Validation: print value_counts for cols 9, 10, 11 and confirm the row count equals RawData row count. Print total Missing Final Events, total Disposition Mismatches, total Total Errors — these must be the same numbers used in the Word doc.

Step 5 — Build `Summary` sheet:
- Group `Formatted Data` by (Warehouse, Carrier), summing Missing Final Event, Disposition Mismatch, Total Errors.
- Keep ONLY groups where Total Errors > 0.
- Sort by Warehouse asc, then Carrier asc.
- Append final row: Warehouse=`Grand Total`, Carrier=`-`, then dataset totals (sum across ALL Formatted Data rows, not just filtered groups — verify by comparing to Step 4 totals).
- Headers exactly: Warehouse, Carrier, Missing Final Events, Disposition Mismatches, Total Errors.
- Validation: print the full Summary dataframe. Confirm Grand Total row's Total Errors matches the sum from Step 4.

Step 6 — Write `/root/Returns_Disposition_Audit.xlsx` using openpyxl/pandas ExcelWriter with sheet names exactly `RawData`, `Formatted Data`, `Summary` (in that order). After writing, reopen the file and print sheet names and a few rows of each to confirm.

Step 7 — Build `/root/Returns_Disposition_Brief.docx` using python-docx with a 3-6 sentence executive summary that includes:
- Plain-language definitions of `Missing Final Event` (no COMPLETED disposition event was recorded for the return line) and `Disposition Mismatch` (the completed event's normalized final disposition differs from the planned disposition).
- The exact computed totals from Step 4 for Missing Final Events, Disposition Mismatches, and Total Errors — write the actual integers into the text.
- At least one actionable recommendation (e.g., reconcile event logging at flagged warehouses/carriers).
- Mention at least two high-priority Return IDs that have the most exceptions (rank by per-Return-ID sum of Total Errors and cite the top two Return ID values literally).

Step 8 — Final validation pass: reopen both output files, print Summary Grand Total, and verify the integers in the docx match. Do not weaken any rule or skip rows.

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