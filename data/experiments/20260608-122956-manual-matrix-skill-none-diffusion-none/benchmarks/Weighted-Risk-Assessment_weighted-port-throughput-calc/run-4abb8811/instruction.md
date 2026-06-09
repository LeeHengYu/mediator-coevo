# Task Instruction

Execute the following steps to produce /root/output/result.xlsx from /root/data/workbook.xlsx.

## Phase 0 – Setup
```bash
mkdir -p /root/output
```

## Phase 1 – Inspect the workbook structure
Using openpyxl (read-only, data_only=False), inspect:
1. Sheet names (confirm 'Task' and 'Data' exist).
2. On 'Task' sheet: read cells D12:D17, D19:D24, D26:D31 (series codes), H10:L10 (years), H35:H40 labels or D35:D40 (port names for Step 2), H42:H47 labels (stat names), H50 label.
3. On 'Data' sheet: read row 21 through row 38 completely – note column A (or whichever column holds the series code) and the header row for years. Identify the exact column that holds the series codes and the exact row that holds the year headers. Print all series codes verbatim (repr) so we can match them exactly.
4. Print all findings before writing any formulas.

## Phase 2 – Write formulas with openpyxl
Open workbook.xlsx with openpyxl (keep formatting: do NOT use data_only). Based on the inspection results, write formulas into the yellow cells.

### Step 1 – Lookup formulas in H12:L17, H19:L24, H26:L31
Use the INDEX/MATCH pattern:
```
=INDEX(Data!<data_range>, MATCH($D12, Data!<series_code_column>, 0), MATCH(H$10, Data!<year_header_row>, 0))
```
Replace `<data_range>`, `<series_code_column>`, and `<year_header_row>` with the actual ranges found in Phase 1. Use absolute row references for the year row ($10) and absolute column references for the series code column ($D). Make sure:
- The series code column range and year header row range are anchored with $ so they don't shift when the formula is filled across rows/columns.
- The data_range covers the full block of numeric data corresponding to the series codes and years.
- String matching: if series codes or years have any type mismatch risk, wrap the lookup values to ensure consistency (though typically INDEX/MATCH handles this if both sides are the same type).

Fill all 6 rows × 5 columns for each of the three blocks (H12:L17, H19:L24, H26:L31).

### Step 2 – Net container flow (H35:L40) and statistics (H42:L47)
For H35:L40, each cell computes:
```
=(H12 - H19) / H26 * 100
```
Adjust row references for each of the 6 ports. The pattern: row 12+i maps to Loaded Inbound, row 19+i maps to Loaded Outbound, row 26+i maps to Terminal Throughput Capacity, for i=0..5.

For H42:L47 (column-wise statistics over H35:L40):
- H42: =MIN(H35:H40)
- H43: =MAX(H35:H40)
- H44: =MEDIAN(H35:H40)
- H45: =AVERAGE(H35:H40)
- H46: =PERCENTILE(H35:H40, 0.25)
- H47: =PERCENTILE(H35:H40, 0.75)

**Important**: Use `PERCENTILE` (legacy name), NOT `PERCENTILE.INC`, to avoid #NAME? errors.

Fill columns H through L.

### Step 3 – Weighted mean (H50:L50)
```
=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)
```
Fill columns H through L.

## Phase 3 – Save and verify
Save to /root/output/result.xlsx.
Re-open the saved file (data_only=False) and print the formulas in a few sample cells (e.g., H12, L17, H35, H42, H46, H50) to confirm they are correctly written.

## Critical constraints
- Do NOT add or remove sheets.
- Do NOT add macros, VBA, external links, or helper tabs.
- Do NOT change existing formatting.
- Only write into the specified cell ranges.
- Use openpyxl to preserve existing content and formatting.

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