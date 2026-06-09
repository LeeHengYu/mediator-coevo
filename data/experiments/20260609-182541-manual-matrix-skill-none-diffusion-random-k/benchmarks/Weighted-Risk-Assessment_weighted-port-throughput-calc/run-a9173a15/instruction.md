# Task Instruction

Execute the following steps in order to produce `/root/output/result.xlsx`.

## 0 — Preparation
```bash
mkdir -p /root/output
```
Open `/root/data/workbook.xlsx` with openpyxl (with `data_only=False` so existing formulas are preserved).

Before writing any formulas, inspect:
- Sheet `Task`: confirm column D has series codes in rows 12-17, 19-24, 26-31. Confirm row 10 has years in columns H-L. Print these values so you know the exact cell references.
- Sheet `Data`: confirm rows 21-38 contain the source data. Print a sample to understand the layout (which column holds the series code, which row holds years, etc.).

## 1 — Lookup formulas in yellow cells (H12:L17, H19:L24, H26:L31)

For each block, write an INDEX/MATCH formula. The pattern for cell `H12` (and similarly for every cell in the three blocks) is:

```
=INDEX(Data!$A$21:$ZZ$38,MATCH($D12,Data!$A$21:$A$38,0),MATCH(H$10,Data!$A$20:$ZZ$20,0))
```

Adjust the column/row ranges after inspecting the actual Data sheet layout:
- The MATCH for the series code should search the column that contains series codes on the Data sheet (likely column A or B — inspect first).
- The MATCH for the year should search the header row of the Data sheet (likely row 20 or 21 — inspect first).
- Use `$D12` (absolute column, relative row) and `H$10` (relative column, absolute row) so the formula copies correctly across the 5-column × 6-row blocks.

Write these formulas into every cell in H12:L17, H19:L24, and H26:L31.

## 2 — Net container flow (H35:L40)

The formula for `H35` is:
```
=(H12-H19)/H26*100
```
where row 12-17 = Loaded Containers Inbound, row 19-24 = Loaded Containers Outbound, row 26-31 = Terminal Throughput Capacity. Map each of the 6 ports (rows 35-40) to the corresponding rows in the three blocks (rows 12&19&26 for port 1, 13&20&27 for port 2, etc.). Write these for H35:L40.

## 3 — Descriptive statistics (H42:L47)

For each column (H through L):
- Row 42 (Minimum): `=MIN(H35:H40)`
- Row 43 (Maximum): `=MAX(H35:H40)`
- Row 44 (Median): `=MEDIAN(H35:H40)`
- Row 45 (Mean): `=AVERAGE(H35:H40)`
- Row 46 (25th percentile): `=_xlfn.PERCENTILE.INC(H35:H40,0.25)`
- Row 47 (75th percentile): `=_xlfn.PERCENTILE.INC(H35:H40,0.75)`

**Critical**: Use the `_xlfn.` prefix for PERCENTILE.INC. Without it, the formula will produce `#NAME?` errors. This was the exact cause of the previous failure.

## 4 — Weighted mean for CPA (H50:L50)

For each column (H through L):
```
=SUMPRODUCT(H35:H40,H26:H31)/SUM(H26:H31)
```
This computes the weighted mean of the net container flow percentages using Terminal Throughput Capacity as weights.

## 5 — Verify row labels

Before writing formulas, read the labels in column A (or B/C/G) for rows 42-47 and row 50 to confirm which row is min, max, median, mean, 25th pct, 75th pct, and weighted mean. Adjust row assignments if the actual labels differ from the assumed order above.

## 6 — Save

Save the workbook to `/root/output/result.xlsx`. Do NOT change formatting, do NOT add sheets or macros.

## 7 — Post-save validation

Reopen `/root/output/result.xlsx` with openpyxl and print the formula strings in cells H46, H47, and H50 to confirm:
- H46 contains `_xlfn.PERCENTILE.INC`
- H47 contains `_xlfn.PERCENTILE.INC`
- H50 contains `SUMPRODUCT`

Also spot-check a few lookup cells (e.g., H12, L31) to confirm they contain INDEX/MATCH formulas referencing the Data sheet.

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