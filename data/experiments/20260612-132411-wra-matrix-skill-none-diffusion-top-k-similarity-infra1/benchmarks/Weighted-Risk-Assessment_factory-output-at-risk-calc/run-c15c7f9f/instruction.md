# Task Instruction

Execute the following steps to produce /root/output/result.xlsx:

1. **Inspect the workbook** – Open /root/data/workbook.xlsx with openpyxl (data_only=False). Print:
   - Sheet names.
   - From sheet `Task`: cells D12:D17, D19:D24, D26:D31 (series codes), row 10 columns H–L (years), cells H35:H40 row labels or D35:D40, H42:H47 labels, H50 label, and any existing content/formatting in the yellow target ranges.
   - From sheet `Data`: rows 21–38 to understand the data layout (which row holds headers, which column holds the series code key, how years are arranged).
   This inspection is critical – do NOT skip it.

2. **Write lookup formulas in H12:L17, H19:L24, H26:L31** – For each cell (r, c) in these three 6×5 blocks, write an INDEX+MATCH formula that:
   - Looks up the series code in column D of the current row against the key column in Data!$rows 21:38.
   - Looks up the year in row 10 of the current column against the header row in Data!$rows 21:38.
   - Uses the pattern: `=INDEX(Data!<data_range>, MATCH($D{row}, Data!<key_column>, 0), MATCH({col}$10, Data!<header_row>, 0))`
   - Use absolute references for the key column ($D{row}) and year row ({col}$10) so formulas copy correctly across the 6×5 grid.
   - Adjust the exact Data! ranges based on what you discover in step 1.

3. **Write Net Production Slack formulas in H35:L40** – For each of the 6 plants (rows 35–40) and 5 year columns (H–L):
   - Formula: `=(H12-H19)/H26*100` (adjusted for the correct row offsets for Finished Output block rows 12–17, Scrap And Rework block rows 19–24, and Rated Production Capacity block rows 26–31).
   - Specifically: row 35 uses data from rows 12, 19, 26; row 36 from 13, 20, 27; etc.

4. **Write statistics formulas in H42:L47** – For each column H–L:
   - H42: `=MIN(H35:H40)`
   - H43: `=MAX(H35:H40)`
   - H44: `=MEDIAN(H35:H40)`
   - H45: `=AVERAGE(H35:H40)`
   - H46: `=PERCENTILE(H35:H40,0.25)` — use PERCENTILE, NOT PERCENTILE.INC or PERCENTILE.EXC (dot-suffixed names cause #NAME? errors in openpyxl/Excel verification).
   - H47: `=PERCENTILE(H35:H40,0.75)` — same note.
   **IMPORTANT**: Verify the exact row-to-statistic mapping by checking any labels in column D or G for rows 42–47. Assign min/max/median/mean/p25/p75 to the correct rows based on the labels you find.

5. **Write weighted mean formula in H50:L50** – For each column H–L:
   - `=SUMPRODUCT(H35:H40,H26:H31)/SUM(H26:H31)`
   This computes the weighted mean of Net Production Slack values using Rated Production Capacity as weights.

6. **Save** – Save the workbook to /root/output/result.xlsx (create /root/output/ if needed). Do NOT change formatting, add sheets, macros, VBA, external links, or helper tabs.

7. **Validate** – Re-open the saved file with openpyxl and print the formulas in a few sample cells from each block (e.g., H12, L17, H35, L40, H42, H47, H50, L50) to confirm they look correct. Also check that both sheets still exist and no extra sheets were added.

Key cautions:
- Use `PERCENTILE` not `PERCENTILE.INC` or `PERCENTILE.EXC` — the dotted versions produce #NAME? errors (this was a failure mode in a related task).
- Confirm the exact row/column layout of Data! before writing any formulas.
- Match the statistic labels (min, max, median, mean, 25th pct, 75th pct) to the correct target rows by reading the existing labels.
- Use mixed references ($D12 and H$10) in INDEX+MATCH formulas for proper grid expansion.

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
Task metadata: author_email=catpaw@meituan.com, author_name=CatPaw Task Engineer, category=spreadsheet-formula-reuse, difficulty=hard, tags=[excel, formulas, lookup, statistics, weighted-mean].
Verifier config: timeout_sec=600.0.