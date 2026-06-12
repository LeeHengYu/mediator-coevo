# Task Instruction

Execute the following steps in order.

## 1 – Inspect the workbook structure

Open `/root/data/workbook.xlsx` with openpyxl (data_only=False). For both sheets (`Task` and `Data`) print:

* Sheet names in the workbook.
* **Task sheet:** rows 9–50, columns A–L (print every cell's coordinate and value).
* **Data sheet:** rows 1–40, all used columns (print every cell's coordinate and value).

Pay special attention to:
- Row 10 on Task: which years appear in H10:L10.
- Column D on Task: which series codes appear in D12:D17, D19:D24, D26:D31.
- Data sheet rows 21–38: what is the layout? Which row holds headers, which column holds series codes, and which columns hold year data? Print the exact cell references.
- Task sheet rows 35–40: labels in column D or nearby for the six hospital clusters.
- Task sheet rows 42–47: labels for min, max, median, mean, 25th pctl, 75th pctl.
- Task sheet row 50: label for Regional Care Grid weighted mean.
- H26:L31 block description (Staffed Bed Capacity).

## 2 – Build and write formulas

Using the exact layout discovered above, write a Python/openpyxl script that:

### Step 1 – Lookup formulas in H12:L17, H19:L24, H26:L31

For each cell (row r, column c) in these three blocks, write an `INDEX/MATCH` formula of the form:

```
=INDEX(Data!<data_area>, MATCH(D{r}, Data!<series_code_column>, 0), MATCH(Task!{col}10, Data!<year_header_row>, 0))
```

where:
- `<data_area>` is the rectangular range on Data that contains the numeric values (rows 21–38, data columns only — exclude the series-code column).
- `<series_code_column>` is the column on Data holding series codes for those same rows.
- `<year_header_row>` is the row on Data holding year headers (only data columns).
- `D{r}` is the series code on Task for that row.
- `{col}10` is the year on Task for that column.

Make sure the `$` signs anchor the lookup arrays but NOT the lookup values, so the formula can conceptually vary by row/column. Use absolute references for the Data ranges.

### Step 2 – Net capacity headroom (H35:L40)

For each of the six clusters (rows 35–40) and each year column (H–L), write:

```
=(H{avail} - H{occup}) / H{staffed} * 100
```

where `{avail}` is the corresponding row in H12:L17, `{occup}` is the corresponding row in H19:L24, and `{staffed}` is the corresponding row in H26:L31. The row offset between the three blocks should be consistent (cluster 1 = rows 12,19,26; cluster 2 = rows 13,20,27; etc.).

### Summary statistics (H42:L47)

For each year column c (H–L), write formulas in:
- Row 42: `=MIN({c}35:{c}40)`
- Row 43: `=MAX({c}35:{c}40)`
- Row 44: `=MEDIAN({c}35:{c}40)`
- Row 45: `=AVERAGE({c}35:{c}40)`
- Row 46: `=PERCENTILE({c}35:{c}40,0.25)` — use `PERCENTILE` (NOT `PERCENTILE.INC`; the previous run got #NAME? with the dotted variant)
- Row 47: `=PERCENTILE({c}35:{c}40,0.75)`

**Important:** If during inspection you see that the labels in rows 42-47 are in a different order (e.g., min is not row 42), match the formula to the label, not to the row number.

### Step 3 – Weighted mean (H50:L50)

For each year column c:
```
=SUMPRODUCT({c}35:{c}40, {c}26:{c}31) / SUM({c}26:{c}31)
```

## 3 – Save

Save the workbook to `/root/output/result.xlsx`. Create the `/root/output/` directory if it does not exist.

## 4 – Verify

Reopen `/root/output/result.xlsx` with openpyxl (data_only=False) and print the formulas in:
- H12, L17 (first/last of block 1)
- H19, L24 (first/last of block 2)
- H26, L31 (first/last of block 3)
- H35, L40 (first/last of headroom)
- H42:H47 (all summary stats for first year column)
- H50 (weighted mean)

Confirm none are None and none contain `#` error markers in the formula string itself.

## Key cautions
- Do NOT use `PERCENTILE.INC` or `PERCENTILE.EXC` — use plain `PERCENTILE`.
- Do NOT add sheets, macros, VBA, external links, or helper tabs.
- Do NOT change any existing formatting.
- Use English-locale comma-separated arguments.
- Anchor Data ranges with `$` for robustness.
- If any step's inspection reveals a layout different from what's assumed above, adapt the formulas to the actual layout before writing.

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