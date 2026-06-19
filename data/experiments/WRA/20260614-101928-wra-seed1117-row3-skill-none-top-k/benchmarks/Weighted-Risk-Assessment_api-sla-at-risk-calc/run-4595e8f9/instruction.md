# Task Instruction

Execute the following steps to produce /root/output/result.xlsx.

## 0 – Preparation
```bash
mkdir -p /root/output
pip install openpyxl
```

## 1 – Inspect the workbook structure
Open `/root/data/workbook.xlsx` with openpyxl (data_only=False) and print:
- Sheet names.
- On sheet `Task`: the contents of cells D12:D17, D19:D24, D26:D31 (series codes), row 10 columns H–L (year headers), rows 35–40 column D (service labels), rows 42–47 column D or E (stat labels: min, max, median, mean, 25th, 75th), row 50 label, and the contents of H26:L31 (to confirm the Covered Request Capacity block location).
- On sheet `Data`: row 21 columns A–G (header row with years), column B rows 22–38 (series codes), and a sample of values to confirm the data layout.

This inspection is critical. Do NOT skip it. Record the exact cell contents before writing any formulas.

## 2 – Write formulas into the workbook
Using openpyxl (load workbook with data_only=False, keep_vba=False), write formulas as described below. Use `INDEX/MATCH` pattern throughout.

### Step 1 – Lookup formulas in H12:L17, H19:L24, H26:L31
For each cell at row `r`, column `c` (H=8 … L=12):
```
=INDEX(Data!$C$22:$G$38, MATCH(Task!$D{r}, Data!$B$22:$B$38, 0), MATCH(Task!{col_letter}$10, Data!$C$21:$G$21, 0))
```
Adjust the Data sheet ranges based on what you found in Step 1. The key references:
- `$D{r}` → series code in column D of the current row (absolute column, relative row)
- `{col_letter}$10` → year in row 10 (relative column, absolute row)
- Data lookup area: the rectangular block of numeric values (exclude header row and label column)
- Row keys: series codes in Data column B
- Column keys: years in Data row 21

Verify the ranges match what you observed. If the data rows are e.g. 22:38, the lookup array must be $C$22:$G$38 and the row-key vector $B$22:$B$38.

### Step 2 – Net SLA buffer in H35:L40
For each cell at row `r` (35–40), column `c` (H–L):
```
=(Task!{c}{{latency_preserved_row}} - Task!{c}{{latency_consumed_row}}) / Task!{c}{{covered_capacity_row}} * 100
```
Map the six services: row 35 corresponds to the first service whose Latency Budget Preserved is in row 12, Latency Budget Consumed is in row 19, and Covered Request Capacity is in row 26. Row 36→rows 13,20,27; etc. through row 40→rows 17,25,31. Confirm this mapping from the series codes / service labels you inspected.

### Step 2 – Summary statistics in H42:L47
For each column `c` (H–L), write these formulas:
- Row 42 (minimum): `=MIN({c}35:{c}40)`
- Row 43 (maximum): `=MAX({c}35:{c}40)`
- Row 44 (median): `=MEDIAN({c}35:{c}40)`
- Row 45 (mean): `=AVERAGE({c}35:{c}40)`
- Row 46 (25th percentile): `=PERCENTILE({c}35:{c}40, 0.25)`
- Row 47 (75th percentile): `=PERCENTILE({c}35:{c}40, 0.75)`

**Important**: Use `PERCENTILE` not `PERCENTILE.INC` or `PERCENTILE.EXC`. The failed artifact from a sibling task got #NAME? errors from using an unsupported function name. Stick to `MIN`, `MAX`, `MEDIAN`, `AVERAGE`, `PERCENTILE`.

However, verify the stat labels in column D/E of rows 42–47 to confirm the correct order (min/max/median/mean/25th/75th). Adjust row assignments if the labels differ.

### Step 3 – Weighted mean in H50:L50
For each column `c` (H–L):
```
=SUMPRODUCT({c}35:{c}40, {c}26:{c}31) / SUM({c}26:{c}31)
```
This computes the weighted mean of Net SLA buffer percentages weighted by Covered Request Capacity.

## 3 – Save
Save the workbook to `/root/output/result.xlsx`. Do NOT change formatting, do NOT add sheets.

## 4 – Validate
Reload the saved workbook (data_only=False) and print the formula strings in a sample of cells (e.g., H12, L17, H35, L40, H42, H47, H50, L50) to confirm they look correct. Also confirm sheet names are unchanged and no extra sheets exist.

## Key Warnings
- Do NOT use dotted function names like PERCENTILE.INC — use PERCENTILE.
- Do NOT hardcode values; use formulas only.
- Do NOT modify any existing formatting.
- Adjust all Data-sheet ranges based on your actual inspection in Step 1.
- If the stat-label order in rows 42–47 differs from min/max/median/mean/25th/75th, match your formulas to the actual labels.

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