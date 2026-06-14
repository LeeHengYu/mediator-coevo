# Task Instruction

Execute the following steps in order.

## 1. Inspect the workbook layout
```bash
cd /root && pip install openpyxl 2>/dev/null
python3 - <<'PYEOF'
import openpyxl, json
wb = openpyxl.load_workbook('data/workbook.xlsx', data_only=False)
for sn in wb.sheetnames:
    print(f'=== Sheet: {sn} ===')
    ws = wb[sn]
    print(f'  dims: {ws.dimensions}')
    for row in ws.iter_rows(min_row=1, max_row=min(ws.max_row, 55), values_only=False):
        for c in row:
            if c.value is not None:
                print(f'  {c.coordinate}: {repr(c.value)}')
PYEOF
```
Read the output carefully. Identify:
- The series codes in column D for rows 12-17, 19-24, 26-31.
- The years in row 10 for columns H-L.
- The Data sheet layout (rows 21-38): which row holds headers, which column holds series codes, where the year data starts.
- The labels in column D (or nearby) for rows 35-40 (the six plants), rows 42-47 (min/max/median/mean/25th/75th), and row 50.
- Any existing formulas or values already present.

## 2. Write the formulas
Using the layout information, write a Python script that:

a) Opens `/root/data/workbook.xlsx` with openpyxl (data_only=False).

b) For each yellow cell in H12:L17, H19:L24, H26:L31 on sheet `Task`, writes an `INDEX/MATCH` formula that:
   - Uses the series code from column D of that row (absolute row reference).
   - Uses the year from row 10 of that column (absolute column reference).
   - Looks up in the Data sheet rows 21:38.
   - Pattern: `=INDEX(Data!<data_range>,MATCH($D<row>,Data!<series_col>,0),MATCH(<year_cell>,Data!<year_header_row>,0))`
   - Adjust the exact ranges based on what you found in step 1.

c) For H35:L40, writes the Net production slack formula:
   `=(H12-H19)/H26*100` (adjusted for each row/column offset).
   - H12:L17 = Finished Output block
   - H19:L24 = Scrap And Rework block  
   - H26:L31 = Rated Production Capacity block
   - Each row in 35-40 corresponds to the same plant row offset in the three blocks above.

d) For H42:L47, writes column-wise statistics over H35:L40:
   - Row 42: `=MIN(H35:H40)` (or whichever row is labeled minimum)
   - Row 43: `=MAX(H35:H40)`
   - Row 44: `=MEDIAN(H35:H40)`
   - Row 45: `=AVERAGE(H35:H40)`
   - Row 46: `=PERCENTILE(H35:H40,0.25)` — use `PERCENTILE` NOT `PERCENTILE.INC`
   - Row 47: `=PERCENTILE(H35:H40,0.75)`
   - Match the actual row labels (min/max/median/mean/25th/75th) to the correct rows from step 1.

e) For H50:L50, writes:
   `=SUMPRODUCT(H35:H40,H26:H31)/SUM(H26:H31)` for each column H-L.

f) Saves to `/root/output/result.xlsx` (create `/root/output/` if needed).

## 3. Validate
```bash
python3 - <<'PYEOF'
import openpyxl
wb = openpyxl.load_workbook('/root/output/result.xlsx', data_only=False)
ws = wb['Task']
# Check formulas exist in key cells
for cell_addr in ['H12','L17','H19','L24','H26','L31','H35','L40','H42','H47','H50','L50']:
    c = ws[cell_addr]
    print(f'{cell_addr}: {repr(c.value)}')
    assert c.value is not None and str(c.value).startswith('='), f'{cell_addr} missing formula'
print('All key cells have formulas.')
PYEOF
```

Also open the file with openpyxl data_only=True (note: openpyxl won't evaluate, but at least confirm no obvious errors in formula strings).

## Critical Notes
- Use `PERCENTILE` (not `PERCENTILE.INC` or `PERCENTILE.EXC`) to avoid #NAME? errors.
- Use `INDEX/MATCH` pattern for lookups.
- Do NOT add sheets, macros, VBA, or external links.
- Do NOT alter existing formatting.
- Verify the exact row/column layout from step 1 before writing any formulas. Do not assume the layout matches other tasks.
- The correspondence between plant rows in the three lookup blocks (12-17, 19-24, 26-31) and the calculation block (35-40) must be row-by-row (first plant in row 12, 19, 26 maps to row 35, etc.).
- For the SUMPRODUCT weighted mean in row 50, the weights are the Rated Production Capacity values from the lookup block H26:L31 (not from the Data sheet directly).

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