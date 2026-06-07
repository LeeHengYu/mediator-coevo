# Task Instruction

## Task: Update /root/data/workbook.xlsx with Excel formulas

You must write Excel formulas (not computed values) into specific cell ranges on the 'Task' sheet, then save to /root/output/result.xlsx.

### Phase 0: Inspect the workbook thoroughly

1. `mkdir -p /root/output`
2. Use openpyxl to open `/root/data/workbook.xlsx` and inspect:
   - Sheet names (confirm 'Task' and 'Data' exist)
   - On 'Task' sheet: read column D values in rows 12-17, 19-24, 26-31 (these are series codes)
   - On 'Task' sheet: read row 10, columns H through L (these are years)
   - On 'Task' sheet: read what's in rows 35-40 column D (port names for net container flow)
   - On 'Task' sheet: read rows 42-47 column D or nearby (stat labels: min, max, median, mean, 25th, 75th percentile)
   - On 'Task' sheet: read row 50 area (CPA weighted mean label)
   - On 'Task' sheet: read the block labels near rows 12, 19, 26 to understand which block is which metric (identify which is Loaded Containers Inbound, Loaded Containers Outbound, Terminal Throughput Capacity)
   - On 'Data' sheet: inspect rows 21-38 structure — identify columns, what row contains headers, how series codes map to data, where years appear
   - Print all of this information clearly

### Phase 1: Write lookup formulas in H12:L17, H19:L24, H26:L31

Based on your inspection, write Excel formulas using INDEX/MATCH pattern. Each formula should:
- Use the series code from column D of that row (e.g., $D12 for row 12)
- Use the year from row 10 of that column (e.g., H$10 for column H)
- Look up data from the 'Data' sheet rows 21:38

The typical pattern would be something like:
`=INDEX(Data!$B$21:$XX$38, MATCH($D12, Data!$A$21:$A$38, 0), MATCH(H$10, Data!$B$20:$XX$20, 0))`

But you MUST adjust the exact ranges based on your inspection of the Data sheet layout. Determine:
- Which column contains the series codes on the Data sheet
- Which row contains the year headers on the Data sheet  
- What the data range is

Use absolute references ($) for the lookup arrays and mixed references for the lookup values so formulas can be filled across the range.

### Phase 2: Net container flow formulas in H35:L40

For each port (6 ports, rows 35-40) and each year (columns H-L):
`=(InboundCell - OutboundCell) / CapacityCell * 100`

Where InboundCell is from the H12:L17 block, OutboundCell from H19:L24, and CapacityCell from H26:L31. Match the port ordering — verify that rows 35-40 correspond to the same ports as rows 12-17 (and 19-24, 26-31). If ordering differs, adjust references accordingly.

### Phase 3: Statistics in H42:L47

For each column (H through L), calculate column-wise stats over H35:L40:
- Row 42: `=MIN(H35:H40)` (adjust column letter)
- Row 43: `=MAX(H35:H40)`
- Row 44: `=MEDIAN(H35:H40)`
- Row 45: `=AVERAGE(H35:H40)`
- Row 46: `=PERCENTILE(H35:H40, 0.25)` — use PERCENTILE, NOT PERCENTILE.INC (openpyxl/older Excel compatibility)
- Row 47: `=PERCENTILE(H35:H40, 0.75)`

**IMPORTANT**: Verify which row is which statistic by reading the labels in column D or nearby. The order (min/max/median/mean/25th/75th) may differ from what I listed — match the actual labels.

### Phase 4: Weighted mean in H50:L50

For each column H through L:
`=SUMPRODUCT(H35:H40, H26:H31) / SUM(H26:H31)`

This calculates the weighted mean of net container flow percentages using Terminal Throughput Capacity as weights.

### Phase 5: Save and verify

1. Save to `/root/output/result.xlsx` preserving all formatting.
2. Re-open the saved file and verify:
   - Cells H12:L17, H19:L24, H26:L31 contain formula strings (not None)
   - Cells H35:L40 contain formula strings
   - Cells H42:L47 contain formula strings
   - Cells H50:L50 contain formula strings
   - Print a sample of formulas from each block

### Critical Notes
- Use `openpyxl` to read and write. Do NOT use data_only=True when writing.
- Write formulas as strings starting with '=' into cells.
- Do NOT add new sheets, macros, VBA, or external links.
- Do NOT delete or modify existing formatting.
- If any cell already has content/formatting, only set the value (formula), don't clear formatting.
- Use PERCENTILE (not PERCENTILE.INC or PERCENTILE.EXC) for the percentile functions.
- Use AVERAGE (not MEAN) for the mean.
- Double-check the Data sheet layout carefully before writing any formulas.

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