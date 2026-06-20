# Task Instruction

Execute the following steps in order to produce `/root/oncocooler_analysis.json` and `/root/oncocooler_summary.md`.

## Step 1 – Inspect all input files

```bash
cat /root/program_catalog.json
cat /root/cooler_cost.csv
cat /root/contract_payment.csv
cat /root/site_overrides.csv
```

Read every field carefully before writing any code.

## Step 2 – Inspect the test/verifier file

```bash
cat /root/test_output.py 2>/dev/null || cat /root/tests/test_output.py 2>/dev/null || find /root -name 'test_output*' -exec cat {} \;
```

Understand exactly what the verifier checks (field names, tolerances, schema).

## Step 3 – Write and run the analysis script

Create `/root/solve.py` with the logic below. Key formulas and rules:

### Filtering
- Load `program_catalog.json`. Keep only programs where `review_flag == "review"`.

### Site count resolution
- Load `site_overrides.csv`. Keep only rows with `approval_state == "approved"`.
- For each `program_code`, keep the row with the highest `version_no`.
- For each in-scope program, use the `active_sites` from that override row. If no approved override exists for that program_code, fall back to `default_active_sites` from the catalog.

### Payment resolution
- Load `contract_payment.csv`. For each row, match its `program_label` to either `program_name` or any entry in the `known_labels` list from the catalog. Ignore rows that don't map to an in-scope program.
- Extract `payment_per_dispatch_per_site_usd`.

### Cooler cost resolution
- Load `cooler_cost.csv`. Match each program's `cooler_type` to get `cooler_cost_usd`.

### Per-program calculations (for BOTH 10-day and 20-day models)

For each in-scope program:

```
annual_revenue = payment_per_dispatch_per_site_usd * active_sites * dispatches_per_year

annual_drug_cost = acquisition_cost_per_1000_units_usd * active_sites * units_per_day * days_per_dispatch * dispatches_per_year / 1000

annual_cooler_cost = cooler_cost_usd * active_sites * dispatches_per_year

annual_margin = annual_revenue - annual_drug_cost - annual_cooler_cost
```

**CRITICAL**: `annual_cooler_cost` must be multiplied by `active_sites` AND `dispatches_per_year`. The previous run failed because `active_sites` was likely missing from the cooler cost formula. The ratio 4536/302.4 = 15 matches `active_sites` being omitted. Make absolutely sure ALL three cost/revenue lines multiply by `active_sites * dispatches_per_year`.

10-day model: days_per_dispatch=10, dispatches_per_year=36
20-day model: days_per_dispatch=20, dispatches_per_year=18

```
annual_margin_difference = annual_margin_20_day - annual_margin_10_day
```

### Totals
```
total_annual_margin_10_day = sum of all program annual_margin_10_day
total_annual_margin_20_day = sum of all program annual_margin_20_day
total_difference = total_annual_margin_20_day - total_annual_margin_10_day
absolute_total_difference = abs(total_difference)
```

### Decision
- If `abs(total_difference) < 10000` → `"move_to_20_day"`
- Otherwise → `"keep_10_day"`

### Output JSON
- Round all currency values to 2 decimal places.
- Sort `programs` array by `program_code` ascending.
- Use the exact JSON schema from the task (keys: `assumptions`, `programs`, `totals`, `recommendation`).
- `recommendation` must have `decision` and `justification` sub-keys.

### Output Markdown (`/root/oncocooler_summary.md`)
- 4–8 non-empty lines.
- Must include: total 10-day margin (USD), total 20-day margin (USD), absolute difference (USD), and the exact decision slug (`move_to_20_day` or `keep_10_day`).

## Step 4 – Run the script

```bash
cd /root && python solve.py
```

## Step 5 – Validate outputs

```bash
cat /root/oncocooler_analysis.json
cat /root/oncocooler_summary.md
python -c "import json; d=json.load(open('/root/oncocooler_analysis.json')); print('Keys:', list(d.keys())); print('Programs:', len(d['programs'])); print('Totals:', d['totals']); print('Rec:', d['recommendation'])"
```

Verify:
- JSON is valid and parseable.
- All required top-level keys present: `assumptions`, `programs`, `totals`, `recommendation`.
- Each program object has all required fields.
- `programs` sorted by `program_code`.
- Summary has 4-8 non-empty lines with required content.

## Step 6 – Run verifier

```bash
cd /root && python -m pytest test_output.py -v 2>/dev/null || python -m pytest tests/test_output.py -v 2>/dev/null || echo 'No test file found at expected paths'
```

If any test fails, read the error carefully, fix `solve.py`, re-run, and re-verify. Pay special attention to:
- Numeric tolerances (the verifier likely uses `math.isclose` or similar)
- Exact key names
- The cooler cost formula including `active_sites * dispatches_per_year`
- Correct site override resolution (highest version_no among approved rows)

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
Task metadata: author_email=gpt54@example.com, author_name=GPT-5.4, category=financial-analysis, difficulty=medium, tags=[oncology, json, csv, structural-adaptation, decision-analysis].
Verifier config: timeout_sec=900.0.