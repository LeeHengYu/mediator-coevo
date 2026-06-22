# Task Instruction

Execute the following steps in order:

1. **Inspect all input files** to understand their structure:
```bash
cat /root/program_catalog.json
cat /root/cooler_cost.csv
cat /root/contract_payment.csv
cat /root/site_overrides.csv
```

2. **Inspect the test file** to understand exact verifier expectations:
```bash
cat /tests/test_outputs.py
```

3. **Create `/root/solve.py`** that does the following:

   a. Load all four input files (JSON for catalog, CSV for the rest).
   
   b. Filter `program_catalog.json` to only programs where `review_flag == "review"`. These are the in-scope programs.
   
   c. For each in-scope program, resolve `active_sites`:
      - Look in `site_overrides.csv` for rows matching the program's `program_code` where `approval_state == "approved"`.
      - If multiple approved rows exist for the same `program_code`, keep only the one with the highest `version_no`.
      - Use that row's active site count. If no approved row exists, use `default_active_sites` from the catalog.
   
   d. For each in-scope program, resolve `payment_per_dispatch_per_site_usd`:
      - Match `contract_payment.csv` rows by checking if the row's `program_label` equals either the program's `program_name` OR any entry in the program's `known_labels` array.
      - Ignore payment rows that don't map to any in-scope program.
   
   e. For each in-scope program, get `cooler_cost_usd` from `cooler_cost.csv` by matching on `cooler_type`.
   
   f. Compute per-program values using these exact formulas:
      - `annual_drug_cost = acquisition_cost_per_1000_units_usd * active_sites * units_per_day * days_per_dispatch * dispatches_per_year / 1000`
      - `annual_cooler_cost = cooler_cost_usd * dispatches_per_year` (NOTE: cooler cost is per dispatch, not per site — but re-check the data; if cooler_cost.csv suggests it's per dispatch total, use that. The formula for annual cooler cost is `cooler_cost_usd * dispatches_per_year` unless the task says otherwise. Actually the task says "Cooler cost uses cooler_cost_usd from cooler_cost.csv" without specifying per-site, so it's just `cooler_cost_usd * dispatches_per_year`.)
      - `annual_revenue = payment_per_dispatch_per_site_usd * active_sites * dispatches_per_year`
      - `annual_margin = annual_revenue - annual_drug_cost - annual_cooler_cost`
      - `annual_margin_difference_20_minus_10 = annual_margin_20_day - annual_margin_10_day`
   
   g. Round ALL currency values to 2 decimal places.
   
   h. Sort programs by `program_code` ascending (string sort).
   
   i. Compute totals:
      - `total_annual_margin_10_day_usd` = sum of all program `annual_margin_10_day_usd`
      - `total_annual_margin_20_day_usd` = sum of all program `annual_margin_20_day_usd`
      - `total_annual_margin_difference_20_minus_10_usd` = sum of all program `annual_margin_difference_20_minus_10_usd`
      - `absolute_total_margin_difference_usd` = abs(total_annual_margin_difference_20_minus_10_usd)
   
   j. Decision rule:
      - If `absolute_total_margin_difference_usd < 10000`, decision = `"move_to_20_day"`
      - Otherwise, decision = `"keep_10_day"`
   
   k. Build the output JSON with **exactly** these keys in `assumptions`:
      ```
      "dispatches_per_year_10_day": 36,
      "dispatches_per_year_20_day": 18,
      "days_per_dispatch_10_day": 10,
      "days_per_dispatch_20_day": 20,
      "switch_threshold_usd": 10000,
      "site_override_rule": "highest approved version_no per program_code, else default_active_sites"
      ```
   
   l. Each program object in the `programs` array must have **exactly** these keys (no more, no less):
      ```
      program_code, program_name, active_sites,
      acquisition_cost_per_1000_units_usd, units_per_day,
      cooler_type, cooler_cost_usd, payment_per_dispatch_per_site_usd,
      annual_drug_cost_10_day_usd, annual_drug_cost_20_day_usd,
      annual_cooler_cost_10_day_usd, annual_cooler_cost_20_day_usd,
      annual_revenue_10_day_usd, annual_revenue_20_day_usd,
      annual_margin_10_day_usd, annual_margin_20_day_usd,
      annual_margin_difference_20_minus_10_usd
      ```
      **CRITICAL**: Use exactly these key names with the `_usd` suffix. Do NOT add extra fields. Do NOT omit any field. Match the schema from the task instruction character-for-character.
   
   m. Write the JSON to `/root/oncocooler_analysis.json` with `json.dump(..., indent=2)`.
   
   n. Write `/root/oncocooler_summary.md` with 4-8 non-empty lines including:
      - Total 10-day margin (USD)
      - Total 20-day margin (USD)
      - Absolute difference (USD)
      - The exact decision slug (`move_to_20_day` or `keep_10_day`)

4. **Run the solver**:
```bash
cd /root && python solve.py
```

5. **Validate the output** — check the JSON structure:
```bash
python3 -c "
import json
data = json.load(open('/root/oncocooler_analysis.json'))
assert set(data['assumptions'].keys()) == {'dispatches_per_year_10_day','dispatches_per_year_20_day','days_per_dispatch_10_day','days_per_dispatch_20_day','switch_threshold_usd','site_override_rule'}, f'Bad assumptions keys: {set(data[\"assumptions\"].keys())}'
expected_prog_keys = {'program_code','program_name','active_sites','acquisition_cost_per_1000_units_usd','units_per_day','cooler_type','cooler_cost_usd','payment_per_dispatch_per_site_usd','annual_drug_cost_10_day_usd','annual_drug_cost_20_day_usd','annual_cooler_cost_10_day_usd','annual_cooler_cost_20_day_usd','annual_revenue_10_day_usd','annual_revenue_20_day_usd','annual_margin_10_day_usd','annual_margin_20_day_usd','annual_margin_difference_20_minus_10_usd'}
for p in data['programs']:
    assert set(p.keys()) == expected_prog_keys, f'Bad program keys for {p.get(\"program_code\",\"?\")}: extra={set(p.keys())-expected_prog_keys}, missing={expected_prog_keys-set(p.keys())}'
print('Schema OK, programs:', len(data['programs']))
print('Decision:', data['recommendation']['decision'])
print('Totals:', json.dumps(data['totals'], indent=2))
"
```

6. **Check the summary file**:
```bash
cat /root/oncocooler_summary.md
wc -l /root/oncocooler_summary.md
```

7. **Run the verifier tests**:
```bash
cd / && python -m pytest tests/test_outputs.py -v 2>&1 | head -80
```

If any test fails, read the error carefully, fix the issue in `solve.py`, re-run, and re-test. Pay special attention to:
- Key name mismatches (the previous run had wrong key names like `annual_cooler_cost_20_day` instead of `annual_cooler_cost_20_day_usd`)
- Missing `assumptions` keys (`site_override_rule`, `switch_threshold_usd`)
- Extra fields in program objects
- Cooler cost formula (check if it should be per-dispatch-total or per-dispatch-per-site by examining the data and test expectations)

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