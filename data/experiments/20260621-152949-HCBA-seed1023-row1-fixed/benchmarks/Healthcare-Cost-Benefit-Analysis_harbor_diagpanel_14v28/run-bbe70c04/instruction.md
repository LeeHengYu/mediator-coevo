# Task Instruction

Execute the following steps in order:

## Step 1: Inspect all input files

Read and display the contents of every input file:
```
cat /root/panel_manifest.json
cat /root/shipper_cost.csv
cat /root/contract_terms.csv
cat /root/network_adjustments.csv
cat /root/lab_capacity_overrides.csv
cat /root/holdouts.json
cat /root/report_template.json
```

Also check if there is a test file:
```
find /root -name 'test_output*' -o -name 'test_outputs*' 2>/dev/null
ls /tests/ 2>/dev/null
```

## Step 2: Write a Python script to compute everything

Create `/root/solve.py` that does the following:

1. **Load all input files** (JSON files with `json.load`, CSV files with `csv.DictReader`).

2. **Filter panels**: From `panel_manifest.json`, keep only panels where `analysis_mode == "review"`.

3. **Apply holdouts**: From `holdouts.json`, remove any panel whose `panel_code` matches a holdout entry with `holdout_state == "exclude"`.

4. **Resolve contract terms**:
   - For each retained panel, find rows in `contract_terms.csv` where `panel_ref` matches either `panel_name` or any entry in `alias_labels`.
   - Keep only rows where `status_flag == "current"`.
   - If multiple current rows match, keep the one with the latest `effective_week` (compare as strings if ISO-like, or parse dates).
   - Extract `base_payment_per_run_per_lab_usd` from the matched contract row.

5. **Network adjustments**:
   - Match panel's `network_tier` to `network_adjustments.csv`.
   - If no match, use `0.0`.
   - Extract `network_adjustment_per_run_per_lab_usd`.

6. **Active labs from lab_capacity_overrides.csv**:
   - Filter to rows where `approval == "approved"`.
   - Ignore rows where `rev` is blank/empty or `active_labs` is blank/empty.
   - Match by `panel_code`.
   - If multiple valid rows, keep the one with the highest numeric `rev`.
   - If no valid override row exists, use `default_active_labs` from `panel_manifest.json`.

7. **Shipper cost**: Match panel's `shipper_class` to `shipper_cost.csv` to get `shipper_cost_usd`.

8. **Compute per-panel metrics**:
   - `total_payment_per_run_per_lab_usd = base_payment_per_run_per_lab_usd + network_adjustment_per_run_per_lab_usd`
   - 14-day: `runs_per_year = 26`, use `tests_per_lab_per_run_14_day`
   - 28-day: `runs_per_year = 13`, use `tests_per_lab_per_run_28_day`
   - `annual_revenue = total_payment_per_run_per_lab_usd * active_labs * runs_per_year`
   - `annual_reagent_cost = reagent_cost_per_1000_tests_usd * active_labs * tests_per_lab_per_run * runs_per_year / 1000`
   - `annual_shipper_cost = shipper_cost_usd * runs_per_year`  (NOTE: shipper cost is per shipment, so it's `shipper_cost_usd * runs_per_year` — but CAREFULLY CHECK the data to see if shipper_cost is already annual or per-run. The formula says `annual_margin = annual_revenue - annual_reagent_cost - annual_shipper_cost`, and the shipper cost column is named `shipper_cost_usd`. Look at the actual data values and the schema to determine if shipper cost needs to be multiplied by runs_per_year or if it's already annual. The task says "Annual margin formula: annual_revenue - annual_reagent_cost - annual_shipper_cost" — the shipper_cost_usd from the CSV is likely per-shipment, so annual_shipper_cost_14_day = shipper_cost_usd * 26, annual_shipper_cost_28_day = shipper_cost_usd * 13. BUT VERIFY: look at the output schema which has `annual_shipper_cost_14_day_usd` and `annual_shipper_cost_28_day_usd` as separate fields, confirming shipper cost differs by cadence, so it must be multiplied by runs_per_year.)
   - `annual_margin = annual_revenue - annual_reagent_cost - annual_shipper_cost`
   - `annual_margin_difference_28_minus_14 = annual_margin_28_day - annual_margin_14_day`

9. **Round all currency values to 2 decimal places.**

10. **Sort panels by `panel_code` ascending.**

11. **Compute totals**:
    - Sum all `annual_margin_14_day_usd` across panels.
    - Sum all `annual_margin_28_day_usd` across panels.
    - `total_difference = total_28 - total_14`
    - `absolute_total_margin_difference_usd = abs(total_difference)` (round to 2 decimals)

12. **Decision**:
    - If `abs(total_difference) < 6000`: `adopt_28_day`
    - Otherwise: `keep_14_day`
    - Write a justification string that includes the absolute difference and threshold.

13. **Build the output JSON**:
    - Copy `metadata` and `audit_notes` from `report_template.json` exactly as-is.
    - Populate `analysis.assumptions` with the fixed values from the schema.
    - Populate `analysis.panels`, `analysis.totals`, `analysis.recommendation`.
    - Write to `/root/diagpanel_policy_report.json` with `json.dump(..., indent=2)`.

14. **Build the summary markdown** `/root/diagpanel_policy_summary.md`:
    - 4-8 non-empty lines.
    - Must include: total 14-day margin (USD), total 28-day margin (USD), absolute difference (USD), and the exact decision slug (`adopt_28_day` or `keep_14_day`).

## Step 3: Run the script
```
cd /root && python solve.py
```

## Step 4: Validate outputs
```
cat /root/diagpanel_policy_report.json
cat /root/diagpanel_policy_summary.md
python -c "import json; d=json.load(open('/root/diagpanel_policy_report.json')); print('panels:', len(d['analysis']['panels'])); print('decision:', d['analysis']['recommendation']['decision']); print('totals:', d['analysis']['totals'])"
```

## Step 5: Run tests if they exist
```
cd / && python -m pytest tests/test_output*.py -v 2>/dev/null || cd /root && python -m pytest test_output*.py -v 2>/dev/null || echo 'No test files found'
```

If tests fail, read the error messages carefully, fix the script, re-run, and re-validate. Pay special attention to:
- Numerical precision (round to 2 decimals)
- Field names matching exactly
- Sort order of panels
- Exact preservation of metadata and audit_notes from template
- The shipper cost calculation (per-run vs annual)
- Contract term matching (check alias_labels carefully — it may be a list or comma-separated string)
- The `active_labs` field should be an integer if the source data is integer
- Make sure `rev` comparison is numeric, not string

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
Task metadata: author_email=gpt54@example.com, author_name=GPT-5.4, category=financial-analysis, difficulty=medium, tags=[diagnostics, json, csv, template-update, decision-analysis].
Verifier config: timeout_sec=900.0.