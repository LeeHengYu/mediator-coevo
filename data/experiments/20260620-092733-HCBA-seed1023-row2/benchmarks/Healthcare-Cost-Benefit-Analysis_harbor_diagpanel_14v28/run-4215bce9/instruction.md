# Task Instruction

Execute the following steps precisely to produce `/root/diagpanel_policy_report.json` and `/root/diagpanel_policy_summary.md`.

## Step 0: Inspect all input files

Read and print the contents of every input file so you understand their exact structure:
```
cat /root/panel_manifest.json
cat /root/shipper_cost.csv
cat /root/contract_terms.csv
cat /root/network_adjustments.csv
cat /root/lab_capacity_overrides.csv
cat /root/holdouts.json
cat /root/report_template.json
```

## Step 1: Write and run a Python script

After inspecting the files, write a single Python script `/root/solve.py` that does everything below, then run it with `python3 /root/solve.py`.

### 1a. Load data
- Load `panel_manifest.json` (list of panel objects).
- Load `shipper_cost.csv`, `contract_terms.csv`, `network_adjustments.csv`, `lab_capacity_overrides.csv` as CSV (use csv.DictReader).
- Load `holdouts.json` (list of holdout objects).
- Load `report_template.json`.

### 1b. Determine retained panels
- From `panel_manifest.json`, keep only panels where `analysis_mode == "review"`.
- Build a set of excluded panel_codes from `holdouts.json` where `holdout_state == "exclude"`.
- Remove any panel whose `panel_code` is in the excluded set.
- The remaining panels are "retained".

### 1c. Resolve contract terms
For each retained panel:
- Collect all rows from `contract_terms.csv` where `panel_ref` matches either the panel's `panel_name` OR any entry in its `alias_labels` list.
- Keep only rows where `status_flag == "current"`.
- If multiple rows remain, pick the one with the latest `effective_week` (compare as strings if they are ISO dates, or parse them).
- Extract `base_payment_per_run_per_lab_usd` (convert to float).

### 1d. Network adjustment
For each retained panel:
- Look up `network_tier` from the panel manifest.
- Find the matching row in `network_adjustments.csv` by `network_tier`.
- If found, use `network_adjustment_per_run_per_lab_usd` (float). If not found, use `0.0`.
- Compute `total_payment_per_run_per_lab_usd = base_payment + network_adjustment`.

### 1e. Shipper cost
For each retained panel:
- Look up `shipper_class` from the panel manifest.
- Find the matching row in `shipper_cost.csv` by `shipper_class`.
- Use `shipper_cost_usd` (float).

### 1f. Active labs (CRITICAL — previous failure was here)
For each retained panel:
- From `lab_capacity_overrides.csv`, find all rows where `panel_code` matches.
- Keep only rows where `approval == "approved"`.
- Among those, discard any row where `rev` is blank/empty OR `active_labs` is blank/empty.
- If valid rows remain, pick the one with the **highest numeric `rev`** (convert `rev` to int or float for comparison). Use its `active_labs` (convert to int).
- If NO valid rows remain, use `default_active_labs` from the panel manifest (convert to int).

### 1g. Compute per-panel financials
For each retained panel:
- `runs_14 = 26`, `runs_28 = 13`
- `tests_14 = tests_per_lab_per_run_14_day` from manifest (int)
- `tests_28 = tests_per_lab_per_run_28_day` from manifest (int)
- `reagent_cost_per_1000 = reagent_cost_per_1000_tests_usd` from manifest (float)
- `annual_revenue_14 = total_payment_per_run_per_lab * active_labs * 26`
- `annual_revenue_28 = total_payment_per_run_per_lab * active_labs * 13`
- `annual_reagent_cost_14 = reagent_cost_per_1000 * active_labs * tests_14 * 26 / 1000`
- `annual_reagent_cost_28 = reagent_cost_per_1000 * active_labs * tests_28 * 13 / 1000`
- `annual_shipper_cost_14 = shipper_cost_usd * 26`
- `annual_shipper_cost_28 = shipper_cost_usd * 13`
- `annual_margin_14 = annual_revenue_14 - annual_reagent_cost_14 - annual_shipper_cost_14`
- `annual_margin_28 = annual_revenue_28 - annual_reagent_cost_28 - annual_shipper_cost_28`
- `difference = annual_margin_28 - annual_margin_14`
- Round ALL currency values to 2 decimal places.

### 1h. Build panels list
Sort retained panels by `panel_code` ascending (string sort). Build each panel object with ALL fields from the schema:
```
panel_code, panel_name, active_labs, reagent_cost_per_1000_tests_usd,
network_tier, network_adjustment_per_run_per_lab_usd, shipper_class,
shipper_cost_usd, base_payment_per_run_per_lab_usd,
total_payment_per_run_per_lab_usd, tests_per_lab_per_run_14_day,
tests_per_lab_per_run_28_day, annual_reagent_cost_14_day_usd,
annual_reagent_cost_28_day_usd, annual_shipper_cost_14_day_usd,
annual_shipper_cost_28_day_usd, annual_revenue_14_day_usd,
annual_revenue_28_day_usd, annual_margin_14_day_usd,
annual_margin_28_day_usd, annual_margin_difference_28_minus_14_usd
```

### 1i. Totals and recommendation
- `total_14 = sum of all annual_margin_14_day_usd`
- `total_28 = sum of all annual_margin_28_day_usd`
- `total_diff = total_28 - total_14`
- `abs_diff = abs(total_diff)`
- Round all to 2 decimals.
- If `abs_diff < 6000`: decision = `adopt_28_day`, else `keep_14_day`.
- Justification: a brief string explaining the decision referencing the threshold.

### 1j. Assemble JSON report
- Start from `report_template.json`. Preserve `metadata` and `audit_notes` EXACTLY as they appear in the template.
- Add `analysis` with `assumptions` (use the exact keys/values from the schema), `panels`, `totals`, `recommendation`.
- Write to `/root/diagpanel_policy_report.json` with `json.dump(..., indent=2)`.

### 1k. Write markdown summary
Write `/root/diagpanel_policy_summary.md` with 4-8 non-empty lines. Include:
- Total 14-day margin formatted with commas: e.g., `1,090.76`
- Total 28-day margin formatted with commas
- Absolute difference formatted with commas
- The exact decision slug (`adopt_28_day` or `keep_14_day`)

Use Python's `"{:,.2f}".format(value)` for comma formatting.

## Step 2: Validate

After running the script:
1. `cat /root/diagpanel_policy_report.json` and verify structure.
2. `cat /root/diagpanel_policy_summary.md` and verify content.
3. If there is a test file (e.g., `test_output.py`), run `cd /root && python -m pytest test_output.py -v` and check results.
4. If tests fail, read the error messages carefully, fix the script, and re-run.

## CRITICAL WARNINGS from previous failure:
- The `rev` field in `lab_capacity_overrides.csv` must be compared NUMERICALLY, not as strings. Convert to int/float.
- Contract matching via `alias_labels`: the manifest field is a list — check if `panel_ref` is IN that list.
- Shipper cost is per-shipment, and annual shipper cost = `shipper_cost_usd * runs_per_year` (NOT multiplied by active_labs).
- The markdown summary MUST use comma-formatted numbers (e.g., `26,185.96` not `26185.96`).
- Double-check that `active_labs` values are correct by printing them during debugging.

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