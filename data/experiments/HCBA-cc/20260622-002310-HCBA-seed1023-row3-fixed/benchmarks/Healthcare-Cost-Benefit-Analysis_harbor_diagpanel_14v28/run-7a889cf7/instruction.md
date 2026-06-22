# Task Instruction

Execute the following steps in order to produce the two required output files.

## Step 1 – Inspect all input files

Read and display the full contents of every input file so you understand their exact structure:

```
cat /root/panel_manifest.json
cat /root/shipper_cost.csv
cat /root/contract_terms.csv
cat /root/network_adjustments.csv
cat /root/lab_capacity_overrides.csv
cat /root/holdouts.json
cat /root/report_template.json
```

## Step 2 – Write and run a Python script

Create `/root/solve.py` that does everything below, then run it with `python3 /root/solve.py`.

### 2a. Load data
- Load `panel_manifest.json` (list of panel objects).
- Load `shipper_cost.csv`, `contract_terms.csv`, `network_adjustments.csv`, `lab_capacity_overrides.csv` (use csv.DictReader).
- Load `holdouts.json` (list of holdout objects).
- Load `report_template.json`.

### 2b. Determine retained panels
- From `panel_manifest.json`, keep only panels where `analysis_mode == "review"`.
- Build a set of excluded panel_codes from `holdouts.json` where `holdout_state == "exclude"`.
- Remove any panel whose `panel_code` is in the excluded set.
- The remaining panels are the "retained" panels.

### 2c. Resolve contract terms
For each retained panel:
- Build a match set: {panel_name} ∪ set(alias_labels) (alias_labels may be a list in the manifest).
- From `contract_terms.csv`, find rows where `panel_ref` is in that match set AND `status_flag == "current"`.
- If multiple rows match, keep the one with the latest `effective_week` (compare as strings if ISO-formatted, or parse as dates).
- Extract `base_payment_per_run_per_lab_usd` (float).

### 2d. Network adjustment
For each retained panel, look up its `network_tier` in `network_adjustments.csv`. If found, use `network_adjustment_per_run_per_lab_usd` (float). If not found, use 0.0.

### 2e. Active labs
From `lab_capacity_overrides.csv`:
- Keep only rows where `approval == "approved"`.
- Drop rows where `rev` is blank/empty or `active_labs` is blank/empty.
- For each `panel_code`, keep the row with the highest numeric `rev`.
- For each retained panel, if an override row exists, use its `active_labs` (int). Otherwise use `default_active_labs` from the manifest (int).

### 2f. Shipper cost
For each retained panel, look up `shipper_class` (from manifest) in `shipper_cost.csv` to get `shipper_cost_usd` (float). This is the annual shipper cost for EACH cadence model (the shipper cost is a flat annual cost — but re-read the file; if the CSV has per-shipment cost, then annual_shipper_cost = shipper_cost_usd * runs_per_year). IMPORTANT: Check the CSV structure. If the cost appears to be per-shipment, multiply by runs_per_year. If it appears annual, use as-is. Look at column names and values to decide. The formulas say `annual_margin = annual_revenue - annual_reagent_cost - annual_shipper_cost`, and the schema has separate `annual_shipper_cost_14_day_usd` and `annual_shipper_cost_28_day_usd` fields — since these differ between cadences, the shipper cost must be per-shipment (or per-run), so: `annual_shipper_cost = shipper_cost_usd * runs_per_year`.

### 2g. Compute per-panel numbers
For each retained panel:
- `total_payment_per_run_per_lab_usd = base_payment + network_adjustment`
- 14-day model (runs_per_year=26):
  - `annual_revenue_14 = total_payment_per_run_per_lab * active_labs * 26`
  - `annual_reagent_cost_14 = reagent_cost_per_1000_tests * active_labs * tests_per_lab_per_run_14_day * 26 / 1000`
  - `annual_shipper_cost_14 = shipper_cost_usd * 26`
  - `annual_margin_14 = annual_revenue_14 - annual_reagent_cost_14 - annual_shipper_cost_14`
- 28-day model (runs_per_year=13): same formulas with 13 and tests_per_lab_per_run_28_day.
- `annual_margin_difference = annual_margin_28 - annual_margin_14`
- Round ALL currency values to 2 decimal places.

### 2h. Totals and decision
- `total_annual_margin_14 = sum of all panels' annual_margin_14`
- `total_annual_margin_28 = sum of all panels' annual_margin_28`
- `total_difference = total_annual_margin_28 - total_annual_margin_14` (also = sum of per-panel differences)
- `absolute_total = abs(total_difference)`
- Round all to 2 decimals.
- If `absolute_total < 6000`: decision = `adopt_28_day`, else `keep_14_day`.
- Justification: a short string like `"Absolute margin difference ${absolute_total} is [below|at or above] the $6000 threshold; recommend {decision}."`

### 2i. Build output JSON
Sort panels by `panel_code` ascending.

Build the output dict exactly matching the schema:
- `metadata` and `audit_notes` copied verbatim from `report_template.json`.
- `analysis.assumptions` with exactly these keys and values:
  - `runs_per_year_14_day`: 26
  - `runs_per_year_28_day`: 13
  - `switch_threshold_usd`: 6000
  - `override_rule`: `"highest numeric approved rev with non-empty active_labs, else default_active_labs"`
  - `holdout_rule`: `"exclude holdout_state=exclude"`
  - `adjustment_rule`: `"missing network_tier adjustment defaults to 0.0"`
- `analysis.panels`: list of panel dicts with every field from the schema.
- `analysis.totals`: dict with the four total fields.
- `analysis.recommendation`: dict with `decision` and `justification`.

Write to `/root/diagpanel_policy_report.json` with `json.dump(..., indent=2)`.

### 2j. Build summary markdown
Write `/root/diagpanel_policy_summary.md` with 4-8 non-empty lines including:
- Total 14-day margin with comma-formatted USD (e.g., `$12,345.67` or `$-1,234.56` — use `"{:,.2f}".format(value)`).
- Total 28-day margin with comma-formatted USD.
- Absolute difference with comma-formatted USD.
- The exact decision slug (`adopt_28_day` or `keep_14_day`).

Example format:
```
# Diagnostics Panel Policy Summary

Total 14-day annual margin: $XX,XXX.XX
Total 28-day annual margin: $XX,XXX.XX
Absolute margin difference: $X,XXX.XX
Recommendation: adopt_28_day
```

## Step 3 – Validate outputs

After running the script:
1. `cat /root/diagpanel_policy_report.json` — verify it parses, has all required keys including `metadata`, `audit_notes`, `analysis.assumptions` (with exactly the 6 keys listed), `analysis.panels` sorted by panel_code, `analysis.totals`, and `analysis.recommendation` with both `decision` and `justification`.
2. `cat /root/diagpanel_policy_summary.md` — verify 4-8 non-empty lines, comma-formatted currency, and exact decision slug.
3. Run `python3 -c "import json; d=json.load(open('/root/diagpanel_policy_report.json')); print('OK')"` to confirm valid JSON.

If anything looks wrong, fix and re-run.

## Key pitfalls to avoid (from cross-task feedback)
- Do NOT omit any keys from `assumptions` — include exactly the 6 keys specified.
- Do NOT add extra keys to `assumptions` beyond those 6.
- Ensure `recommendation` contains `justification`.
- Use comma-formatted numbers in the summary markdown (e.g., `$-7,106.39` not `$-7106.39`).
- Preserve `metadata` and `audit_notes` from the template exactly — do not modify, reorder, or omit them.

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