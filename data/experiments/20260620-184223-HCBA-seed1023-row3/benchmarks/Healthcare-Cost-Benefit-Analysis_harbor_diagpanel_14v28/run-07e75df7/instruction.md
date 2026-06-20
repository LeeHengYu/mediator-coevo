# Task Instruction

Execute the following steps carefully and in order.

## 1. Inspect all input files

```bash
cat /root/panel_manifest.json
cat /root/shipper_cost.csv
cat /root/contract_terms.csv
cat /root/network_adjustments.csv
cat /root/lab_capacity_overrides.csv
cat /root/holdouts.json
cat /root/report_template.json
```

Read every file completely before writing any code.

## 2. Inspect the verifier

Look for any test or verifier script in the task directory:
```bash
find / -maxdepth 4 -name 'test_*.py' -o -name 'verify*.py' -o -name 'check*.py' 2>/dev/null | head -20
```
If found, read the verifier to understand exact assertions (field names, rounding, formatting expectations).

## 3. Write and run a Python script

Create `/root/solve.py` implementing the full logic below. Pay very close attention to the shipper cost model — the previous attempt got margins wildly wrong, likely due to shipper cost interpretation.

### Key logic:

**a) Load data:**
- Load `panel_manifest.json`, `holdouts.json`, `report_template.json` as JSON.
- Load `shipper_cost.csv`, `contract_terms.csv`, `network_adjustments.csv`, `lab_capacity_overrides.csv` as CSV.

**b) Filter panels:**
- Keep only panels where `analysis_mode == 'review'`.
- Exclude panels whose `panel_code` appears in `holdouts.json` with `holdout_state == 'exclude'`.
- These are the "retained" panels.

**c) Contract matching:**
- For each retained panel, find rows in `contract_terms.csv` where `panel_ref` matches either `panel_name` or any element in `alias_labels`.
- Keep only rows where `status_flag == 'current'`.
- If multiple, keep the one with the latest `effective_week` (parse as date or string sort — inspect the format first).
- Extract `base_payment_per_run_per_lab_usd`.

**d) Network adjustment:**
- Match panel's `network_tier` to `network_adjustments.csv`.
- If no match, use `0.0`.
- `total_payment_per_run_per_lab_usd = base_payment + network_adjustment`.

**e) Active labs (CAREFUL with override logic):**
- From `lab_capacity_overrides.csv`, keep rows where `approval == 'approved'`.
- Discard rows where `rev` is blank/empty or `active_labs` is blank/empty.
- Match by `panel_code`.
- If multiple valid rows for same `panel_code`, keep highest numeric `rev`.
- If no valid override row, use `default_active_labs` from `panel_manifest.json`.
- Cast `active_labs` to integer.

**f) Shipper cost:**
- Match panel's `shipper_class` to `shipper_cost.csv` to get `shipper_cost_usd`.
- IMPORTANT: Read the shipper_cost.csv carefully. The `shipper_cost_usd` is the cost per shipment/run. So:
  - `annual_shipper_cost_14_day_usd = shipper_cost_usd * 26`
  - `annual_shipper_cost_28_day_usd = shipper_cost_usd * 13`
- BUT: If the previous run produced wildly wrong margins, double-check whether `shipper_cost_usd` might be an annual figure or per-lab-per-run. Inspect the CSV values and cross-reference with expected magnitudes. If the values in the CSV are large (thousands), they might be annual. If small (tens/hundreds), they are per-run. Adjust accordingly. The task says "shipper cost uses shipper_cost_usd from shipper_cost.csv" — the formula only says `annual_shipper_cost` without specifying the per-run multiplication for shipper, unlike revenue and reagent which explicitly state `* runs_per_year`. Re-read the task: the annual margin formula is `annual_revenue - annual_reagent_cost - annual_shipper_cost`. The task does NOT give an explicit formula for annual_shipper_cost. So look at the output schema: it has `annual_shipper_cost_14_day_usd` and `annual_shipper_cost_28_day_usd` as separate fields. If these differ between 14-day and 28-day, then shipper cost must depend on runs_per_year. The most logical formula is: `annual_shipper_cost = shipper_cost_usd * active_labs * runs_per_year`. Try this formula. If shipper_cost_usd is per-shipment and there's one shipment per lab per run, this makes sense.

**g) Revenue:**
- `annual_revenue = total_payment_per_run_per_lab_usd * active_labs * runs_per_year`

**h) Reagent cost:**
- `annual_reagent_cost = reagent_cost_per_1000_tests_usd * active_labs * tests_per_lab_per_run * runs_per_year / 1000`

**i) Margin:**
- `annual_margin = annual_revenue - annual_reagent_cost - annual_shipper_cost`

**j) Per-panel difference:**
- `annual_margin_difference_28_minus_14 = annual_margin_28_day - annual_margin_14_day`

**k) Totals:**
- Sum all per-panel values for 14-day margin, 28-day margin, and difference.
- `absolute_total_margin_difference_usd = abs(total_difference)`

**l) Decision:**
- If `abs(total_difference) < 6000`: `adopt_28_day`
- Otherwise: `keep_14_day`

**m) Round all currency values to 2 decimal places.**

**n) Sort panels by `panel_code` ascending.**

**o) Build JSON output** using the exact schema from the task. Preserve `metadata` and `audit_notes` from `report_template.json` exactly.

**p) Write `/root/diagpanel_policy_report.json`** with `json.dumps(indent=2)`.

**q) Write `/root/diagpanel_policy_summary.md`** with 4-8 non-empty lines including:
- Total 14-day margin with comma formatting (e.g., `1,234.56` not `$1,234.56` — avoid `$` prefix per the cross-task failure artifact)
- Total 28-day margin
- Absolute difference
- The exact decision slug (`adopt_28_day` or `keep_14_day`)

## 4. Run the script
```bash
python3 /root/solve.py
```

## 5. Validate outputs
```bash
cat /root/diagpanel_policy_report.json
cat /root/diagpanel_policy_summary.md
python3 -c "import json; d=json.load(open('/root/diagpanel_policy_report.json')); print('panels:', len(d['analysis']['panels'])); print('decision:', d['analysis']['recommendation']['decision']); print('totals:', d['analysis']['totals'])"
```

## 6. Run the verifier if found
If you found a test file in step 2, run it:
```bash
cd /root && python3 -m pytest test_output.py -v 2>&1 | head -80
```
If any test fails, read the failure message carefully, fix the script, re-run, and re-validate. Pay special attention to:
- Numeric closeness assertions
- Field name mismatches
- Summary formatting requirements

## Critical reminders from previous failure:
- The previous run had `close(369.2, 6276.4)` and `close(26185.96, 1090.76)` — these are HUGE discrepancies suggesting the shipper cost formula was wrong (likely missing `* active_labs` or using wrong multiplier). Make sure `annual_shipper_cost = shipper_cost_usd * active_labs * runs_per_year` if that's what produces correct magnitudes.
- Double-check contract matching: print which contract row maps to which panel.
- Double-check active_labs resolution: print which override or default is used.
- Print intermediate values for debugging before writing final output.

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