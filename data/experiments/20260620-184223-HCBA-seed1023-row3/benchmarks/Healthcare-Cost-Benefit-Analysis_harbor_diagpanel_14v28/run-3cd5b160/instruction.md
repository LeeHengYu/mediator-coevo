# Task Instruction

Execute the following steps exactly:

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

## 2. Write and run a Python script

After inspecting the files, create `/root/solve.py` with the following logic:

```python
import json
import csv
import math

# Load all inputs
with open('/root/panel_manifest.json') as f:
    manifest = json.load(f)

with open('/root/shipper_cost.csv') as f:
    shipper_rows = list(csv.DictReader(f))

with open('/root/contract_terms.csv') as f:
    contract_rows = list(csv.DictReader(f))

with open('/root/network_adjustments.csv') as f:
    network_rows = list(csv.DictReader(f))

with open('/root/lab_capacity_overrides.csv') as f:
    lab_override_rows = list(csv.DictReader(f))

with open('/root/holdouts.json') as f:
    holdouts = json.load(f)

with open('/root/report_template.json') as f:
    template = json.load(f)

# Build lookup: shipper_class -> shipper_cost_usd
shipper_lookup = {}
for row in shipper_rows:
    sc = row.get('shipper_class', '').strip()
    cost = row.get('shipper_cost_usd', '').strip()
    if sc and cost:
        shipper_lookup[sc] = float(cost)

# Build lookup: network_tier -> network_adjustment_per_run_per_lab_usd
network_lookup = {}
for row in network_rows:
    tier = row.get('network_tier', '').strip()
    adj = row.get('network_adjustment_per_run_per_lab_usd', '').strip()
    if tier and adj:
        network_lookup[tier] = float(adj)

# Build holdout exclusion set
exclude_codes = set()
if isinstance(holdouts, list):
    for h in holdouts:
        if h.get('holdout_state') == 'exclude':
            exclude_codes.add(h.get('panel_code', ''))
elif isinstance(holdouts, dict):
    for k, v in holdouts.items():
        if isinstance(v, dict) and v.get('holdout_state') == 'exclude':
            exclude_codes.add(v.get('panel_code', k))

# Filter manifest panels: analysis_mode == 'review' and not excluded
panels_list = manifest if isinstance(manifest, list) else manifest.get('panels', [])
retained = []
for p in panels_list:
    if p.get('analysis_mode') != 'review':
        continue
    if p.get('panel_code', '') in exclude_codes:
        continue
    retained.append(p)

print(f"Retained panels: {[p['panel_code'] for p in retained]}")

# Build contract lookup: for each retained panel, find matching current contract row
# Match contract_terms panel_ref to panel_name or any alias_labels entry
def get_contract_for_panel(panel):
    pname = panel.get('panel_name', '')
    aliases = panel.get('alias_labels', [])
    if isinstance(aliases, str):
        aliases = [aliases]
    match_names = set()
    match_names.add(pname)
    for a in aliases:
        match_names.add(a)
    
    candidates = []
    for row in contract_rows:
        if row.get('status_flag', '').strip() != 'current':
            continue
        pref = row.get('panel_ref', '').strip()
        if pref in match_names:
            candidates.append(row)
    
    if not candidates:
        return None
    
    # Keep latest effective_week
    best = max(candidates, key=lambda r: r.get('effective_week', ''))
    return best

# Build lab override lookup: panel_code -> active_labs
def get_active_labs(panel):
    pcode = panel.get('panel_code', '')
    candidates = []
    for row in lab_override_rows:
        if row.get('panel_code', '').strip() != pcode:
            continue
        if row.get('approval', '').strip() != 'approved':
            continue
        rev_val = row.get('rev', '').strip()
        labs_val = row.get('active_labs', '').strip()
        if rev_val == '' or labs_val == '':
            continue
        candidates.append(row)
    
    if not candidates:
        return panel.get('default_active_labs')
    
    best = max(candidates, key=lambda r: float(r['rev'].strip()))
    return int(best['active_labs'].strip())

# Process each retained panel
panel_results = []
for p in retained:
    pcode = p['panel_code']
    pname = p['panel_name']
    
    contract = get_contract_for_panel(p)
    if contract is None:
        print(f"WARNING: No contract found for {pcode} ({pname})")
        continue
    
    base_payment = float(contract.get('base_payment_per_run_per_lab_usd', 0))
    
    network_tier = p.get('network_tier', '')
    net_adj = network_lookup.get(network_tier, 0.0)
    
    shipper_class = p.get('shipper_class', '')
    shipper_cost = shipper_lookup.get(shipper_class, 0.0)
    
    active_labs = get_active_labs(p)
    if active_labs is None:
        active_labs = p.get('default_active_labs', 0)
    active_labs = int(active_labs)
    
    reagent_cost_per_1000 = float(p.get('reagent_cost_per_1000_tests_usd', 0))
    tests_14 = int(p.get('tests_per_lab_per_run_14_day', 0))
    tests_28 = int(p.get('tests_per_lab_per_run_28_day', 0))
    
    total_payment = base_payment + net_adj
    
    # 14-day
    runs_14 = 26
    annual_revenue_14 = total_payment * active_labs * runs_14
    annual_reagent_14 = reagent_cost_per_1000 * active_labs * tests_14 * runs_14 / 1000
    annual_shipper_14 = shipper_cost  # per year? Need to check - likely shipper_cost * runs_per_year
    # Actually re-reading: annual_shipper_cost is not explicitly defined in formulas.
    # The instructions say: annual margin = annual_revenue - annual_reagent_cost - annual_shipper_cost
    # shipper_cost_usd from shipper_cost.csv is likely per-shipment, so annual = shipper_cost * runs_per_year
    # But let me check if it's already annual. The formula doesn't specify multiplication.
    # Given the schema has annual_shipper_cost_14_day_usd and annual_shipper_cost_28_day_usd as separate fields,
    # and 14-day has more runs, it's likely shipper_cost * runs_per_year
    annual_shipper_14 = shipper_cost * runs_14
    annual_margin_14 = annual_revenue_14 - annual_reagent_14 - annual_shipper_14
    
    # 28-day
    runs_28 = 13
    annual_revenue_28 = total_payment * active_labs * runs_28
    annual_reagent_28 = reagent_cost_per_1000 * active_labs * tests_28 * runs_28 / 1000
    annual_shipper_28 = shipper_cost * runs_28
    annual_margin_28 = annual_revenue_28 - annual_reagent_28 - annual_shipper_28
    
    diff = annual_margin_28 - annual_margin_14
    
    panel_results.append({
        'panel_code': pcode,
        'panel_name': pname,
        'active_labs': active_labs,
        'reagent_cost_per_1000_tests_usd': round(reagent_cost_per_1000, 2),
        'network_tier': network_tier,
        'network_adjustment_per_run_per_lab_usd': round(net_adj, 2),
        'shipper_class': shipper_class,
        'shipper_cost_usd': round(shipper_cost, 2),
        'base_payment_per_run_per_lab_usd': round(base_payment, 2),
        'total_payment_per_run_per_lab_usd': round(total_payment, 2),
        'tests_per_lab_per_run_14_day': tests_14,
        'tests_per_lab_per_run_28_day': tests_28,
        'annual_reagent_cost_14_day_usd': round(annual_reagent_14, 2),
        'annual_reagent_cost_28_day_usd': round(annual_reagent_28, 2),
        'annual_shipper_cost_14_day_usd': round(annual_shipper_14, 2),
        'annual_shipper_cost_28_day_usd': round(annual_shipper_28, 2),
        'annual_revenue_14_day_usd': round(annual_revenue_14, 2),
        'annual_revenue_28_day_usd': round(annual_revenue_28, 2),
        'annual_margin_14_day_usd': round(annual_margin_14, 2),
        'annual_margin_28_day_usd': round(annual_margin_28, 2),
        'annual_margin_difference_28_minus_14_usd': round(diff, 2)
    })

# Sort by panel_code ascending
panel_results.sort(key=lambda x: x['panel_code'])

# Totals
total_margin_14 = round(sum(p['annual_margin_14_day_usd'] for p in panel_results), 2)
total_margin_28 = round(sum(p['annual_margin_28_day_usd'] for p in panel_results), 2)
total_diff = round(sum(p['annual_margin_difference_28_minus_14_usd'] for p in panel_results), 2)
abs_diff = round(abs(total_diff), 2)

# Decision
if abs_diff < 6000:
    decision = 'adopt_28_day'
    justification = f'Absolute total margin difference of {abs_diff} USD is below the 6000 USD threshold, recommending adoption of the 28-day cadence.'
else:
    decision = 'keep_14_day'
    justification = f'Absolute total margin difference of {abs_diff} USD meets or exceeds the 6000 USD threshold, recommending retention of the 14-day cadence.'

# Build output
output = {
    'metadata': template.get('metadata', {}),
    'audit_notes': template.get('audit_notes', []),
    'analysis': {
        'assumptions': {
            'runs_per_year_14_day': 26,
            'runs_per_year_28_day': 13,
            'switch_threshold_usd': 6000,
            'override_rule': 'highest numeric approved rev with non-empty active_labs, else default_active_labs',
            'holdout_rule': 'exclude holdout_state=exclude',
            'adjustment_rule': 'missing network_tier adjustment defaults to 0.0'
        },
        'panels': panel_results,
        'totals': {
            'total_annual_margin_14_day_usd': total_margin_14,
            'total_annual_margin_28_day_usd': total_margin_28,
            'total_annual_margin_difference_28_minus_14_usd': total_diff,
            'absolute_total_margin_difference_usd': abs_diff
        },
        'recommendation': {
            'decision': decision,
            'justification': justification
        }
    }
}

with open('/root/diagpanel_policy_report.json', 'w') as f:
    json.dump(output, f, indent=2)

print("JSON report written.")
print(f"Total 14-day margin: {total_margin_14}")
print(f"Total 28-day margin: {total_margin_28}")
print(f"Total difference: {total_diff}")
print(f"Absolute difference: {abs_diff}")
print(f"Decision: {decision}")

# Write summary markdown
def fmt(val):
    # Format as comma-separated number with 2 decimals, no $ prefix
    if val < 0:
        return f"-{abs(val):,.2f}"
    return f"{val:,.2f}"

with open('/root/diagpanel_policy_summary.md', 'w') as f:
    f.write(f"# Diagnostics Panel Policy Summary\n")
    f.write(f"\n")
    f.write(f"Total 14-day annual margin: {fmt(total_margin_14)} USD\n")
    f.write(f"Total 28-day annual margin: {fmt(total_margin_28)} USD\n")
    f.write(f"Absolute margin difference: {fmt(abs_diff)} USD\n")
    f.write(f"Recommendation: {decision}\n")

print("Summary written.")
```

Run:
```bash
python3 /root/solve.py
```

## 3. Validate outputs

```bash
cat /root/diagpanel_policy_report.json
cat /root/diagpanel_policy_summary.md
```

After seeing the outputs, verify:
- `metadata` and `audit_notes` match the template exactly
- `assumptions` has all 6 required keys with correct values
- Each panel has ALL required fields from the schema
- Panels are sorted by `panel_code` ascending
- Currency values are rounded to 2 decimals
- Summary has 4-8 non-empty lines, includes the totals and the exact decision slug
- Summary does NOT use `$` prefix on numbers (use raw comma-formatted values with USD suffix)

## 4. If any issues found after inspection

Fix the script and re-run. Pay special attention to:
- The `active_labs` resolution (override vs default)
- Contract matching (panel_ref matching panel_name OR alias_labels)
- The shipper cost being multiplied by runs_per_year for annual cost
- Holdout exclusion logic
- The exact schema field names and assumptions keys

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