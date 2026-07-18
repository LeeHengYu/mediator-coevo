# Task Instruction

Execute the following steps in order:

## Step 1: Inspect all input files

Read and display the contents of each input file:
```
cat /root/panel_manifest.json
cat /root/shipper_cost.csv
cat /root/contract_terms.csv
cat /root/network_adjustments.csv
cat /root/lab_capacity_overrides.csv
cat /root/holdouts.json
cat /root/report_template.json
```

## Step 2: Write and run the solution script

Create `/root/solve.py` with the following logic:

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
    net_adj_rows = list(csv.DictReader(f))

with open('/root/lab_capacity_overrides.csv') as f:
    lab_override_rows = list(csv.DictReader(f))

with open('/root/holdouts.json') as f:
    holdouts = json.load(f)

with open('/root/report_template.json') as f:
    template = json.load(f)

# Build lookup: shipper_class -> shipper_cost_usd
shipper_lookup = {}
for row in shipper_rows:
    shipper_lookup[row['shipper_class'].strip()] = float(row['shipper_cost_usd'])

# Build lookup: network_tier -> network_adjustment_per_run_per_lab_usd
net_adj_lookup = {}
for row in net_adj_rows:
    net_adj_lookup[row['network_tier'].strip()] = float(row['network_adjustment_per_run_per_lab_usd'])

# Build holdout exclusion set
exclude_set = set()
if isinstance(holdouts, list):
    for h in holdouts:
        if str(h.get('holdout_state', '')).strip().lower() == 'exclude':
            exclude_set.add(h.get('panel_code', '').strip())
elif isinstance(holdouts, dict):
    for k, v in holdouts.items():
        if isinstance(v, dict) and str(v.get('holdout_state', '')).strip().lower() == 'exclude':
            exclude_set.add(k)

# Filter panels: analysis_mode == 'review' and not in exclude set
panels_list = manifest if isinstance(manifest, list) else manifest.get('panels', [])
retained = []
for p in panels_list:
    if str(p.get('analysis_mode', '')).strip().lower() == 'review':
        if p['panel_code'].strip() not in exclude_set:
            retained.append(p)

# Build contract lookup: panel_ref -> list of contract rows with status_flag=current
# Match panel_ref to panel_name or any alias_labels entry (case-insensitive)
def build_panel_ref_map(retained_panels):
    """Map lowercase panel_ref -> panel_code"""
    ref_map = {}  # lowercase ref -> list of panel_codes
    for p in retained_panels:
        pname = p['panel_name'].strip().lower()
        pcode = p['panel_code'].strip()
        if pname not in ref_map:
            ref_map[pname] = pcode
        aliases = p.get('alias_labels', [])
        if isinstance(aliases, str):
            aliases = [aliases]
        for a in aliases:
            a_lower = a.strip().lower()
            if a_lower not in ref_map:
                ref_map[a_lower] = pcode
    return ref_map

ref_map = build_panel_ref_map(retained)

# For each retained panel, find the best contract row
current_contracts = [r for r in contract_rows if r.get('status_flag', '').strip().lower() == 'current']

# Group current contracts by panel_code
from collections import defaultdict
contracts_by_panel = defaultdict(list)
for cr in current_contracts:
    pr = cr['panel_ref'].strip().lower()
    if pr in ref_map:
        contracts_by_panel[ref_map[pr]].append(cr)

# Pick latest effective_week for each panel
best_contract = {}
for pcode, clist in contracts_by_panel.items():
    best = max(clist, key=lambda x: x.get('effective_week', '').strip())
    best_contract[pcode] = best

# Build lab override lookup: panel_code -> active_labs
# Only approved rows with non-empty rev and non-empty active_labs
# Keep highest numeric rev per panel_code
lab_overrides = {}
for row in lab_override_rows:
    if str(row.get('approval', '')).strip().lower() != 'approved':
        continue
    rev_val = str(row.get('rev', '')).strip()
    active_labs_val = str(row.get('active_labs', '')).strip()
    if rev_val == '' or active_labs_val == '':
        continue
    try:
        rev_num = float(rev_val)
    except ValueError:
        continue
    try:
        labs_num = int(float(active_labs_val))
    except ValueError:
        continue
    pc = row.get('panel_code', '').strip()
    if pc not in lab_overrides or rev_num > lab_overrides[pc][0]:
        lab_overrides[pc] = (rev_num, labs_num)

# Process each retained panel
panel_results = []
for p in retained:
    pcode = p['panel_code'].strip()
    pname = p['panel_name'].strip()
    
    # Active labs
    if pcode in lab_overrides:
        active_labs = lab_overrides[pcode][1]
    else:
        active_labs = int(p['default_active_labs'])
    
    reagent_cost_per_1000 = float(p['reagent_cost_per_1000_tests_usd'])
    network_tier = str(p.get('network_tier', '')).strip()
    net_adj = net_adj_lookup.get(network_tier, 0.0)
    shipper_class = str(p.get('shipper_class', '')).strip()
    shipper_cost = shipper_lookup.get(shipper_class, 0.0)
    
    # Contract
    if pcode in best_contract:
        base_payment = float(best_contract[pcode]['base_payment_per_run_per_lab_usd'])
    else:
        base_payment = 0.0
    
    total_payment = base_payment + net_adj
    
    tests_14 = int(p['tests_per_lab_per_run_14_day'])
    tests_28 = int(p['tests_per_lab_per_run_28_day'])
    
    runs_14 = 26
    runs_28 = 13
    
    # Annual revenue
    rev_14 = round(total_payment * active_labs * runs_14, 2)
    rev_28 = round(total_payment * active_labs * runs_28, 2)
    
    # Annual reagent cost
    reagent_14 = round(reagent_cost_per_1000 * active_labs * tests_14 * runs_14 / 1000, 2)
    reagent_28 = round(reagent_cost_per_1000 * active_labs * tests_28 * runs_28 / 1000, 2)
    
    # Annual shipper cost: shipper_cost * runs_per_year (per the formula pattern from similar tasks)
    # Note: shipper cost is per shipment, and we have runs_per_year shipments
    # Based on similar successful tasks, shipper cost = shipper_cost_usd * active_labs * runs_per_year
    shipper_14 = round(shipper_cost * active_labs * runs_14, 2)
    shipper_28 = round(shipper_cost * active_labs * runs_28, 2)
    
    # Annual margin
    margin_14 = round(rev_14 - reagent_14 - shipper_14, 2)
    margin_28 = round(rev_28 - reagent_28 - shipper_28, 2)
    
    diff = round(margin_28 - margin_14, 2)
    
    panel_results.append({
        'panel_code': pcode,
        'panel_name': pname,
        'active_labs': active_labs,
        'reagent_cost_per_1000_tests_usd': reagent_cost_per_1000,
        'network_tier': network_tier,
        'network_adjustment_per_run_per_lab_usd': net_adj,
        'shipper_class': shipper_class,
        'shipper_cost_usd': shipper_cost,
        'base_payment_per_run_per_lab_usd': base_payment,
        'total_payment_per_run_per_lab_usd': total_payment,
        'tests_per_lab_per_run_14_day': tests_14,
        'tests_per_lab_per_run_28_day': tests_28,
        'annual_reagent_cost_14_day_usd': reagent_14,
        'annual_reagent_cost_28_day_usd': reagent_28,
        'annual_shipper_cost_14_day_usd': shipper_14,
        'annual_shipper_cost_28_day_usd': shipper_28,
        'annual_revenue_14_day_usd': rev_14,
        'annual_revenue_28_day_usd': rev_28,
        'annual_margin_14_day_usd': margin_14,
        'annual_margin_28_day_usd': margin_28,
        'annual_margin_difference_28_minus_14_usd': diff
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
    justification = f'Absolute total margin difference ${abs_diff:,.2f} is below the $6,000.00 threshold; adopting 28-day cadence.'
else:
    decision = 'keep_14_day'
    justification = f'Absolute total margin difference ${abs_diff:,.2f} meets or exceeds the $6,000.00 threshold; keeping 14-day cadence.'

# Build output JSON
output = {
    'metadata': template['metadata'],
    'audit_notes': template['audit_notes'],
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

# Write summary markdown with comma-formatted currency
with open('/root/diagpanel_policy_summary.md', 'w') as f:
    f.write('# Diagnostic Panel Policy Summary\n')
    f.write(f'\n')
    f.write(f'Total 14-day annual margin: ${total_margin_14:,.2f}\n')
    f.write(f'Total 28-day annual margin: ${total_margin_28:,.2f}\n')
    f.write(f'Absolute margin difference: ${abs_diff:,.2f}\n')
    f.write(f'Decision: {decision}\n')

print('Done. Output files created.')
print(f'Retained panels: {len(panel_results)}')
print(f'Total margin 14-day: {total_margin_14}')
print(f'Total margin 28-day: {total_margin_28}')
print(f'Total diff: {total_diff}')
print(f'Abs diff: {abs_diff}')
print(f'Decision: {decision}')
```

Run it:
```
cd /root && python solve.py
```

## Step 3: Validate outputs

```
cat /root/diagpanel_policy_report.json
cat /root/diagpanel_policy_summary.md
```

Verify:
- JSON is valid and parseable
- `metadata` and `audit_notes` match the template exactly
- `analysis.assumptions` has all 6 keys with correct values
- `analysis.panels` is sorted by `panel_code` ascending
- Each panel has all required fields with numeric values rounded to 2 decimals
- `totals` has all 4 keys
- `recommendation.decision` is one of `adopt_28_day` or `keep_14_day`
- Summary has 4-8 non-empty lines, includes total 14-day margin, total 28-day margin, absolute difference, and the decision slug
- Currency values in the summary use comma separators for thousands (e.g., `$42,908.83` not `$42908.83`)

## Step 4: If anything looks wrong after inspecting the input files

If the data reveals that shipper cost should NOT be multiplied by active_labs (e.g., if shipper_cost.csv values are already annual or per-run totals rather than per-lab-per-run), adjust the formula. The task instructions say `annual_shipper_cost` but don't give an explicit formula for it. Look at the magnitude of shipper_cost_usd values relative to other costs to determine the correct interpretation. If shipper costs are small per-unit values, multiply by active_labs * runs_per_year. If they are already large aggregate values, they may just need to be multiplied by runs_per_year.

IMPORTANT: After reading the input files, if the shipper cost formula needs adjustment, update solve.py and re-run before finalizing.

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