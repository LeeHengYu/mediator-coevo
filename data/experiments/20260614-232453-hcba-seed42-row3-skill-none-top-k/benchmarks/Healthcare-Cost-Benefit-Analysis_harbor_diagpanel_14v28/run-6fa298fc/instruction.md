# Task Instruction

Execute the following steps exactly, in order.

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

## 2. Write and run the Python solution

Create `/root/solve.py` with the following logic. Read all input files carefully before writing the script.

```python
import json
import csv
import math

# ── Load inputs ──
with open('/root/panel_manifest.json') as f:
    manifest = json.load(f)

with open('/root/shipper_cost.csv') as f:
    shipper_rows = list(csv.DictReader(f))

with open('/root/contract_terms.csv') as f:
    contract_rows = list(csv.DictReader(f))

with open('/root/network_adjustments.csv') as f:
    net_adj_rows = list(csv.DictReader(f))

with open('/root/lab_capacity_overrides.csv') as f:
    lab_cap_rows = list(csv.DictReader(f))

with open('/root/holdouts.json') as f:
    holdouts = json.load(f)

with open('/root/report_template.json') as f:
    template = json.load(f)

# ── Build lookup maps ──

# Shipper cost map: shipper_class -> cost
shipper_map = {}
for r in shipper_rows:
    shipper_map[r['shipper_class'].strip()] = float(r['shipper_cost_usd'].strip())

# Network adjustment map: network_tier -> adjustment
net_adj_map = {}
for r in net_adj_rows:
    net_adj_map[r['network_tier'].strip()] = float(r['network_adjustment_per_run_per_lab_usd'].strip())

# Holdout exclusion set
exclude_set = set()
if isinstance(holdouts, list):
    for h in holdouts:
        if h.get('holdout_state') == 'exclude':
            exclude_set.add(h.get('panel_code', '').strip())
elif isinstance(holdouts, dict):
    for k, v in holdouts.items():
        if isinstance(v, dict) and v.get('holdout_state') == 'exclude':
            exclude_set.add(k.strip())
        elif isinstance(v, list):
            for h in v:
                if h.get('holdout_state') == 'exclude':
                    exclude_set.add(h.get('panel_code', '').strip())

# ── Identify retained panels ──
panels_list = manifest if isinstance(manifest, list) else manifest.get('panels', [])

retained = []
for p in panels_list:
    if p.get('analysis_mode') != 'review':
        continue
    pc = p.get('panel_code', '').strip()
    if pc in exclude_set:
        continue
    retained.append(p)

# ── Resolve contract terms ──
# Build name/alias -> panel_code mapping
name_to_panel = {}
for p in retained:
    pc = p['panel_code'].strip()
    pn = p.get('panel_name', '').strip()
    name_to_panel[pn] = pc
    for alias in p.get('alias_labels', []):
        name_to_panel[alias.strip()] = pc
    name_to_panel[pc] = pc  # also allow direct match

# Filter current contracts and map to retained panels
contract_map = {}  # panel_code -> best contract row
for cr in contract_rows:
    if cr.get('status_flag', '').strip() != 'current':
        continue
    panel_ref = cr.get('panel_ref', '').strip()
    pc = name_to_panel.get(panel_ref)
    if pc is None:
        continue
    ew = cr.get('effective_week', '').strip()
    if pc not in contract_map:
        contract_map[pc] = cr
    else:
        existing_ew = contract_map[pc].get('effective_week', '').strip()
        if ew > existing_ew:
            contract_map[pc] = cr

# ── Resolve lab capacity overrides ──
# Filter approved rows with non-empty rev and active_labs
valid_overrides = {}
for lr in lab_cap_rows:
    if lr.get('approval', '').strip() != 'approved':
        continue
    rev_str = lr.get('rev', '').strip()
    al_str = lr.get('active_labs', '').strip()
    if rev_str == '' or al_str == '':
        continue
    pc = lr.get('panel_code', '').strip()
    rev_num = float(rev_str)
    if pc not in valid_overrides or rev_num > valid_overrides[pc][0]:
        valid_overrides[pc] = (rev_num, int(float(al_str)))

# ── Compute per-panel analysis ──
panel_results = []
for p in retained:
    pc = p['panel_code'].strip()
    pn = p.get('panel_name', '').strip()
    nt = p.get('network_tier', '').strip()
    sc = p.get('shipper_class', '').strip()
    reagent_cost_per_1000 = float(p['reagent_cost_per_1000_tests_usd'])
    tests_14 = int(p['tests_per_lab_per_run_14_day'])
    tests_28 = int(p['tests_per_lab_per_run_28_day'])
    default_labs = int(p['default_active_labs'])

    # Active labs
    if pc in valid_overrides:
        active_labs = valid_overrides[pc][1]
    else:
        active_labs = default_labs

    # Contract
    cr = contract_map.get(pc)
    if cr is None:
        # This shouldn't happen for retained panels but handle gracefully
        base_payment = 0.0
    else:
        base_payment = float(cr['base_payment_per_run_per_lab_usd'].strip())

    # Network adjustment
    net_adj = net_adj_map.get(nt, 0.0)

    # Shipper cost
    shipper_cost = shipper_map.get(sc, 0.0)

    total_payment = base_payment + net_adj

    # 14-day model: 26 runs/year
    runs_14 = 26
    runs_28 = 13

    annual_revenue_14 = total_payment * active_labs * runs_14
    annual_revenue_28 = total_payment * active_labs * runs_28

    annual_reagent_14 = reagent_cost_per_1000 * active_labs * tests_14 * runs_14 / 1000.0
    annual_reagent_28 = reagent_cost_per_1000 * active_labs * tests_28 * runs_28 / 1000.0

    annual_shipper_14 = shipper_cost * runs_14
    annual_shipper_28 = shipper_cost * runs_28

    annual_margin_14 = annual_revenue_14 - annual_reagent_14 - annual_shipper_14
    annual_margin_28 = annual_revenue_28 - annual_reagent_28 - annual_shipper_28

    diff = annual_margin_28 - annual_margin_14

    panel_results.append({
        'panel_code': pc,
        'panel_name': pn,
        'active_labs': active_labs,
        'reagent_cost_per_1000_tests_usd': round(reagent_cost_per_1000, 2),
        'network_tier': nt,
        'network_adjustment_per_run_per_lab_usd': round(net_adj, 2),
        'shipper_class': sc,
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

# ── Totals ──
total_margin_14 = round(sum(pr['annual_margin_14_day_usd'] for pr in panel_results), 2)
total_margin_28 = round(sum(pr['annual_margin_28_day_usd'] for pr in panel_results), 2)
total_diff = round(total_margin_28 - total_margin_14, 2)
abs_diff = round(abs(total_diff), 2)

# Decision
if abs_diff < 6000:
    decision = 'adopt_28_day'
    justification = f'Absolute total margin difference ${abs_diff} is below the $6,000 threshold; adopting 28-day cadence.'
else:
    decision = 'keep_14_day'
    justification = f'Absolute total margin difference ${abs_diff} meets or exceeds the $6,000 threshold; keeping 14-day cadence.'

# ── Build output JSON ──
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

print('JSON report written.')
print(f'Total 14-day margin: {total_margin_14}')
print(f'Total 28-day margin: {total_margin_28}')
print(f'Absolute difference: {abs_diff}')
print(f'Decision: {decision}')

# ── Build summary markdown ──
def fmt(v):
    """Format number with commas and 2 decimals."""
    return f'{v:,.2f}'

lines = [
    '# Diagnostics Panel Policy Summary',
    '',
    f'Total 14-day annual margin: ${fmt(total_margin_14)}',
    f'Total 28-day annual margin: ${fmt(total_margin_28)}',
    f'Absolute margin difference: ${fmt(abs_diff)}',
    f'Recommendation: {decision}',
    '',
    f'{justification}'
]

with open('/root/diagpanel_policy_summary.md', 'w') as f:
    f.write('\n'.join(lines) + '\n')

print('Summary written.')
```

Run the script:
```bash
python3 /root/solve.py
```

## 3. Validate outputs

```bash
cat /root/diagpanel_policy_report.json
cat /root/diagpanel_policy_summary.md
```

Verify:
- JSON is valid and parseable.
- All required keys exist in each panel entry (especially `annual_shipper_cost_14_day_usd`, `annual_shipper_cost_28_day_usd`, `annual_reagent_cost_14_day_usd`, `annual_reagent_cost_28_day_usd`).
- `assumptions.runs_per_year_14_day` is 26 and `runs_per_year_28_day` is 13 (NOT swapped).
- `metadata` and `audit_notes` match the template exactly.
- Panels are sorted by `panel_code` ascending.
- Summary has 4-8 non-empty lines and includes total 14-day margin, total 28-day margin, absolute difference (all with commas in formatting like `1,234.56`), and the exact decision slug.

## 4. Run the verifier if available

```bash
ls /root/test_output.py 2>/dev/null && cd /root && python3 -m pytest test_output.py -v
```

If any test fails, read the error carefully, fix the issue in solve.py, re-run, and re-validate. Pay special attention to:
- Shipper cost formula: `shipper_cost_usd * runs_per_year` (per panel, NOT multiplied by active_labs — re-check the task instructions; if the verifier expects it multiplied by active_labs, adjust accordingly)
- Reagent cost formula correctness
- Rounding to 2 decimals
- Summary formatting with commas in numbers

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