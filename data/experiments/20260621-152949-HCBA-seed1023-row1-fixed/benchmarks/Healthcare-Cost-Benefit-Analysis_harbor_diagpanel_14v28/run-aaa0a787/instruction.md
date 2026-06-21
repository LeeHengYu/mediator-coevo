# Task Instruction

Execute the following steps exactly in order.

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

Also inspect the test file:
```bash
find /root -name 'test_output*' -o -name 'test_outputs*' | head -5
cat /root/tests/test_output*.py 2>/dev/null || cat /root/tests/test_outputs*.py 2>/dev/null || find /root -name '*.py' -path '*/test*' -exec cat {} \;
```

## 2. Write the Python solution script

Create `/root/solve.py` with the following logic (adapt field names to what you see in the actual data files):

```python
import json
import csv
import math

# Load inputs
with open('/root/panel_manifest.json') as f:
    manifest = json.load(f)

with open('/root/holdouts.json') as f:
    holdouts = json.load(f)

with open('/root/report_template.json') as f:
    template = json.load(f)

def read_csv(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))

shipper_rows = read_csv('/root/shipper_cost.csv')
contract_rows = read_csv('/root/contract_terms.csv')
network_rows = read_csv('/root/network_adjustments.csv')
override_rows = read_csv('/root/lab_capacity_overrides.csv')

# Build lookup: shipper_class -> shipper_cost_usd
shipper_map = {}
for r in shipper_rows:
    shipper_map[r['shipper_class'].strip()] = float(r['shipper_cost_usd'])

# Build lookup: network_tier -> network_adjustment_per_run_per_lab_usd
network_map = {}
for r in network_rows:
    network_map[r['network_tier'].strip()] = float(r['network_adjustment_per_run_per_lab_usd'])

# Build holdout exclusion set
exclude_set = set()
for h in holdouts:
    if h.get('holdout_state') == 'exclude':
        exclude_set.add(h.get('panel_code', '').strip())

# Filter manifest: analysis_mode == 'review' and not excluded
panels = []
for p in manifest:
    if p.get('analysis_mode') != 'review':
        continue
    if p['panel_code'].strip() in exclude_set:
        continue
    panels.append(p)

# Build mapping from panel_ref (name or alias) to panel_code
ref_to_code = {}
for p in panels:
    pname = p['panel_name'].strip()
    pcode = p['panel_code'].strip()
    ref_to_code[pname] = pcode
    for alias in p.get('alias_labels', []):
        ref_to_code[alias.strip()] = pcode

# Resolve contract_terms: status_flag == 'current', match panel_ref, latest effective_week
current_contracts = []
for r in contract_rows:
    if r.get('status_flag', '').strip() != 'current':
        continue
    pref = r.get('panel_ref', '').strip()
    if pref in ref_to_code:
        current_contracts.append((ref_to_code[pref], r))

# For each panel_code, keep the contract with latest effective_week
best_contract = {}
for code, r in current_contracts:
    ew = r.get('effective_week', '').strip()
    if code not in best_contract or ew > best_contract[code][1]:
        best_contract[code] = (r, ew)

# Resolve lab_capacity_overrides: approval=='approved', non-empty rev and active_labs, highest rev
best_override = {}
for r in override_rows:
    if r.get('approval', '').strip() != 'approved':
        continue
    rev_str = r.get('rev', '').strip()
    labs_str = r.get('active_labs', '').strip()
    if rev_str == '' or labs_str == '':
        continue
    pcode = r.get('panel_code', '').strip()
    rev_num = float(rev_str)
    if pcode not in best_override or rev_num > best_override[pcode][0]:
        best_override[pcode] = (rev_num, int(float(labs_str)))

# Build panel results
result_panels = []
for p in panels:
    pcode = p['panel_code'].strip()
    pname = p['panel_name'].strip()
    
    # Active labs
    if pcode in best_override:
        active_labs = best_override[pcode][1]
    else:
        active_labs = int(p['default_active_labs'])
    
    # Contract
    if pcode not in best_contract:
        continue  # should not happen for valid data
    cr = best_contract[pcode][0]
    base_payment = float(cr['base_payment_per_run_per_lab_usd'])
    
    # Network adjustment
    ntier = p.get('network_tier', '').strip()
    net_adj = network_map.get(ntier, 0.0)
    
    total_payment = base_payment + net_adj
    
    # Shipper
    sclass = p.get('shipper_class', '').strip()
    shipper_cost = shipper_map.get(sclass, 0.0)
    
    # Reagent
    reagent_cost_per_1000 = float(p['reagent_cost_per_1000_tests_usd'])
    
    # Tests per lab per run
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
    
    # Annual shipper cost
    shipper_14 = round(shipper_cost * runs_14, 2)
    shipper_28 = round(shipper_cost * runs_28, 2)
    
    # Annual margin
    margin_14 = round(rev_14 - reagent_14 - shipper_14, 2)
    margin_28 = round(rev_28 - reagent_28 - shipper_28, 2)
    
    diff = round(margin_28 - margin_14, 2)
    
    result_panels.append({
        'panel_code': pcode,
        'panel_name': pname,
        'active_labs': active_labs,
        'reagent_cost_per_1000_tests_usd': reagent_cost_per_1000,
        'network_tier': ntier,
        'network_adjustment_per_run_per_lab_usd': net_adj,
        'shipper_class': sclass,
        'shipper_cost_usd': shipper_cost,
        'base_payment_per_run_per_lab_usd': base_payment,
        'total_payment_per_run_per_lab_usd': round(total_payment, 2),
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
result_panels.sort(key=lambda x: x['panel_code'])

# Totals
total_margin_14 = round(sum(p['annual_margin_14_day_usd'] for p in result_panels), 2)
total_margin_28 = round(sum(p['annual_margin_28_day_usd'] for p in result_panels), 2)
total_diff = round(sum(p['annual_margin_difference_28_minus_14_usd'] for p in result_panels), 2)
abs_diff = round(abs(total_diff), 2)

# Decision
if abs_diff < 6000:
    decision = 'adopt_28_day'
    justification = f'Absolute total margin difference ${abs_diff} is below the $6000 threshold, recommending adoption of the 28-day cadence.'
else:
    decision = 'keep_14_day'
    justification = f'Absolute total margin difference ${abs_diff} exceeds the $6000 threshold, recommending retention of the 14-day cadence.'

# Build report
report = {
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
        'panels': result_panels,
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
    json.dump(report, f, indent=2)

# Summary markdown
lines = [
    '# Diagnostics Panel Policy Summary',
    '',
    f'Total 14-day annual margin: ${total_margin_14:,.2f} USD',
    f'Total 28-day annual margin: ${total_margin_28:,.2f} USD',
    f'Absolute margin difference: ${abs_diff:,.2f} USD',
    f'Decision: {decision}',
    '',
    f'Justification: {justification}'
]

with open('/root/diagpanel_policy_summary.md', 'w') as f:
    f.write('\n'.join(lines) + '\n')

print('Done.')
```

## 3. IMPORTANT: After inspecting the actual data files, adapt the script

Before running the script, carefully check:
- The exact field names in each CSV and JSON (they may differ slightly from what I assumed above, e.g., `tests_per_lab_per_run_14_day` might be `tests_per_lab_per_run_14d`). Adapt accordingly.
- Whether `panel_manifest.json` is a list or a dict with a key containing the list.
- Whether `holdouts.json` is a list or has a wrapper key.
- Whether `active_labs` in override CSV should be parsed as int or float.
- Whether `shipper_cost` is per-run or annual (the formula says `annual_shipper_cost = shipper_cost_usd * runs_per_year`, implying it's per-shipment/per-run).
- The `tests_per_lab_per_run` fields — check if they are integers or floats in the manifest.

## 4. Run the script

```bash
cd /root && python solve.py
```

## 5. Validate outputs

```bash
cat /root/diagpanel_policy_report.json | python -m json.tool > /dev/null && echo 'Valid JSON'
cat /root/diagpanel_policy_summary.md
wc -l /root/diagpanel_policy_summary.md
```

## 6. Run the test suite

```bash
cd /root && python -m pytest tests/ -v 2>&1 | head -80
```

## 7. If tests fail, debug

Read the exact assertion errors. Common pitfalls from prior failures:
- **14-day = 26 runs/year, 28-day = 13 runs/year** — do NOT swap these.
- **All per-panel fields must be present** in the JSON output including `network_tier`, `shipper_class`, `reagent_cost_per_1000_tests_usd`, `tests_per_lab_per_run_14_day`, `tests_per_lab_per_run_28_day`.
- **Assumptions object** must contain exactly the 6 keys shown in the schema: `runs_per_year_14_day`, `runs_per_year_28_day`, `switch_threshold_usd`, `override_rule`, `holdout_rule`, `adjustment_rule`. No extra keys.
- **Shipper cost formula**: `annual_shipper_cost = shipper_cost_usd * runs_per_year` (shipper cost is per shipment/run).
- **Margin formula**: `revenue - reagent_cost - shipper_cost`.
- **metadata and audit_notes** must be copied exactly from report_template.json.

Fix any issues and re-run until all tests pass.

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