# Task Instruction

Execute the following steps in order.

## Step 1 – Inspect all input files

Read and display every input file so we understand the exact data:

```bash
cat /root/panel_manifest.json
cat /root/shipper_cost.csv
cat /root/contract_terms.csv
cat /root/network_adjustments.csv
cat /root/lab_capacity_overrides.csv
cat /root/holdouts.json
cat /root/report_template.json
```

Also inspect the test file to understand what the verifier checks:

```bash
find /root -name '*.py' -path '*/test*' | head -20
cat /root/tests/test_output.py 2>/dev/null || cat /root/tests/test_outputs.py 2>/dev/null || find /root -name 'test_*' -exec cat {} \;
```

## Step 2 – Write and run the Python solution

Create `/root/solve.py` with the following logic. Read the task description below very carefully; every formula matters.

```python
import json, csv, math, os

# ── Load inputs ──────────────────────────────────────────────
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

# ── Build lookup maps ────────────────────────────────────────

# Shipper cost by shipper_class
shipper_map = {}
for r in shipper_rows:
    shipper_map[r['shipper_class'].strip()] = float(r['shipper_cost_usd'].strip())

# Network adjustment by network_tier
network_map = {}
for r in network_rows:
    network_map[r['network_tier'].strip()] = float(r['network_adjustment_per_run_per_lab_usd'].strip())

# ── Determine retained panels ───────────────────────────────
# panels is a list (or dict keyed by panel_code) in manifest
if isinstance(manifest, list):
    panels_list = manifest
else:
    # Could be dict with 'panels' key or top-level dict keyed by panel_code
    panels_list = manifest.get('panels', manifest if not any(k in manifest for k in ['metadata','audit_notes']) else [])
    if isinstance(panels_list, dict):
        # dict keyed by panel_code
        panels_list = list(panels_list.values())

# Filter to analysis_mode == 'review'
review_panels = [p for p in panels_list if p.get('analysis_mode','').strip().lower() == 'review']

# Exclude holdouts
exclude_codes = set()
if isinstance(holdouts, list):
    for h in holdouts:
        if h.get('holdout_state','').strip().lower() == 'exclude':
            exclude_codes.add(h.get('panel_code','').strip())
elif isinstance(holdouts, dict):
    for k, v in holdouts.items():
        if isinstance(v, dict) and v.get('holdout_state','').strip().lower() == 'exclude':
            exclude_codes.add(v.get('panel_code', k).strip())

retained = [p for p in review_panels if p['panel_code'].strip() not in exclude_codes]

# ── Resolve contract terms ───────────────────────────────────
# Build mapping: for each retained panel, find contract rows where
# panel_ref matches panel_name or any alias_labels entry, status_flag=current,
# then pick latest effective_week.

def resolve_contract(panel):
    pname = panel['panel_name'].strip()
    aliases = panel.get('alias_labels', [])
    if isinstance(aliases, str):
        aliases = [a.strip() for a in aliases.split(',')]
    match_set = set([pname] + [a.strip() for a in aliases])
    
    candidates = []
    for cr in contract_rows:
        if cr.get('status_flag','').strip().lower() != 'current':
            continue
        if cr.get('panel_ref','').strip() in match_set:
            candidates.append(cr)
    
    if not candidates:
        return None
    # latest effective_week
    candidates.sort(key=lambda x: x.get('effective_week',''), reverse=True)
    return candidates[0]

# ── Resolve lab capacity overrides ───────────────────────────
def resolve_active_labs(panel):
    pc = panel['panel_code'].strip()
    candidates = []
    for row in override_rows:
        if row.get('panel_code','').strip() != pc:
            continue
        if row.get('approval','').strip().lower() != 'approved':
            continue
        rev_val = row.get('rev','').strip()
        labs_val = row.get('active_labs','').strip()
        if rev_val == '' or labs_val == '':
            continue
        candidates.append(row)
    
    if not candidates:
        return int(panel['default_active_labs'])
    
    # highest numeric rev
    candidates.sort(key=lambda x: float(x['rev'].strip()), reverse=True)
    return int(candidates[0]['active_labs'].strip())

# ── Compute per-panel metrics ────────────────────────────────
RUNS_14 = 26
RUNS_28 = 13
THRESHOLD = 6000

results = []
for panel in retained:
    contract = resolve_contract(panel)
    if contract is None:
        # Skip panels with no contract? Or use 0? Let's skip – but this shouldn't happen.
        continue
    
    pc = panel['panel_code'].strip()
    pname = panel['panel_name'].strip()
    active_labs = resolve_active_labs(panel)
    
    reagent_cost_per_1000 = float(panel['reagent_cost_per_1000_tests_usd'])
    nt = panel.get('network_tier','').strip()
    net_adj = network_map.get(nt, 0.0)
    
    sc = panel.get('shipper_class','').strip()
    shipper_cost = shipper_map.get(sc, 0.0)
    
    base_pay = float(contract['base_payment_per_run_per_lab_usd'].strip())
    total_pay = base_pay + net_adj
    
    t14 = int(panel['tests_per_lab_per_run_14_day'])
    t28 = int(panel['tests_per_lab_per_run_28_day'])
    
    # Annual revenue: (base + net_adj) * active_labs * runs_per_year
    rev_14 = total_pay * active_labs * RUNS_14
    rev_28 = total_pay * active_labs * RUNS_28
    
    # Annual reagent cost: reagent_cost_per_1000 * active_labs * tests_per_lab_per_run * runs / 1000
    reagent_14 = reagent_cost_per_1000 * active_labs * t14 * RUNS_14 / 1000.0
    reagent_28 = reagent_cost_per_1000 * active_labs * t28 * RUNS_28 / 1000.0
    
    # Annual shipper cost: shipper_cost * active_labs * runs_per_year
    # NOTE: The previous run failed because shipper cost was shipper_cost * runs only.
    # The formula says annual_margin = annual_revenue - annual_reagent_cost - annual_shipper_cost
    # Revenue and reagent are both per-lab * labs * runs. Shipper should also be per-lab.
    # Actually let's re-read: shipper_cost_usd from shipper_cost.csv. It could be per-shipment.
    # The task says: annual_shipper_cost. Let's check if the numbers work with shipper * runs or shipper * labs * runs.
    # From feedback: "shipper_cost * active_labs * runs" was suggested.
    # Let's use shipper_cost * active_labs * runs_per_year
    shipper_14 = shipper_cost * active_labs * RUNS_14
    shipper_28 = shipper_cost * active_labs * RUNS_28
    
    margin_14 = rev_14 - reagent_14 - shipper_14
    margin_28 = rev_28 - reagent_28 - shipper_28
    diff = margin_28 - margin_14
    
    results.append({
        'panel_code': pc,
        'panel_name': pname,
        'active_labs': active_labs,
        'reagent_cost_per_1000_tests_usd': round(reagent_cost_per_1000, 2),
        'network_tier': nt,
        'network_adjustment_per_run_per_lab_usd': round(net_adj, 2),
        'shipper_class': sc,
        'shipper_cost_usd': round(shipper_cost, 2),
        'base_payment_per_run_per_lab_usd': round(base_pay, 2),
        'total_payment_per_run_per_lab_usd': round(total_pay, 2),
        'tests_per_lab_per_run_14_day': t14,
        'tests_per_lab_per_run_28_day': t28,
        'annual_reagent_cost_14_day_usd': round(reagent_14, 2),
        'annual_reagent_cost_28_day_usd': round(reagent_28, 2),
        'annual_shipper_cost_14_day_usd': round(shipper_14, 2),
        'annual_shipper_cost_28_day_usd': round(shipper_28, 2),
        'annual_revenue_14_day_usd': round(rev_14, 2),
        'annual_revenue_28_day_usd': round(rev_28, 2),
        'annual_margin_14_day_usd': round(margin_14, 2),
        'annual_margin_28_day_usd': round(margin_28, 2),
        'annual_margin_difference_28_minus_14_usd': round(diff, 2)
    })

# Sort by panel_code ascending
results.sort(key=lambda x: x['panel_code'])

total_margin_14 = round(sum(r['annual_margin_14_day_usd'] for r in results), 2)
total_margin_28 = round(sum(r['annual_margin_28_day_usd'] for r in results), 2)
total_diff = round(sum(r['annual_margin_difference_28_minus_14_usd'] for r in results), 2)
abs_diff = round(abs(total_diff), 2)

if abs_diff < THRESHOLD:
    decision = 'adopt_28_day'
    justification = f'Absolute total margin difference ${abs_diff} is below the ${THRESHOLD} threshold; switching to 28-day cadence is acceptable.'
else:
    decision = 'keep_14_day'
    justification = f'Absolute total margin difference ${abs_diff} exceeds the ${THRESHOLD} threshold; retaining 14-day cadence is recommended.'

# ── Build output JSON ────────────────────────────────────────
output = {
    'metadata': template['metadata'],
    'audit_notes': template['audit_notes'],
    'analysis': {
        'assumptions': {
            'runs_per_year_14_day': RUNS_14,
            'runs_per_year_28_day': RUNS_28,
            'switch_threshold_usd': THRESHOLD,
            'override_rule': 'highest numeric approved rev with non-empty active_labs, else default_active_labs',
            'holdout_rule': 'exclude holdout_state=exclude',
            'adjustment_rule': 'missing network_tier adjustment defaults to 0.0'
        },
        'panels': results,
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

# ── Build summary markdown ───────────────────────────────────
lines = [
    '# Diagnostics Panel Policy Summary',
    f'Total 14-day annual margin: ${total_margin_14} USD',
    f'Total 28-day annual margin: ${total_margin_28} USD',
    f'Absolute margin difference: ${abs_diff} USD',
    f'Decision: {decision}',
    justification
]

with open('/root/diagpanel_policy_summary.md', 'w') as f:
    f.write('\n'.join(lines) + '\n')

print('Done. Output files written.')
print(f'Panels analyzed: {len(results)}')
print(f'Total 14-day margin: {total_margin_14}')
print(f'Total 28-day margin: {total_margin_28}')
print(f'Total difference: {total_diff}')
print(f'Abs difference: {abs_diff}')
print(f'Decision: {decision}')
```

**IMPORTANT**: Before running `solve.py`, first inspect all input files carefully. After inspecting them, review the script above against the actual data structures. In particular:

- Check if `panel_manifest.json` is a list or a dict with a `panels` key.
- Check the exact field names in all CSV files (watch for whitespace, casing).
- Check `holdouts.json` structure.
- Check if `shipper_class` is in the panel manifest or contract terms.
- Adjust the script if any field names or structures differ from assumptions.

Then run:
```bash
python3 /root/solve.py
```

## Step 3 – Validate

After running, verify:
```bash
cat /root/diagpanel_policy_report.json
cat /root/diagpanel_policy_summary.md
```

Then run the test suite:
```bash
cd /root && python3 -m pytest tests/ -v 2>&1 | head -80
```

If tests fail, read the error messages carefully, identify the mismatch, fix `solve.py`, and re-run. Pay special attention to:
- Whether shipper cost should be `shipper_cost * runs` (per-network) or `shipper_cost * active_labs * runs` (per-lab). If the first attempt with `* active_labs * runs` fails, try just `* runs`.
- Whether any field names in the output JSON differ from what the test expects.
- Whether numeric values match expected values in the test.

Iterate until tests pass.

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