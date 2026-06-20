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

Also inspect the test file:
```
cat /root/test_output.py
```

## Step 2: Write the Python computation script

Create `/root/solve.py` that does the following:

```python
import json, csv, math

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

# Build lookup: network_tier -> adjustment
network_map = {}
for r in network_rows:
    network_map[r['network_tier'].strip()] = float(r['network_adjustment_per_run_per_lab_usd'])

# Build holdout exclude set
exclude_set = set()
for h in holdouts:
    if h.get('holdout_state') == 'exclude':
        exclude_set.add(h['panel_code'].strip())

# Filter manifest: analysis_mode == 'review' and not excluded
panels = []
for p in manifest['panels']:
    if p.get('analysis_mode') == 'review' and p['panel_code'].strip() not in exclude_set:
        panels.append(p)

# For each panel, resolve contract_terms row
# Match panel_ref to panel_name or any alias_labels entry
# Only status_flag == 'current', latest effective_week
def resolve_contract(panel):
    pname = panel['panel_name'].strip()
    aliases = [a.strip() for a in panel.get('alias_labels', [])]
    candidates = []
    for cr in contract_rows:
        if cr.get('status_flag', '').strip() != 'current':
            continue
        ref = cr['panel_ref'].strip()
        if ref == pname or ref in aliases:
            candidates.append(cr)
    if not candidates:
        return None
    # latest effective_week
    candidates.sort(key=lambda x: x['effective_week'].strip(), reverse=True)
    return candidates[0]

# Resolve lab_capacity_overrides
def resolve_active_labs(panel):
    pcode = panel['panel_code'].strip()
    candidates = []
    for r in override_rows:
        if r.get('panel_code', '').strip() != pcode:
            continue
        if r.get('approval', '').strip() != 'approved':
            continue
        rev_val = r.get('rev', '').strip()
        labs_val = r.get('active_labs', '').strip()
        if rev_val == '' or labs_val == '':
            continue
        candidates.append(r)
    if not candidates:
        return int(panel['default_active_labs'])
    # highest numeric rev
    candidates.sort(key=lambda x: float(x['rev'].strip()), reverse=True)
    return int(candidates[0]['active_labs'].strip())

result_panels = []
for p in panels:
    contract = resolve_contract(p)
    if contract is None:
        continue  # shouldn't happen per task design
    
    pcode = p['panel_code'].strip()
    pname = p['panel_name'].strip()
    active_labs = resolve_active_labs(p)
    reagent_cost_per_1000 = float(p['reagent_cost_per_1000_tests_usd'])
    ntier = p['network_tier'].strip()
    net_adj = network_map.get(ntier, 0.0)
    shipper_class = p['shipper_class'].strip()
    shipper_cost = shipper_map.get(shipper_class, 0.0)
    base_pay = float(contract['base_payment_per_run_per_lab_usd'])
    total_pay = round(base_pay + net_adj, 2)
    tests_14 = int(p['tests_per_lab_per_run_14_day'])
    tests_28 = int(p['tests_per_lab_per_run_28_day'])
    
    runs_14 = 26
    runs_28 = 13
    
    annual_rev_14 = round((base_pay + net_adj) * active_labs * runs_14, 2)
    annual_rev_28 = round((base_pay + net_adj) * active_labs * runs_28, 2)
    
    annual_reagent_14 = round(reagent_cost_per_1000 * active_labs * tests_14 * runs_14 / 1000, 2)
    annual_reagent_28 = round(reagent_cost_per_1000 * active_labs * tests_28 * runs_28 / 1000, 2)
    
    annual_shipper_14 = round(shipper_cost * runs_14, 2)
    annual_shipper_28 = round(shipper_cost * runs_28, 2)
    
    annual_margin_14 = round(annual_rev_14 - annual_reagent_14 - annual_shipper_14, 2)
    annual_margin_28 = round(annual_rev_28 - annual_reagent_28 - annual_shipper_28, 2)
    
    diff = round(annual_margin_28 - annual_margin_14, 2)
    
    result_panels.append({
        "panel_code": pcode,
        "panel_name": pname,
        "active_labs": active_labs,
        "reagent_cost_per_1000_tests_usd": reagent_cost_per_1000,
        "network_tier": ntier,
        "network_adjustment_per_run_per_lab_usd": net_adj,
        "shipper_class": shipper_class,
        "shipper_cost_usd": shipper_cost,
        "base_payment_per_run_per_lab_usd": base_pay,
        "total_payment_per_run_per_lab_usd": total_pay,
        "tests_per_lab_per_run_14_day": tests_14,
        "tests_per_lab_per_run_28_day": tests_28,
        "annual_reagent_cost_14_day_usd": annual_reagent_14,
        "annual_reagent_cost_28_day_usd": annual_reagent_28,
        "annual_shipper_cost_14_day_usd": annual_shipper_14,
        "annual_shipper_cost_28_day_usd": annual_shipper_28,
        "annual_revenue_14_day_usd": annual_rev_14,
        "annual_revenue_28_day_usd": annual_rev_28,
        "annual_margin_14_day_usd": annual_margin_14,
        "annual_margin_28_day_usd": annual_margin_28,
        "annual_margin_difference_28_minus_14_usd": diff
    })

# Sort by panel_code ascending
result_panels.sort(key=lambda x: x['panel_code'])

total_margin_14 = round(sum(p['annual_margin_14_day_usd'] for p in result_panels), 2)
total_margin_28 = round(sum(p['annual_margin_28_day_usd'] for p in result_panels), 2)
total_diff = round(total_margin_28 - total_margin_14, 2)
abs_diff = round(abs(total_diff), 2)

if abs_diff < 6000:
    decision = 'adopt_28_day'
    justification = f'Absolute total margin difference ${abs_diff:,.2f} is below the $6,000.00 threshold; switching to 28-day cadence is acceptable.'
else:
    decision = 'keep_14_day'
    justification = f'Absolute total margin difference ${abs_diff:,.2f} exceeds the $6,000.00 threshold; retaining 14-day cadence is recommended.'

report = {
    "metadata": template["metadata"],
    "audit_notes": template["audit_notes"],
    "analysis": {
        "assumptions": {
            "runs_per_year_14_day": 26,
            "runs_per_year_28_day": 13,
            "switch_threshold_usd": 6000,
            "override_rule": "highest numeric approved rev with non-empty active_labs, else default_active_labs",
            "holdout_rule": "exclude holdout_state=exclude",
            "adjustment_rule": "missing network_tier adjustment defaults to 0.0"
        },
        "panels": result_panels,
        "totals": {
            "total_annual_margin_14_day_usd": total_margin_14,
            "total_annual_margin_28_day_usd": total_margin_28,
            "total_annual_margin_difference_28_minus_14_usd": total_diff,
            "absolute_total_margin_difference_usd": abs_diff
        },
        "recommendation": {
            "decision": decision,
            "justification": justification
        }
    }
}

with open('/root/diagpanel_policy_report.json', 'w') as f:
    json.dump(report, f, indent=2)

# Write summary markdown
lines = [
    '# Diagnostics Panel Policy Summary',
    f'Total 14-day annual margin: ${total_margin_14:,.2f}',
    f'Total 28-day annual margin: ${total_margin_28:,.2f}',
    f'Absolute margin difference: ${abs_diff:,.2f}',
    f'Recommendation: {decision}',
]
with open('/root/diagpanel_policy_summary.md', 'w') as f:
    f.write('\n'.join(lines) + '\n')

print('Done. Report and summary written.')
print(f'Panels: {len(result_panels)}, Decision: {decision}')
print(f'14-day margin: {total_margin_14}, 28-day margin: {total_margin_28}, diff: {total_diff}')
```

**IMPORTANT**: Before writing the script, inspect all input files first. The script above is a template — you MUST adapt it based on the actual structure of the input files (field names, JSON structure, etc.). In particular:
- Check if `panel_manifest.json` has a `panels` array or is structured differently.
- Check exact CSV column names (they may have spaces or different casing).
- Check `holdouts.json` structure.
- Check `report_template.json` structure.

## Step 3: Run the script
```
cd /root && python solve.py
```

## Step 4: Validate outputs
```
cat /root/diagpanel_policy_report.json
cat /root/diagpanel_policy_summary.md
```

Verify:
- JSON has correct schema with all required keys
- `metadata` and `audit_notes` match template exactly
- `assumptions` has exactly the 6 required keys
- `panels` are sorted by `panel_code`
- `totals` has the 4 required keys
- `recommendation` has `decision` and `justification`
- Summary has 4-8 non-empty lines with all required values

## Step 5: Run the test suite
```
cd /root && python -m pytest test_output.py -v
```

If any tests fail, read the error messages carefully, re-inspect the input files and your logic, fix the script, re-run, and re-test. Pay special attention to:
- Exact field name matching between CSVs and your code
- The shipper cost formula (it uses `shipper_cost_usd * runs_per_year` — confirm this is correct per the task or if it should also multiply by active_labs; re-read the task instructions carefully — the task says `annual_shipper_cost` without specifying multiplication by active_labs, so check the test expectations)
- The contract matching logic (match `panel_ref` to `panel_name` OR any `alias_labels` entry)
- Rounding: all currency to 2 decimals
- Whether `active_labs` should be int or could be float

Iterate until all tests pass.

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