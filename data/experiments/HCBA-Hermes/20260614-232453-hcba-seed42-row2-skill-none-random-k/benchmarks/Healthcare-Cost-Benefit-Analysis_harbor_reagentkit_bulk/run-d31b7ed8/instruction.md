# Task Instruction

## Task: Healthcare Cost-Benefit Analysis — Reagent Kit Policy

You must produce two output files by reading and processing five input files according to precise rules. Follow every step carefully.

### Step 1: Read all input files

```bash
cat /root/assay_manifest.json
cat /root/carrier_cost.csv
cat /root/billing.csv
cat /root/lab_overrides.csv
cat /root/report_template.json
```

Read and understand the structure of each file before proceeding.

### Step 2: Write a Python script `/root/solve.py` that does the following

```python
import json, csv, math
from collections import defaultdict

# 1. Load assay_manifest.json
with open('/root/assay_manifest.json') as f:
    manifest = json.load(f)

# 2. Load carrier_cost.csv into a dict: carrier_type -> carrier_cost_usd (float)
with open('/root/carrier_cost.csv') as f:
    reader = csv.DictReader(f)
    carrier_costs = {}
    for row in reader:
        carrier_costs[row['carrier_type'].strip()] = float(row['carrier_cost_usd'].strip())

# 3. Load billing.csv
with open('/root/billing.csv') as f:
    reader = csv.DictReader(f)
    billing_rows = [row for row in reader]

# 4. Load lab_overrides.csv
with open('/root/lab_overrides.csv') as f:
    reader = csv.DictReader(f)
    override_rows = [row for row in reader]

# 5. Load report_template.json
with open('/root/report_template.json') as f:
    template = json.load(f)

# 6. Filter in-scope assays
assays_list = manifest.get('assays', manifest) if isinstance(manifest, dict) and 'assays' in manifest else manifest
if isinstance(manifest, dict) and 'assays' not in manifest:
    # Maybe it's a dict keyed by assay_id
    # Handle both list and dict formats
    pass

# Normalize: assays_list should be a list of assay dicts
if isinstance(assays_list, dict):
    # Could be {assay_id: {...}, ...}
    temp = []
    for k, v in assays_list.items():
        if isinstance(v, dict):
            if 'assay_id' not in v:
                v['assay_id'] = k
            temp.append(v)
    assays_list = temp

in_scope = [a for a in assays_list if a.get('in_scope') == True or a.get('in_scope') == 'true']

# 7. Build alias -> assay mapping
# For each in-scope assay, map assay_name and all aliases to the assay
assay_by_label = {}
for a in in_scope:
    aid = a['assay_id']
    name = a['assay_name']
    assay_by_label[name.strip().lower()] = a
    for alias in a.get('aliases', []):
        assay_by_label[alias.strip().lower()] = a

# 8. Resolve billing: match assay_label to assay, keep active, latest effective_month
# Group by assay_id
billing_by_assay = defaultdict(list)
for row in billing_rows:
    is_active = row.get('is_active', '').strip().lower()
    if is_active not in ('true', '1', 'yes'):
        continue
    label = row.get('assay_label', '').strip().lower()
    if label in assay_by_label:
        assay = assay_by_label[label]
        billing_by_assay[assay['assay_id']].append(row)

# For each assay, keep the row with latest effective_month
retained_billing = {}
for aid, rows in billing_by_assay.items():
    best = max(rows, key=lambda r: r.get('effective_month', ''))
    retained_billing[aid] = best

# 9. Resolve lab overrides
# Group approved rows by assay_id, keep highest revision
override_by_assay = defaultdict(list)
for row in override_rows:
    if row.get('status', '').strip().lower() == 'approved':
        override_by_assay[row['assay_id'].strip()].append(row)

resolved_labs = {}
for aid, rows in override_by_assay.items():
    best = max(rows, key=lambda r: int(r.get('revision', 0)))
    resolved_labs[aid] = int(best.get('active_labs', best.get('lab_count', 0)))

# 10. Build per-assay analysis
assay_results = []
for a in in_scope:
    aid = a['assay_id']
    aname = a['assay_name']
    
    # Active labs
    if aid in resolved_labs:
        active_labs = resolved_labs[aid]
    else:
        active_labs = int(a.get('default_active_labs', 0))
    
    reagent_price = float(a['reagent_price_per_1000_tests_usd'])
    carrier_type = a.get('carrier_type', '')
    carrier_cost = carrier_costs.get(carrier_type, 0.0)
    
    # Billing
    if aid in retained_billing:
        payment = float(retained_billing[aid].get('payment_per_run_per_lab_usd', 0))
    else:
        payment = 0.0
    
    tpr_small = int(a['tests_per_lab_per_run_small'])
    tpr_bulk = int(a['tests_per_lab_per_run_bulk'])
    
    runs_small = 24
    runs_bulk = 12
    
    # Annual revenue
    rev_small = payment * active_labs * runs_small
    rev_bulk = payment * active_labs * runs_bulk
    
    # Annual reagent cost
    rc_small = reagent_price * active_labs * tpr_small * runs_small / 1000.0
    rc_bulk = reagent_price * active_labs * tpr_bulk * runs_bulk / 1000.0
    
    # Annual carrier cost
    # carrier_cost is per-shipment. Shipments = runs_per_year (each run needs a shipment)
    cc_small = carrier_cost * runs_small
    cc_bulk = carrier_cost * runs_bulk
    
    # Annual margin
    margin_small = rev_small - rc_small - cc_small
    margin_bulk = rev_bulk - rc_bulk - cc_bulk
    
    diff = margin_bulk - margin_small
    
    assay_results.append({
        'assay_id': aid,
        'assay_name': aname,
        'active_labs': active_labs,
        'reagent_price_per_1000_tests_usd': round(reagent_price, 2),
        'carrier_type': carrier_type,
        'carrier_cost_usd': round(carrier_cost, 2),
        'payment_per_run_per_lab_usd': round(payment, 2),
        'tests_per_lab_per_run_small': tpr_small,
        'tests_per_lab_per_run_bulk': tpr_bulk,
        'annual_reagent_cost_small_kit_usd': round(rc_small, 2),
        'annual_reagent_cost_bulk_kit_usd': round(rc_bulk, 2),
        'annual_carrier_cost_small_kit_usd': round(cc_small, 2),
        'annual_carrier_cost_bulk_kit_usd': round(cc_bulk, 2),
        'annual_revenue_small_kit_usd': round(rev_small, 2),
        'annual_revenue_bulk_kit_usd': round(rev_bulk, 2),
        'annual_margin_small_kit_usd': round(margin_small, 2),
        'annual_margin_bulk_kit_usd': round(margin_bulk, 2),
        'annual_margin_difference_bulk_minus_small_usd': round(diff, 2)
    })

# Sort by assay_id ascending
assay_results.sort(key=lambda x: x['assay_id'])

# Totals
total_margin_small = sum(a['annual_margin_small_kit_usd'] for a in assay_results)
total_margin_bulk = sum(a['annual_margin_bulk_kit_usd'] for a in assay_results)
total_diff = round(total_margin_bulk - total_margin_small, 2)
abs_diff = round(abs(total_diff), 2)

# Decision
if abs_diff < 7000:
    decision = 'adopt_bulk_kit'
    justification = f'The absolute total margin difference of ${abs_diff} is below the $7,000 threshold, so bulk-kit restocking is recommended.'
else:
    decision = 'keep_small_kit'
    justification = f'The absolute total margin difference of ${abs_diff} exceeds the $7,000 threshold, so the small-kit cadence should be retained.'

# Build output JSON
report = {
    'metadata': template.get('metadata', {}),
    'analysis': {
        'assumptions': {
            'runs_per_year_small_kit': 24,
            'runs_per_year_bulk_kit': 12,
            'switch_threshold_usd': 7000,
            'lab_override_rule': 'highest approved revision per assay_id, else default_active_labs',
            'billing_rule': 'latest active effective_month per assay'
        },
        'assays': assay_results,
        'totals': {
            'total_annual_margin_small_kit_usd': round(total_margin_small, 2),
            'total_annual_margin_bulk_kit_usd': round(total_margin_bulk, 2),
            'total_annual_margin_difference_bulk_minus_small_usd': total_diff,
            'absolute_total_margin_difference_usd': abs_diff
        },
        'recommendation': {
            'decision': decision,
            'justification': justification
        }
    }
}

with open('/root/reagent_policy_report.json', 'w') as f:
    json.dump(report, f, indent=2)

# Build summary markdown
lines = [
    '# Reagent Kit Policy Summary',
    f'Total annual margin (small-kit): ${round(total_margin_small, 2):,.2f}',
    f'Total annual margin (bulk-kit): ${round(total_margin_bulk, 2):,.2f}',
    f'Absolute margin difference: ${abs_diff:,.2f}',
    f'Recommendation: {decision}',
    justification
]

with open('/root/reagent_policy_summary.md', 'w') as f:
    f.write('\n'.join(lines) + '\n')

print('Done. Files written.')
print(json.dumps(report, indent=2))
```

**IMPORTANT**: Before writing the script, first read all input files to understand their exact structure (column names, field names, data types). Then adapt the script accordingly. The script above is a template — you MUST adjust field names, column names, and data access patterns to match the actual file contents.

Specific things to verify after reading the files:
- In `lab_overrides.csv`: what is the column name for the lab count? It could be `active_labs`, `lab_count`, `num_labs`, etc.
- In `assay_manifest.json`: is it a list or dict? What are the exact field names?
- In `billing.csv`: exact column names.
- In `carrier_cost.csv`: exact column names.
- The `carrier_type` field: where does it come from for each assay? Check the manifest.
- How `carrier_cost` applies: the formula says `annual_carrier_cost` but doesn't specify per-lab or per-shipment. Look at the carrier_cost.csv values and the schema output fields to infer. The schema has `annual_carrier_cost_small_kit_usd` and `annual_carrier_cost_bulk_kit_usd` — think about whether carrier cost is per-shipment (runs_per_year), per-lab-per-shipment (active_labs * runs_per_year), or flat annual. The annual margin formula is `annual_revenue - annual_reagent_cost - annual_carrier_cost`, so carrier cost must be annual. Consider that carrier cost might be per-shipment and there's one shipment per run, so `annual_carrier_cost = carrier_cost_usd * runs_per_year`. Or it could be per-lab. Examine the data values to determine which interpretation makes sense.

### Step 3: Run the script

```bash
python3 /root/solve.py
```

### Step 4: Validate outputs

1. Verify `/root/reagent_policy_report.json` is valid JSON and has the correct schema:
```bash
python3 -c "
import json
with open('/root/reagent_policy_report.json') as f:
    r = json.load(f)
assert 'metadata' in r
assert 'analysis' in r
assert 'assays' in r['analysis']
assert 'totals' in r['analysis']
assert 'recommendation' in r['analysis']
assert 'assumptions' in r['analysis']
print('Schema OK')
print('Assay count:', len(r['analysis']['assays']))
print('Decision:', r['analysis']['recommendation']['decision'])
print('Totals:', json.dumps(r['analysis']['totals'], indent=2))
# Check assays sorted by assay_id
ids = [a['assay_id'] for a in r['analysis']['assays']]
assert ids == sorted(ids), 'Assays not sorted!'
print('Sort OK')
# Check all currency values are rounded to 2 decimals
for a in r['analysis']['assays']:
    for k, v in a.items():
        if 'usd' in k and isinstance(v, float):
            assert round(v, 2) == v, f'{k} not rounded: {v}'
print('Rounding OK')
"
```

2. Verify `/root/reagent_policy_summary.md`:
```bash
cat /root/reagent_policy_summary.md
python3 -c "
with open('/root/reagent_policy_summary.md') as f:
    lines = [l for l in f.read().strip().split('\\n') if l.strip()]
assert 4 <= len(lines) <= 8, f'Expected 4-8 non-empty lines, got {len(lines)}'
text = ' '.join(lines).lower()
assert 'adopt_bulk_kit' in text or 'keep_small_kit' in text, 'Missing decision slug'
print(f'Summary OK: {len(lines)} lines')
"
```

### Key Reminders
- Read ALL input files first before writing the script.
- Match `assay_label` in billing.csv to BOTH `assay_name` AND `aliases` from the manifest (case-insensitive matching recommended).
- Only `in_scope: true` assays.
- Only `is_active: true` billing rows.
- Only `status: approved` override rows.
- The `metadata` object from `report_template.json` must be preserved exactly as-is.
- All USD values rounded to 2 decimal places.
- Assays sorted by `assay_id` ascending.
- The summary must contain the exact slug `adopt_bulk_kit` or `keep_small_kit`.

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
Task metadata: author_email=gpt54@example.com, author_name=GPT-5.4, category=financial-analysis, difficulty=medium, tags=[lab-operations, json, csv, template-update, decision-analysis].
Verifier config: timeout_sec=900.0.