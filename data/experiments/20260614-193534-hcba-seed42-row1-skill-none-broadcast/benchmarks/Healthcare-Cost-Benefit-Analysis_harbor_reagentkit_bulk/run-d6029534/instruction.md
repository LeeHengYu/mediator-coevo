# Task Instruction

Execute the following steps in order:

## Step 1: Inspect all input files

Read and display the contents of each input file:
```
cat /root/assay_manifest.json
cat /root/carrier_cost.csv
cat /root/billing.csv
cat /root/lab_overrides.csv
cat /root/report_template.json
```

## Step 2: Write and run a Python script to produce both output files

Create `/root/solve.py` with the following logic:

```python
import json, csv, math
from collections import defaultdict

# Load inputs
with open('/root/assay_manifest.json') as f:
    manifest = json.load(f)
with open('/root/report_template.json') as f:
    template = json.load(f)

# Parse CSVs
def read_csv(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))

carrier_rows = read_csv('/root/carrier_cost.csv')
billing_rows = read_csv('/root/billing.csv')
override_rows = read_csv('/root/lab_overrides.csv')

# Build carrier cost lookup: carrier_type -> carrier_cost_usd
carrier_lookup = {}
for r in carrier_rows:
    carrier_lookup[r['carrier_type'].strip()] = float(r['carrier_cost_usd'])

# Filter in-scope assays from manifest
# manifest could be a list or dict; inspect and handle accordingly
if isinstance(manifest, dict) and 'assays' in manifest:
    assay_list = manifest['assays']
elif isinstance(manifest, list):
    assay_list = manifest
else:
    # try other keys
    assay_list = manifest.get('assays', manifest.get('assay_manifest', []))

in_scope = [a for a in assay_list if a.get('in_scope') is True or a.get('in_scope') == 'true']

# Build alias -> assay_id mapping for billing resolution
# Each assay has assay_name and possibly aliases
alias_to_assay = {}
for a in in_scope:
    aid = a['assay_id']
    alias_to_assay[a['assay_name'].strip().lower()] = aid
    for al in a.get('aliases', []):
        alias_to_assay[al.strip().lower()] = aid

# Filter active billing rows and resolve to assay_id, keep latest effective_month per assay
active_billing = [b for b in billing_rows if b.get('is_active', '').strip().lower() == 'true']

billing_by_assay = {}
for b in active_billing:
    label = b['assay_label'].strip().lower()
    aid = alias_to_assay.get(label)
    if aid is None:
        continue
    em = b['effective_month'].strip()
    if aid not in billing_by_assay or em > billing_by_assay[aid]['effective_month'].strip():
        billing_by_assay[aid] = b

# Resolve active labs from lab_overrides
approved = [r for r in override_rows if r.get('status', '').strip().lower() == 'approved']
override_by_assay = {}
for r in approved:
    aid = r['assay_id'].strip()
    rev = int(r['revision'])
    if aid not in override_by_assay or rev > override_by_assay[aid][0]:
        override_by_assay[aid] = (rev, r)

# Process each in-scope assay
assay_results = []
for a in in_scope:
    aid = a['assay_id']
    aname = a['assay_name']
    
    # Active labs
    if aid in override_by_assay:
        active_labs = int(override_by_assay[aid][1]['active_labs'])
    else:
        active_labs = int(a['default_active_labs'])
    
    reagent_price = float(a['reagent_price_per_1000_tests_usd'])
    carrier_type = a['carrier_type'].strip()
    carrier_cost = carrier_lookup[carrier_type]
    
    # Billing
    billing_row = billing_by_assay.get(aid)
    if billing_row is None:
        # This shouldn't happen for in-scope assays but handle gracefully
        payment_per_run = 0.0
    else:
        payment_per_run = float(billing_row['payment_per_run_per_lab_usd'])
    
    tpr_small = int(a['tests_per_lab_per_run_small'])
    tpr_bulk = int(a['tests_per_lab_per_run_bulk'])
    
    runs_small = 24
    runs_bulk = 12
    
    # Annual revenue
    rev_small = payment_per_run * active_labs * runs_small
    rev_bulk = payment_per_run * active_labs * runs_bulk
    
    # Annual reagent cost
    rc_small = reagent_price * active_labs * tpr_small * runs_small / 1000.0
    rc_bulk = reagent_price * active_labs * tpr_bulk * runs_bulk / 1000.0
    
    # Annual carrier cost
    # carrier_cost is per-shipment; shipments = runs_per_year * active_labs (one shipment per run per lab)
    # NOTE: The task says "annual_carrier_cost" but doesn't give an explicit formula.
    # The carrier_cost.csv gives a per-shipment cost. Each run requires a shipment to each lab.
    cc_small = carrier_cost * active_labs * runs_small
    cc_bulk = carrier_cost * active_labs * runs_bulk
    
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
        'payment_per_run_per_lab_usd': round(payment_per_run, 2),
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
total_margin_small = round(sum(a['annual_margin_small_kit_usd'] for a in assay_results), 2)
total_margin_bulk = round(sum(a['annual_margin_bulk_kit_usd'] for a in assay_results), 2)
total_diff = round(sum(a['annual_margin_difference_bulk_minus_small_usd'] for a in assay_results), 2)
abs_diff = round(abs(total_diff), 2)

# Decision
if abs_diff < 7000:
    decision = 'adopt_bulk_kit'
    justification = f'Absolute total margin difference ${abs_diff} is below the $7000 threshold, so switching to bulk kit is recommended.'
else:
    decision = 'keep_small_kit'
    justification = f'Absolute total margin difference ${abs_diff} meets or exceeds the $7000 threshold, so keeping small kit is recommended.'

# Build output JSON
output = {
    'metadata': template['metadata'],
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
            'total_annual_margin_small_kit_usd': total_margin_small,
            'total_annual_margin_bulk_kit_usd': total_margin_bulk,
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
    json.dump(output, f, indent=2)

# Write summary markdown
lines = [
    '# Reagent Policy Summary',
    f'Total small-kit annual margin: ${total_margin_small:,.2f} USD',
    f'Total bulk-kit annual margin: ${total_margin_bulk:,.2f} USD',
    f'Absolute margin difference: ${abs_diff:,.2f} USD',
    f'Recommendation: {decision}',
    f'Justification: {justification}'
]

with open('/root/reagent_policy_summary.md', 'w') as f:
    f.write('\n'.join(lines) + '\n')

print('Done.')
print(f'Total small-kit margin: {total_margin_small}')
print(f'Total bulk-kit margin: {total_margin_bulk}')
print(f'Total difference: {total_diff}')
print(f'Absolute difference: {abs_diff}')
print(f'Decision: {decision}')
```

Run it:
```
python3 /root/solve.py
```

## Step 3: Validate outputs

1. Display the generated JSON report:
```
cat /root/reagent_policy_report.json
```

2. Display the summary:
```
cat /root/reagent_policy_summary.md
```

3. Verify the JSON is valid:
```
python3 -c "import json; d=json.load(open('/root/reagent_policy_report.json')); print('Valid JSON'); print('Assay count:', len(d['analysis']['assays'])); print('Metadata:', d['metadata']); print('Decision:', d['analysis']['recommendation']['decision'])"
```

4. Verify the markdown has 4-8 non-empty lines and contains the required terms:
```
python3 -c "
lines = [l for l in open('/root/reagent_policy_summary.md').read().strip().split('\n') if l.strip()]
print(f'Non-empty lines: {len(lines)}')
assert 4 <= len(lines) <= 8, f'Expected 4-8 lines, got {len(lines)}'
text = open('/root/reagent_policy_summary.md').read()
for term in ['USD', 'adopt_bulk_kit', 'keep_small_kit']:
    # At least one of adopt_bulk_kit or keep_small_kit must appear
    pass
print('Summary validation passed')
"
```

## Important Notes

- After Step 1, inspect the actual structure of assay_manifest.json carefully. If it's structured differently than expected (e.g., different key names, nested differently), adapt the Python script before running it.
- The carrier cost formula is: `carrier_cost_usd * active_labs * runs_per_year` (one shipment per lab per run).
- If the script fails, read the error, fix the issue, and re-run. Do not give up.
- The `metadata` object from report_template.json must be preserved exactly as-is in the output.
- All currency values must be rounded to 2 decimal places.
- The assays array must be sorted by assay_id ascending.

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