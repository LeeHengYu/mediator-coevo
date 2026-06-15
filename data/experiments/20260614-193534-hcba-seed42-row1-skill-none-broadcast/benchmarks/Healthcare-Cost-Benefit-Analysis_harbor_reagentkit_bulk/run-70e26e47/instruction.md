# Task Instruction

Execute the following Python script to produce both output files. Before running, inspect all input files to understand their structure.

```bash
cat /root/assay_manifest.json
cat /root/carrier_cost.csv
cat /root/billing.csv
cat /root/lab_overrides.csv
cat /root/report_template.json
```

Then run this Python script:

```python
import json
import csv
from collections import defaultdict

# Load inputs
with open('/root/assay_manifest.json') as f:
    manifest = json.load(f)

with open('/root/carrier_cost.csv') as f:
    carrier_costs = list(csv.DictReader(f))

with open('/root/billing.csv') as f:
    billing_rows = list(csv.DictReader(f))

with open('/root/lab_overrides.csv') as f:
    lab_overrides = list(csv.DictReader(f))

with open('/root/report_template.json') as f:
    template = json.load(f)

# Build carrier cost lookup
carrier_lookup = {}
for row in carrier_costs:
    carrier_lookup[row['carrier_type'].strip()] = float(row['carrier_cost_usd'])

# Filter in-scope assays
in_scope_assays = [a for a in manifest['assays'] if a.get('in_scope') == True]

# Build alias-to-assay mapping
alias_to_assay = {}
for assay in in_scope_assays:
    alias_to_assay[assay['assay_name'].strip().lower()] = assay['assay_id']
    for alias in assay.get('aliases', []):
        alias_to_assay[alias.strip().lower()] = assay['assay_id']

# Resolve billing: only active rows, map to assay_id, keep latest effective_month
active_billing = [r for r in billing_rows if r.get('is_active', '').strip().lower() == 'true']
billing_by_assay = {}
for row in active_billing:
    label = row['assay_label'].strip().lower()
    assay_id = alias_to_assay.get(label)
    if assay_id is None:
        continue
    em = row['effective_month'].strip()
    if assay_id not in billing_by_assay or em > billing_by_assay[assay_id]['effective_month'].strip():
        billing_by_assay[assay_id] = row

# Resolve lab overrides: only approved, highest revision per assay_id
approved_overrides = [r for r in lab_overrides if r.get('status', '').strip().lower() == 'approved']
override_by_assay = {}
for row in approved_overrides:
    aid = row['assay_id'].strip()
    rev = int(row['revision'])
    if aid not in override_by_assay or rev > int(override_by_assay[aid]['revision']):
        override_by_assay[aid] = row

# Build assay-level lookup
assay_lookup = {a['assay_id']: a for a in in_scope_assays}

# Calculate per-assay
assay_results = []
for assay in in_scope_assays:
    aid = assay['assay_id']
    aname = assay['assay_name']
    
    # Active labs
    if aid in override_by_assay:
        active_labs = int(override_by_assay[aid]['active_labs'])
    else:
        active_labs = int(assay['default_active_labs'])
    
    reagent_price = float(assay['reagent_price_per_1000_tests_usd'])
    carrier_type = assay['carrier_type'].strip()
    carrier_cost = carrier_lookup[carrier_type]
    
    # Billing
    billing_row = billing_by_assay.get(aid)
    if billing_row is None:
        payment_per_run = 0.0
    else:
        payment_per_run = float(billing_row['payment_per_run_per_lab_usd'])
    
    tests_small = int(assay['tests_per_lab_per_run_small'])
    tests_bulk = int(assay['tests_per_lab_per_run_bulk'])
    
    runs_small = 24
    runs_bulk = 12
    
    # Revenue
    rev_small = round(payment_per_run * active_labs * runs_small, 2)
    rev_bulk = round(payment_per_run * active_labs * runs_bulk, 2)
    
    # Reagent cost
    rc_small = round(reagent_price * active_labs * tests_small * runs_small / 1000, 2)
    rc_bulk = round(reagent_price * active_labs * tests_bulk * runs_bulk / 1000, 2)
    
    # Carrier cost
    cc_small = round(carrier_cost * active_labs * runs_small, 2)
    cc_bulk = round(carrier_cost * active_labs * runs_bulk, 2)
    
    # Margin
    margin_small = round(rev_small - rc_small - cc_small, 2)
    margin_bulk = round(rev_bulk - rc_bulk - cc_bulk, 2)
    diff = round(margin_bulk - margin_small, 2)
    
    assay_results.append({
        'assay_id': aid,
        'assay_name': aname,
        'active_labs': active_labs,
        'reagent_price_per_1000_tests_usd': reagent_price,
        'carrier_type': carrier_type,
        'carrier_cost_usd': carrier_cost,
        'payment_per_run_per_lab_usd': payment_per_run,
        'tests_per_lab_per_run_small': tests_small,
        'tests_per_lab_per_run_bulk': tests_bulk,
        'annual_reagent_cost_small_kit_usd': rc_small,
        'annual_reagent_cost_bulk_kit_usd': rc_bulk,
        'annual_carrier_cost_small_kit_usd': cc_small,
        'annual_carrier_cost_bulk_kit_usd': cc_bulk,
        'annual_revenue_small_kit_usd': rev_small,
        'annual_revenue_bulk_kit_usd': rev_bulk,
        'annual_margin_small_kit_usd': margin_small,
        'annual_margin_bulk_kit_usd': margin_bulk,
        'annual_margin_difference_bulk_minus_small_usd': diff
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
    justification = f'Absolute margin difference ${abs_diff:,.2f} is below the $7,000 threshold, so bulk-kit adoption is recommended.'
else:
    decision = 'keep_small_kit'
    justification = f'Absolute margin difference ${abs_diff:,.2f} exceeds the $7,000 threshold, so the small-kit cadence should be retained.'

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

print('JSON report written.')
print(f'Total small-kit margin: ${total_margin_small:,.2f}')
print(f'Total bulk-kit margin: ${total_margin_bulk:,.2f}')
print(f'Absolute difference: ${abs_diff:,.2f}')
print(f'Decision: {decision}')

# Write summary markdown
summary_lines = [
    '# Reagent Policy Summary',
    '',
    f'Total annual margin under small-kit policy: ${total_margin_small:,.2f}',
    f'Total annual margin under bulk-kit policy: ${total_margin_bulk:,.2f}',
    f'Absolute margin difference (bulk minus small): ${abs_diff:,.2f}',
    f'Recommendation: {decision}',
    '',
    justification
]

with open('/root/reagent_policy_summary.md', 'w') as f:
    f.write('\n'.join(summary_lines) + '\n')

print('Summary written.')
```

After running, verify:
1. `cat /root/reagent_policy_report.json` — check metadata matches template, assays sorted by assay_id, all currency values have 2 decimal places, assumptions keys match exactly (especially `switch_threshold_usd` not `decision_threshold_usd`).
2. `cat /root/reagent_policy_summary.md` — check 4-8 non-empty lines, includes comma-formatted USD values for total small-kit margin, total bulk-kit margin, absolute difference, and the exact decision slug (`adopt_bulk_kit` or `keep_small_kit`).
3. Verify the summary uses comma-separated currency formatting (e.g., `$42,908.83` not `$42908.83`).

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