# Task Instruction

Execute the following steps in order:

## 1. Inspect all input files

```bash
cat /root/assay_manifest.json
cat /root/carrier_cost.csv
cat /root/billing.csv
cat /root/lab_overrides.csv
cat /root/report_template.json
```

## 2. Write and run a Python script

Create `/root/solve.py` with the following logic:

```python
import json
import csv
from collections import defaultdict

# Load inputs
with open('/root/assay_manifest.json') as f:
    manifest = json.load(f)

with open('/root/carrier_cost.csv') as f:
    carrier_rows = list(csv.DictReader(f))

with open('/root/billing.csv') as f:
    billing_rows = list(csv.DictReader(f))

with open('/root/lab_overrides.csv') as f:
    override_rows = list(csv.DictReader(f))

with open('/root/report_template.json') as f:
    template = json.load(f)

# Build carrier cost lookup: carrier_type -> carrier_cost_usd
carrier_lookup = {}
for row in carrier_rows:
    carrier_lookup[row['carrier_type'].strip()] = float(row['carrier_cost_usd'])

# Filter in-scope assays
assays = manifest.get('assays', manifest) if isinstance(manifest, list) else manifest.get('assays', [])
# Handle if manifest is a dict with assays key or is itself a list
if isinstance(manifest, dict) and 'assays' not in manifest:
    # Try to find the assays key
    for k, v in manifest.items():
        if isinstance(v, list):
            assays = v
            break

in_scope = [a for a in assays if a.get('in_scope') == True]

# Build alias -> assay_id mapping
alias_to_assay = {}
for a in in_scope:
    aid = a['assay_id']
    alias_to_assay[a['assay_name'].strip().lower()] = aid
    for alias in a.get('aliases', []):
        alias_to_assay[alias.strip().lower()] = aid

# Resolve billing: only active rows, match by assay_label to assay_name or alias
# Group by assay_id, keep latest effective_month
billing_by_assay = {}
for row in billing_rows:
    if row.get('is_active', '').strip().lower() != 'true':
        continue
    label = row['assay_label'].strip().lower()
    aid = alias_to_assay.get(label)
    if aid is None:
        continue
    em = row['effective_month'].strip()
    if aid not in billing_by_assay or em > billing_by_assay[aid]['effective_month']:
        billing_by_assay[aid] = {
            'effective_month': em,
            'payment_per_run_per_lab_usd': float(row['payment_per_run_per_lab_usd'])
        }

# Resolve lab overrides: approved rows, highest revision per assay_id
override_by_assay = {}
for row in override_rows:
    if row.get('status', '').strip().lower() != 'approved':
        continue
    aid = row['assay_id'].strip()
    rev = int(row['revision'])
    if aid not in override_by_assay or rev > override_by_assay[aid]['revision']:
        override_by_assay[aid] = {
            'revision': rev,
            'active_labs': int(row['active_labs'])
        }

# Process each in-scope assay
result_assays = []
for a in in_scope:
    aid = a['assay_id']
    aname = a['assay_name']
    
    # Active labs
    if aid in override_by_assay:
        active_labs = override_by_assay[aid]['active_labs']
    else:
        active_labs = int(a['default_active_labs'])
    
    reagent_price = float(a['reagent_price_per_1000_tests_usd'])
    carrier_type = a['carrier_type'].strip()
    carrier_cost = carrier_lookup[carrier_type]
    
    tests_small = int(a['tests_per_lab_per_run_small'])
    tests_bulk = int(a['tests_per_lab_per_run_bulk'])
    
    # Billing
    billing_info = billing_by_assay.get(aid, {})
    payment_per_run = billing_info.get('payment_per_run_per_lab_usd', 0.0)
    
    runs_small = 24
    runs_bulk = 12
    
    # Annual revenue = payment_per_run_per_lab_usd * active_labs * runs_per_year
    rev_small = round(payment_per_run * active_labs * runs_small, 2)
    rev_bulk = round(payment_per_run * active_labs * runs_bulk, 2)
    
    # Annual reagent cost = reagent_price_per_1000_tests * active_labs * tests_per_lab_per_run * runs / 1000
    reagent_small = round(reagent_price * active_labs * tests_small * runs_small / 1000, 2)
    reagent_bulk = round(reagent_price * active_labs * tests_bulk * runs_bulk / 1000, 2)
    
    # Annual carrier cost = carrier_cost_usd * active_labs * runs_per_year
    # (IMPORTANT: carrier cost is per lab per run, multiply by active_labs and runs)
    carrier_small = round(carrier_cost * active_labs * runs_small, 2)
    carrier_bulk = round(carrier_cost * active_labs * runs_bulk, 2)
    
    # Annual margin = revenue - reagent_cost - carrier_cost
    margin_small = round(rev_small - reagent_small - carrier_small, 2)
    margin_bulk = round(rev_bulk - reagent_bulk - carrier_bulk, 2)
    
    diff = round(margin_bulk - margin_small, 2)
    
    result_assays.append({
        'assay_id': aid,
        'assay_name': aname,
        'active_labs': active_labs,
        'reagent_price_per_1000_tests_usd': reagent_price,
        'carrier_type': carrier_type,
        'carrier_cost_usd': carrier_cost,
        'payment_per_run_per_lab_usd': payment_per_run,
        'tests_per_lab_per_run_small': tests_small,
        'tests_per_lab_per_run_bulk': tests_bulk,
        'annual_reagent_cost_small_kit_usd': reagent_small,
        'annual_reagent_cost_bulk_kit_usd': reagent_bulk,
        'annual_carrier_cost_small_kit_usd': carrier_small,
        'annual_carrier_cost_bulk_kit_usd': carrier_bulk,
        'annual_revenue_small_kit_usd': rev_small,
        'annual_revenue_bulk_kit_usd': rev_bulk,
        'annual_margin_small_kit_usd': margin_small,
        'annual_margin_bulk_kit_usd': margin_bulk,
        'annual_margin_difference_bulk_minus_small_usd': diff
    })

# Sort by assay_id ascending
result_assays.sort(key=lambda x: x['assay_id'])

# Totals
total_margin_small = round(sum(a['annual_margin_small_kit_usd'] for a in result_assays), 2)
total_margin_bulk = round(sum(a['annual_margin_bulk_kit_usd'] for a in result_assays), 2)
total_diff = round(total_margin_bulk - total_margin_small, 2)
abs_diff = round(abs(total_diff), 2)

# Decision
if abs_diff < 7000:
    decision = 'adopt_bulk_kit'
else:
    decision = 'keep_small_kit'

justification = (f"The absolute total margin difference is ${abs_diff:.2f}, which is "
                 f"{'below' if abs_diff < 7000 else 'at or above'} the $7,000 threshold. "
                 f"Recommendation: {decision}.")

# Build output
output = {
    'metadata': template.get('metadata', {}),
    'analysis': {
        'assumptions': {
            'runs_per_year_small_kit': 24,
            'runs_per_year_bulk_kit': 12,
            'switch_threshold_usd': 7000,
            'lab_override_rule': 'highest approved revision per assay_id, else default_active_labs',
            'billing_rule': 'latest active effective_month per assay'
        },
        'assays': result_assays,
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
print(f'Total small-kit margin: ${total_margin_small:.2f}')
print(f'Total bulk-kit margin: ${total_margin_bulk:.2f}')
print(f'Absolute difference: ${abs_diff:.2f}')
print(f'Decision: {decision}')

# Write summary markdown (4-8 non-empty lines)
lines = [
    '# Reagent Policy Summary',
    '',
    f'Total annual margin under small-kit policy: ${total_margin_small:.2f} USD',
    f'Total annual margin under bulk-kit policy: ${total_margin_bulk:.2f} USD',
    f'Absolute margin difference (bulk minus small): ${abs_diff:.2f} USD',
    f'Decision threshold: $7,000.00 USD',
    f'Final decision: {decision}',
    f'Justification: {justification}'
]

with open('/root/reagent_policy_summary.md', 'w') as f:
    f.write('\n'.join(lines) + '\n')

print('Summary written.')
```

Run the script:
```bash
python3 /root/solve.py
```

## 3. Validate outputs

```bash
cat /root/reagent_policy_report.json
cat /root/reagent_policy_summary.md
```

Verify:
- The JSON has `metadata` matching the template exactly.
- `analysis.assays` is sorted by `assay_id`.
- All currency values are rounded to 2 decimals.
- The summary has 4-8 non-empty lines and includes total small-kit margin, total bulk-kit margin, absolute difference, and the exact decision slug.
- The carrier cost formula uses `carrier_cost_usd * active_labs * runs_per_year` (this was the key bug from the previous iteration where `active_labs` was missing).

## 4. Handle edge cases during inspection

When inspecting the input files in step 1, pay attention to:
- The exact key names in `assay_manifest.json` (the manifest might be a dict with an `assays` key, or the structure may vary).
- Whether `aliases` is present and its format.
- Column names in CSVs (check for whitespace or unexpected naming).
- Whether `is_active` in billing.csv is boolean string 'true'/'True' or other format.

If the manifest structure differs from what the script assumes, adjust the script accordingly before running. The script includes some flexibility but you should verify after reading the files.

If the test/verifier script exists (e.g., `test_output.py`), run it after generating outputs:
```bash
cd /root && python3 -m pytest test_output.py -v 2>&1 | head -80
```
Fix any failures before finishing.

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