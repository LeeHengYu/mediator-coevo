# Task Instruction

Execute the following steps in order:

## 1. Inspect all input files

```bash
cat /root/therapy_catalog.json
cat /root/bag_supply_cost.csv
cat /root/delivery_payment.csv
cat /root/patient_overrides.csv
```

## 2. Inspect the test/verifier files

```bash
find /root -name '*.py' -path '*/test*' | head -20
cat /tests/test_outputs.py 2>/dev/null || cat /root/tests/test_outputs.py 2>/dev/null || find / -name 'test_output*' -exec cat {} \;
```

## 3. Write and run a Python script to produce both output files

Create `/root/solve.py` with the following logic:

```python
import json
import csv
import os

# Load therapy catalog
with open('/root/therapy_catalog.json', 'r') as f:
    catalog = json.load(f)

# Load bag supply cost
bag_costs = {}
with open('/root/bag_supply_cost.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        bag_costs[int(row['bag_size_ml'])] = float(row['bag_supply_cost_usd'])

# Load delivery payment
delivery_payments_raw = []
with open('/root/delivery_payment.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        delivery_payments_raw.append(row)

# Load patient overrides
patient_overrides_raw = []
with open('/root/patient_overrides.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        patient_overrides_raw.append(row)

# Step 1: Filter in-scope therapies
in_scope = {}
for t in catalog['therapies'] if isinstance(catalog, dict) and 'therapies' in catalog else catalog:
    # Handle both possible structures
    therapy = t
    if therapy.get('include_in_review', False):
        code = therapy['therapy_code']
        in_scope[code] = therapy

# Step 2: Build name/alias -> therapy_code mapping for delivery payments
name_to_code = {}
for code, therapy in in_scope.items():
    name_to_code[therapy['therapy_name'].strip().lower()] = code
    for alias in therapy.get('aliases', []):
        name_to_code[alias.strip().lower()] = code

# Step 3: Resolve delivery payments to in-scope therapies
payment_map = {}  # therapy_code -> payment_per_delivery_per_patient_usd
for row in delivery_payments_raw:
    label = row['therapy_label'].strip().lower()
    if label in name_to_code:
        code = name_to_code[label]
        payment_map[code] = float(row['payment_per_delivery_per_patient_usd'])

# Step 4: Resolve patient overrides
# Filter approved, in-scope, keep highest revision per therapy_code
approved = {}
for row in patient_overrides_raw:
    if row['status'].strip().lower() != 'approved':
        continue
    code = row['therapy_code'].strip()
    if code not in in_scope:
        continue
    rev = int(row['revision'])
    if code not in approved or rev > approved[code]['revision']:
        approved[code] = {'revision': rev, 'active_patients': int(row['active_patients'])}

# Step 5: Compute per-therapy metrics
therapies_output = []
for code in sorted(in_scope.keys()):
    therapy = in_scope[code]
    active_patients = approved[code]['active_patients'] if code in approved else 0
    drug_cost_per_1000 = float(therapy['drug_cost_per_1000_mg_usd'])
    dose_mg = float(therapy['dose_mg_per_day'])
    bag_size = int(therapy['bag_size_ml'])
    bag_cost = bag_costs[bag_size]
    payment = payment_map.get(code, 0.0)

    # 7-day model
    deliveries_7 = 52
    days_7 = 7
    annual_drug_cost_7 = drug_cost_per_1000 * active_patients * dose_mg * days_7 * deliveries_7 / 1000
    annual_supply_cost_7 = bag_cost * active_patients * deliveries_7
    annual_revenue_7 = payment * active_patients * deliveries_7
    annual_margin_7 = annual_revenue_7 - annual_drug_cost_7 - annual_supply_cost_7

    # 14-day model
    deliveries_14 = 26
    days_14 = 14
    annual_drug_cost_14 = drug_cost_per_1000 * active_patients * dose_mg * days_14 * deliveries_14 / 1000
    annual_supply_cost_14 = bag_cost * active_patients * deliveries_14
    annual_revenue_14 = payment * active_patients * deliveries_14
    annual_margin_14 = annual_revenue_14 - annual_drug_cost_14 - annual_supply_cost_14

    diff = annual_margin_14 - annual_margin_7

    therapies_output.append({
        'therapy_code': code,
        'therapy_name': therapy['therapy_name'],
        'active_patients': active_patients,
        'drug_cost_per_1000_mg_usd': round(drug_cost_per_1000, 2),
        'dose_mg_per_day': round(dose_mg, 2),
        'bag_size_ml': bag_size,
        'bag_supply_cost_usd': round(bag_cost, 2),
        'payment_per_delivery_per_patient_usd': round(payment, 2),
        'annual_drug_cost_7_day_usd': round(annual_drug_cost_7, 2),
        'annual_drug_cost_14_day_usd': round(annual_drug_cost_14, 2),
        'annual_supply_cost_7_day_usd': round(annual_supply_cost_7, 2),
        'annual_supply_cost_14_day_usd': round(annual_supply_cost_14, 2),
        'annual_revenue_7_day_usd': round(annual_revenue_7, 2),
        'annual_revenue_14_day_usd': round(annual_revenue_14, 2),
        'annual_margin_7_day_usd': round(annual_margin_7, 2),
        'annual_margin_14_day_usd': round(annual_margin_14, 2),
        'annual_margin_difference_14_minus_7_usd': round(diff, 2)
    })

# Step 6: Totals
total_margin_7 = sum(t['annual_margin_7_day_usd'] for t in therapies_output)
total_margin_14 = sum(t['annual_margin_14_day_usd'] for t in therapies_output)
total_diff = round(total_margin_14 - total_margin_7, 2)
abs_diff = round(abs(total_diff), 2)

# Step 7: Decision
if abs_diff < 15000:
    decision = 'move_to_14_day'
    justification = f'The absolute margin difference of ${abs_diff:,.2f} is below the $15,000.00 threshold, so switching to 14-day delivery is recommended.'
else:
    decision = 'keep_7_day'
    justification = f'The absolute margin difference of ${abs_diff:,.2f} exceeds the $15,000.00 threshold, so keeping 7-day delivery is recommended.'

result = {
    'assumptions': {
        'deliveries_per_year_7_day': 52,
        'deliveries_per_year_14_day': 26,
        'days_per_delivery_7_day': 7,
        'days_per_delivery_14_day': 14,
        'switch_threshold_usd': 15000,
        'patient_override_rule': 'highest approved revision per therapy_code'
    },
    'therapies': therapies_output,
    'totals': {
        'total_annual_margin_7_day_usd': round(total_margin_7, 2),
        'total_annual_margin_14_day_usd': round(total_margin_14, 2),
        'total_annual_margin_difference_14_minus_7_usd': total_diff,
        'absolute_total_margin_difference_usd': abs_diff
    },
    'recommendation': {
        'decision': decision,
        'justification': justification
    }
}

with open('/root/infusion_batch_analysis.json', 'w') as f:
    json.dump(result, f, indent=2)

print('JSON written.')
print(json.dumps(result, indent=2))

# Step 8: Write summary markdown
# IMPORTANT: Use f'{value:,.2f}' for thousands-separator formatting
lines = [
    '# Infusion Batch Analysis Summary',
    '',
    f'Total 7-day annual margin: ${round(total_margin_7, 2):,.2f}',
    f'Total 14-day annual margin: ${round(total_margin_14, 2):,.2f}',
    f'Absolute margin difference: ${abs_diff:,.2f}',
    f'Recommendation: {decision}',
    '',
    justification
]

with open('/root/infusion_batch_summary.md', 'w') as f:
    f.write('\n'.join(lines) + '\n')

print('Summary written.')
```

Run the script:
```bash
python3 /root/solve.py
```

## 4. Validate outputs

```bash
cat /root/infusion_batch_analysis.json
cat /root/infusion_batch_summary.md
python3 -c "import json; d=json.load(open('/root/infusion_batch_analysis.json')); print('Therapies:', len(d['therapies'])); print('Decision:', d['recommendation']['decision']); print('Sorted codes:', [t['therapy_code'] for t in d['therapies']])"
```

## 5. Run the verifier tests

```bash
cd / && python -m pytest tests/ -x -v 2>&1 | head -80
```

If tests fail, read the error output carefully, fix the issue in solve.py, re-run, and re-test. Pay special attention to:
- Currency formatting in the summary must use comma thousands separators (e.g., `$27,000.00` not `$27000.00`)
- The `annual_supply_cost` formula: it is `bag_supply_cost_usd * active_patients * deliveries_per_year` (one bag per delivery per patient)
- The JSON schema field names must match exactly
- The therapies array must be sorted by therapy_code ascending
- All currency values rounded to 2 decimal places
- The decision slug must be exactly `move_to_14_day` or `keep_7_day`
- The summary must have 4-8 non-empty lines and include all required values

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
Task metadata: author_email=gpt54@example.com, author_name=GPT-5.4, category=financial-analysis, difficulty=medium, tags=[home-infusion, json, csv, alias-resolution, decision-analysis].
Verifier config: timeout_sec=900.0.