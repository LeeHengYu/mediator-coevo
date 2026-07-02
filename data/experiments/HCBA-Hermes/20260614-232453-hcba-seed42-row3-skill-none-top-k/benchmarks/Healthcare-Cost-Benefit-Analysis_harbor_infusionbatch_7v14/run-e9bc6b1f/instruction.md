# Task Instruction

Execute the following steps in order:

1. **Inspect all input files** to understand their structure:
```bash
cat /root/therapy_catalog.json
cat /root/bag_supply_cost.csv
cat /root/delivery_payment.csv
cat /root/patient_overrides.csv
```

2. **Inspect the test file** to understand exact validation expectations:
```bash
cat /tests/test_outputs.py
```

3. **Create `/root/solve.py`** that does the following:

```python
import json
import csv

# Load therapy catalog
with open('/root/therapy_catalog.json') as f:
    catalog = json.load(f)

# Filter to in-scope therapies (include_in_review == true)
in_scope = {t['therapy_code']: t for t in catalog['therapies'] if t.get('include_in_review') is True}

# Build alias-to-therapy_code mapping
# Map therapy_name and all aliases to therapy_code
name_to_code = {}
for code, t in in_scope.items():
    name_to_code[t['therapy_name'].strip().lower()] = code
    for alias in t.get('aliases', []):
        name_to_code[alias.strip().lower()] = code

# Load bag supply costs
bag_costs = {}
with open('/root/bag_supply_cost.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        bag_costs[int(row['bag_size_ml'])] = float(row['bag_supply_cost_usd'])

# Load delivery payments, resolve by alias mapping
payments = {}
with open('/root/delivery_payment.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        label = row['therapy_label'].strip().lower()
        if label in name_to_code:
            code = name_to_code[label]
            payments[code] = float(row['payment_per_delivery_per_patient_usd'])

# Load patient overrides - approved only, highest revision per therapy_code
patient_counts = {}
patient_revisions = {}
with open('/root/patient_overrides.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row['status'].strip().lower() != 'approved':
            continue
        code = row['therapy_code'].strip()
        if code not in in_scope:
            continue
        rev = int(row['revision'])
        if code not in patient_revisions or rev > patient_revisions[code]:
            patient_revisions[code] = rev
            patient_counts[code] = int(row['active_patients'])

# Constants
del_7 = 52
del_14 = 26
days_7 = 7
days_14 = 14
threshold = 15000

therapies_output = []
for code in sorted(in_scope.keys()):
    t = in_scope[code]
    active = patient_counts.get(code, 0)
    drug_cost_per_1000 = float(t['drug_cost_per_1000_mg_usd'])
    dose_per_day = float(t['dose_mg_per_day'])
    bag_ml = int(t['bag_size_ml'])
    bag_cost = bag_costs.get(bag_ml, 0.0)
    payment = payments.get(code, 0.0)

    # Annual drug cost = drug_cost_per_1000_mg * active * dose_mg_per_day * days_per_delivery * deliveries / 1000
    drug_7 = round(drug_cost_per_1000 * active * dose_per_day * days_7 * del_7 / 1000, 2)
    drug_14 = round(drug_cost_per_1000 * active * dose_per_day * days_14 * del_14 / 1000, 2)

    # Annual supply cost = bag_supply_cost * active * deliveries_per_year
    supply_7 = round(bag_cost * active * del_7, 2)
    supply_14 = round(bag_cost * active * del_14, 2)

    # Annual revenue = payment * active * deliveries
    rev_7 = round(payment * active * del_7, 2)
    rev_14 = round(payment * active * del_14, 2)

    margin_7 = round(rev_7 - drug_7 - supply_7, 2)
    margin_14 = round(rev_14 - drug_14 - supply_14, 2)
    diff = round(margin_14 - margin_7, 2)

    therapies_output.append({
        'therapy_code': code,
        'therapy_name': t['therapy_name'],
        'active_patients': active,
        'drug_cost_per_1000_mg_usd': drug_cost_per_1000,
        'dose_mg_per_day': dose_per_day,
        'bag_size_ml': bag_ml,
        'bag_supply_cost_usd': bag_cost,
        'payment_per_delivery_per_patient_usd': payment,
        'annual_drug_cost_7_day_usd': drug_7,
        'annual_drug_cost_14_day_usd': drug_14,
        'annual_supply_cost_7_day_usd': supply_7,
        'annual_supply_cost_14_day_usd': supply_14,
        'annual_revenue_7_day_usd': rev_7,
        'annual_revenue_14_day_usd': rev_14,
        'annual_margin_7_day_usd': margin_7,
        'annual_margin_14_day_usd': margin_14,
        'annual_margin_difference_14_minus_7_usd': diff
    })

total_margin_7 = round(sum(t['annual_margin_7_day_usd'] for t in therapies_output), 2)
total_margin_14 = round(sum(t['annual_margin_14_day_usd'] for t in therapies_output), 2)
total_diff = round(total_margin_14 - total_margin_7, 2)
abs_diff = round(abs(total_diff), 2)

if abs_diff < threshold:
    decision = 'move_to_14_day'
    justification = f'Absolute margin difference ${abs_diff} is below ${threshold} threshold; switching to 14-day delivery is acceptable.'
else:
    decision = 'keep_7_day'
    justification = f'Absolute margin difference ${abs_diff} exceeds ${threshold} threshold; keeping 7-day delivery is recommended.'

output = {
    'assumptions': {
        'deliveries_per_year_7_day': del_7,
        'deliveries_per_year_14_day': del_14,
        'days_per_delivery_7_day': days_7,
        'days_per_delivery_14_day': days_14,
        'switch_threshold_usd': threshold,
        'patient_override_rule': 'highest approved revision per therapy_code'
    },
    'therapies': therapies_output,
    'totals': {
        'total_annual_margin_7_day_usd': total_margin_7,
        'total_annual_margin_14_day_usd': total_margin_14,
        'total_annual_margin_difference_14_minus_7_usd': total_diff,
        'absolute_total_margin_difference_usd': abs_diff
    },
    'recommendation': {
        'decision': decision,
        'justification': justification
    }
}

with open('/root/infusion_batch_analysis.json', 'w') as f:
    json.dump(output, f, indent=2)

# Write summary markdown
lines = [
    '# Infusion Batch Analysis Summary',
    f'Total 7-day margin: ${total_margin_7:,.2f} USD',
    f'Total 14-day margin: ${total_margin_14:,.2f} USD',
    f'Absolute difference: ${abs_diff:,.2f} USD',
    f'Decision: {decision}',
    justification
]
with open('/root/infusion_batch_summary.md', 'w') as f:
    f.write('\n'.join(lines) + '\n')

print('Done.')
print(json.dumps(output, indent=2))
```

4. **Run the script**:
```bash
python /root/solve.py
```

5. **Verify output structure** — check that the JSON has the correct root keys and therapy keys:
```bash
python3 -c "
import json
with open('/root/infusion_batch_analysis.json') as f:
    d = json.load(f)
assert set(d.keys()) == {'assumptions','therapies','totals','recommendation'}, f'Root keys: {set(d.keys())}'
if d['therapies']:
    expected_keys = {'therapy_code','therapy_name','active_patients','drug_cost_per_1000_mg_usd','dose_mg_per_day','bag_size_ml','bag_supply_cost_usd','payment_per_delivery_per_patient_usd','annual_drug_cost_7_day_usd','annual_drug_cost_14_day_usd','annual_supply_cost_7_day_usd','annual_supply_cost_14_day_usd','annual_revenue_7_day_usd','annual_revenue_14_day_usd','annual_margin_7_day_usd','annual_margin_14_day_usd','annual_margin_difference_14_minus_7_usd'}
    actual = set(d['therapies'][0].keys())
    assert actual == expected_keys, f'Therapy keys mismatch: missing={expected_keys-actual}, extra={actual-expected_keys}'
print('Schema OK')
"
```

6. **Verify the summary markdown** has the required content:
```bash
python3 -c "
with open('/root/infusion_batch_summary.md') as f:
    text = f.read()
lines = [l for l in text.strip().split('\n') if l.strip()]
assert 4 <= len(lines) <= 8, f'Line count: {len(lines)}'
assert 'move_to_14_day' in text or 'keep_7_day' in text, 'Missing decision slug'
print('Summary OK')
"
```

7. **Run the test suite** if available:
```bash
cd / && python -m pytest tests/test_outputs.py -v 2>&1 | head -80
```

8. **If tests fail**, read the exact assertion error, re-read the relevant input files, fix `solve.py`, and re-run. Pay special attention to:
   - Exact key names in `assumptions` (e.g., `switch_threshold_usd`, `patient_override_rule`)
   - Whether `therapy_catalog.json` uses a top-level `therapies` array or is structured differently (adapt the loading code accordingly)
   - Whether `aliases` field exists and its format
   - Rounding: all currency to 2 decimal places
   - Sorting: therapies by `therapy_code` ascending
   - The supply cost formula: `bag_supply_cost_usd * active_patients * deliveries_per_year` (one bag per delivery per patient)
   - The decision threshold uses strict `<` (less than), not `<=`

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