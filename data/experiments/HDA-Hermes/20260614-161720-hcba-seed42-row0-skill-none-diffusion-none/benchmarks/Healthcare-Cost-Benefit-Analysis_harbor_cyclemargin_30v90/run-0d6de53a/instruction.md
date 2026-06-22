# Task Instruction

## Task: Healthcare Cost-Benefit Analysis — 30-day vs 90-day Refill Cycle Margin Comparison

### Step 1: Inspect Input Files
Read and display the contents of all three input CSV files:
```
cat /root/acquisition_cost.csv
cat /root/packaging_cost.csv
cat /root/reimbursement.csv
```
Understand the column names, therapy names, and how they join together. The key join columns are likely `therapy` and `canister_size_units`.

### Step 2: Write a Python script to perform the analysis

Create `/root/solve.py` with the following logic:

```python
import csv, json, math

# Read CSVs
def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))

acq = read_csv('/root/acquisition_cost.csv')
pkg = read_csv('/root/packaging_cost.csv')
reimb = read_csv('/root/reimbursement.csv')

# Build lookup dicts
# acquisition_cost.csv should have: therapy, price_per_1000_doses_usd
# packaging_cost.csv should have: canister_size_units, packaging_cost_usd (and possibly therapy)
# reimbursement.csv should have: therapy, reimbursement_per_fill_240_patients_usd

# We need to join these. Inspect columns carefully.
# The packaging_cost is matched by canister_size_units, so acquisition_cost.csv likely has canister_size_units too.

patients = 240
fills_30 = 12
fills_90 = 4
doses_30 = 60
doses_90 = 180
threshold = 12000

# Build packaging lookup by canister_size_units
pkg_lookup = {}
for row in pkg:
    key = int(row['canister_size_units'])
    pkg_lookup[key] = float(row['packaging_cost_usd'])

# Build reimbursement lookup by therapy
reimb_lookup = {}
for row in reimb:
    reimb_lookup[row['therapy'].strip()] = float(row['reimbursement_per_fill_240_patients_usd'])

therapies = []
for row in acq:
    therapy_name = row['therapy'].strip()
    price_per_1000 = float(row['price_per_1000_doses_usd'])
    canister_size = int(row['canister_size_units'])
    packaging_cost = pkg_lookup[canister_size]
    reimb_per_fill = reimb_lookup[therapy_name]

    # Drug cost: total doses per year = patients * 2 inhalations/day * 365 days
    # But actually, drug cost per fill = (doses_per_fill / 1000) * price_per_1000 per patient
    # Annual drug cost = drug_cost_per_fill * patients * fills_per_year
    # doses_per_fill for 30-day = 60, for 90-day = 180
    
    # Drug cost per fill per patient = (doses_per_fill / 1000) * price_per_1000
    # Annual drug cost = drug_cost_per_fill_per_patient * patients * fills_per_year
    
    annual_drug_cost_30 = (doses_30 / 1000.0) * price_per_1000 * patients * fills_30
    annual_drug_cost_90 = (doses_90 / 1000.0) * price_per_1000 * patients * fills_90
    
    # Packaging cost per patient per fill
    annual_pkg_30 = packaging_cost * patients * fills_30
    annual_pkg_90 = packaging_cost * patients * fills_90
    
    # Reimbursement per fill for 240 patients (already for all patients)
    annual_reimb_30 = reimb_per_fill * fills_30
    annual_reimb_90 = reimb_per_fill * fills_90
    
    margin_30 = annual_reimb_30 - annual_drug_cost_30 - annual_pkg_30
    margin_90 = annual_reimb_90 - annual_drug_cost_90 - annual_pkg_90
    diff = margin_90 - margin_30
    
    therapies.append({
        'therapy': therapy_name,
        'price_per_1000_doses_usd': round(price_per_1000, 2),
        'canister_size_units': canister_size,
        'packaging_cost_usd': round(packaging_cost, 2),
        'reimbursement_per_fill_240_patients_usd': round(reimb_per_fill, 2),
        'annual_drug_cost_30_day_usd': round(annual_drug_cost_30, 2),
        'annual_drug_cost_90_day_usd': round(annual_drug_cost_90, 2),
        'annual_packaging_cost_30_day_usd': round(annual_pkg_30, 2),
        'annual_packaging_cost_90_day_usd': round(annual_pkg_90, 2),
        'annual_reimbursement_30_day_usd': round(annual_reimb_30, 2),
        'annual_reimbursement_90_day_usd': round(annual_reimb_90, 2),
        'annual_margin_30_day_usd': round(margin_30, 2),
        'annual_margin_90_day_usd': round(margin_90, 2),
        'annual_margin_difference_90_minus_30_usd': round(diff, 2)
    })

# Sort alphabetically by therapy
therapies.sort(key=lambda x: x['therapy'])

total_margin_30 = round(sum(t['annual_margin_30_day_usd'] for t in therapies), 2)
total_margin_90 = round(sum(t['annual_margin_90_day_usd'] for t in therapies), 2)
total_diff = round(sum(t['annual_margin_difference_90_minus_30_usd'] for t in therapies), 2)
abs_diff = round(abs(total_diff), 2)

if abs_diff < 12000:
    decision = 'adopt_90_day'
    justification = f'The absolute total margin difference of ${abs_diff} is below the ${threshold} threshold, so switching to 90-day fills is recommended.'
else:
    decision = 'keep_30_day'
    justification = f'The absolute total margin difference of ${abs_diff} exceeds the ${threshold} threshold, so keeping 30-day fills is recommended.'

result = {
    'assumptions': {
        'patients_per_therapy': patients,
        'fills_per_year_30_day': fills_30,
        'fills_per_year_90_day': fills_90,
        'doses_per_fill_30_day': doses_30,
        'doses_per_fill_90_day': doses_90,
        'switch_threshold_usd': threshold
    },
    'therapies': therapies,
    'totals': {
        'total_annual_margin_30_day_usd': total_margin_30,
        'total_annual_margin_90_day_usd': total_margin_90,
        'total_annual_margin_difference_90_minus_30_usd': total_diff,
        'absolute_total_margin_difference_usd': abs_diff
    },
    'recommendation': {
        'decision': decision,
        'justification': justification
    }
}

with open('/root/cycle_margin_analysis.json', 'w') as f:
    json.dump(result, f, indent=2)

# Write summary markdown
lines = [
    '# Refill Cycle Margin Analysis Summary',
    f'Total 30-day annual margin: ${total_margin_30:,.2f} USD',
    f'Total 90-day annual margin: ${total_margin_90:,.2f} USD',
    f'Total margin difference (90-day minus 30-day): ${total_diff:,.2f} USD',
    f'Absolute difference: ${abs_diff:,.2f} USD',
    f'Threshold for switching: ${threshold:,.2f} USD',
    f'Decision: {decision}',
    f'Justification: {justification}'
]

with open('/root/cycle_margin_summary.md', 'w') as f:
    f.write('\n'.join(lines) + '\n')

print('Done.')
print(json.dumps(result, indent=2))
```

### Step 3: Run the script
```
python3 /root/solve.py
```

If there are errors (e.g., column name mismatches), inspect the CSV headers again and fix the script accordingly.

### Step 4: Validate outputs

1. Verify JSON is valid and matches schema:
```
python3 -c "
import json
with open('/root/cycle_margin_analysis.json') as f:
    d = json.load(f)
assert 'assumptions' in d
assert 'therapies' in d and isinstance(d['therapies'], list)
assert 'totals' in d
assert 'recommendation' in d
assert d['recommendation']['decision'] in ('adopt_90_day', 'keep_30_day')
# Check therapies sorted alphabetically
names = [t['therapy'] for t in d['therapies']]
assert names == sorted(names), f'Not sorted: {names}'
# Check all required keys exist in each therapy
required = ['therapy','price_per_1000_doses_usd','canister_size_units','packaging_cost_usd','reimbursement_per_fill_240_patients_usd','annual_drug_cost_30_day_usd','annual_drug_cost_90_day_usd','annual_packaging_cost_30_day_usd','annual_packaging_cost_90_day_usd','annual_reimbursement_30_day_usd','annual_reimbursement_90_day_usd','annual_margin_30_day_usd','annual_margin_90_day_usd','annual_margin_difference_90_minus_30_usd']
for t in d['therapies']:
    for k in required:
        assert k in t, f'Missing key {k} in therapy {t.get(\"therapy\", \"unknown\")}'
print('JSON validation passed')
"
```

2. Verify markdown:
```
cat /root/cycle_margin_summary.md
```
Confirm it has 4-8 non-empty lines, includes total 30-day margin, total 90-day margin, absolute difference, and the exact decision slug.

3. Verify drug cost is the same for both models (since total annual doses = patients × 2 × 365 regardless of fill cycle). The drug costs for 30-day and 90-day should be identical: `(60/1000)*price*240*12 = (180/1000)*price*240*4` — both equal `(price/1000)*240*2*365... wait, let me verify: 60*12=720 doses/patient/year vs 180*4=720 doses/patient/year. Yes, both are 720 doses/patient/year. So annual_drug_cost_30 should equal annual_drug_cost_90 for each therapy. The margin difference comes only from packaging and reimbursement differences. Verify this in the output.

### Important Notes
- The drug cost formula: `(doses_per_fill / 1000) * price_per_1000_doses_usd * patients * fills_per_year`. Since 60×12 = 180×4 = 720, drug costs are identical across models.
- Packaging cost per fill is per patient per fill. Annual = packaging_cost_usd × 240 patients × fills_per_year. 30-day has 12 fills, 90-day has 4 fills, so packaging is 3x higher for 30-day.
- Reimbursement per fill is already for 240 patients. Annual = reimbursement_per_fill × fills_per_year. 30-day has 12 fills, 90-day has 4 fills, so reimbursement is 3x higher for 30-day.
- The margin difference per therapy = (reimb×4 - drug90 - pkg×240×4) - (reimb×12 - drug30 - pkg×240×12) = reimb×(4-12) - 0 - pkg×240×(4-12) = -8×reimb + 8×240×pkg = 8×(240×pkg - reimb)
- Decision: if |total_difference| < 12000 → adopt_90_day, else keep_30_day.

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
Task metadata: author_email=gpt54@example.com, author_name=GPT-5.4, category=financial-analysis, difficulty=medium, tags=[healthcare, unit-economics, csv, json, decision-analysis].
Verifier config: timeout_sec=900.0.