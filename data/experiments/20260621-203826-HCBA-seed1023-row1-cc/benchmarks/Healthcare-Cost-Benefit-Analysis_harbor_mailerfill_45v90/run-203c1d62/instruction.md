# Task Instruction

Execute the following steps in order:

## Step 1 – Inspect input files
```bash
cat /root/compound_cost.csv
cat /root/mailer_cost.csv
cat /root/base_payment.csv
cat /root/service_fee.csv
```
Note the column names and values exactly.

## Step 2 – Inspect the test suite
```bash
cat /root/test_output.py
```
Understand every assertion the verifier makes so the outputs satisfy them.

## Step 3 – Write and run the computation script

Create `/root/solve.py` with the following logic:

```python
import csv, json, math

def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))

compound = read_csv('/root/compound_cost.csv')
mailer = read_csv('/root/mailer_cost.csv')
base = read_csv('/root/base_payment.csv')
service = read_csv('/root/service_fee.csv')

# Build lookup dicts keyed by medication
compound_d = {r['medication']: float(r['price_per_1000_doses_usd']) for r in compound}
mailer_d = {r['medication']: r for r in mailer}  # has mailer_format and mailer_cost_usd
base_d = {r['medication']: float(r['base_payment_per_fill_150_patients_usd']) for r in base}
service_d = {r['medication']: float(r['service_fee_per_fill_150_patients_usd']) for r in service}

patients = 150
fills_45 = 8
fills_90 = 4
doses_45 = 45
doses_90 = 90
threshold = 8500

medications_list = sorted(compound_d.keys())

results = []
for med in medications_list:
    price_per_1000 = compound_d[med]
    mailer_format = mailer_d[med]['mailer_format']
    mailer_cost = float(mailer_d[med]['mailer_cost_usd'])
    base_pay = base_d[med]
    svc_fee = service_d[med]
    total_payment_per_fill = round(base_pay + svc_fee, 2)

    # Drug cost per fill = (doses_per_fill * patients * price_per_1000) / 1000
    drug_cost_per_fill_45 = (doses_45 * patients * price_per_1000) / 1000.0
    drug_cost_per_fill_90 = (doses_90 * patients * price_per_1000) / 1000.0

    annual_drug_45 = round(drug_cost_per_fill_45 * fills_45, 2)
    annual_drug_90 = round(drug_cost_per_fill_90 * fills_90, 2)

    # Mailer cost per fill = mailer_cost * patients
    mailer_per_fill = mailer_cost * patients
    annual_mailer_45 = round(mailer_per_fill * fills_45, 2)
    annual_mailer_90 = round(mailer_per_fill * fills_90, 2)

    # Payment
    annual_payment_45 = round(total_payment_per_fill * fills_45, 2)
    annual_payment_90 = round(total_payment_per_fill * fills_90, 2)

    # Margin
    margin_45 = round(annual_payment_45 - annual_drug_45 - annual_mailer_45, 2)
    margin_90 = round(annual_payment_90 - annual_drug_90 - annual_mailer_90, 2)
    diff = round(margin_90 - margin_45, 2)

    results.append({
        "medication": med,
        "price_per_1000_doses_usd": price_per_1000,
        "mailer_format": mailer_format,
        "mailer_cost_usd": mailer_cost,
        "base_payment_per_fill_150_patients_usd": base_pay,
        "service_fee_per_fill_150_patients_usd": svc_fee,
        "total_payment_per_fill_150_patients_usd": total_payment_per_fill,
        "annual_drug_cost_45_day_usd": annual_drug_45,
        "annual_drug_cost_90_day_usd": annual_drug_90,
        "annual_mailer_cost_45_day_usd": annual_mailer_45,
        "annual_mailer_cost_90_day_usd": annual_mailer_90,
        "annual_payment_45_day_usd": annual_payment_45,
        "annual_payment_90_day_usd": annual_payment_90,
        "annual_margin_45_day_usd": margin_45,
        "annual_margin_90_day_usd": margin_90,
        "annual_margin_difference_90_minus_45_usd": diff
    })

total_margin_45 = round(sum(r['annual_margin_45_day_usd'] for r in results), 2)
total_margin_90 = round(sum(r['annual_margin_90_day_usd'] for r in results), 2)
total_diff = round(total_margin_90 - total_margin_45, 2)
abs_diff = round(abs(total_diff), 2)

if abs_diff < 8500:
    decision = 'shift_to_90_day'
    justification = (f'The absolute total margin difference of ${abs_diff} '
                     f'is below the ${threshold} threshold, so switching to '
                     f'90-day fills is recommended.')
else:
    decision = 'keep_45_day'
    justification = (f'The absolute total margin difference of ${abs_diff} '
                     f'exceeds the ${threshold} threshold, so keeping '
                     f'45-day fills is recommended.')

output = {
    "assumptions": {
        "patients_per_medication": patients,
        "fills_per_year_45_day": fills_45,
        "fills_per_year_90_day": fills_90,
        "doses_per_fill_45_day": doses_45,
        "doses_per_fill_90_day": doses_90,
        "switch_threshold_usd": threshold
    },
    "medications": results,
    "totals": {
        "total_annual_margin_45_day_usd": total_margin_45,
        "total_annual_margin_90_day_usd": total_margin_90,
        "total_annual_margin_difference_90_minus_45_usd": total_diff,
        "absolute_total_margin_difference_usd": abs_diff
    },
    "recommendation": {
        "decision": decision,
        "justification": justification
    }
}

with open('/root/mailer_policy_analysis.json', 'w') as f:
    json.dump(output, f, indent=2)

# Markdown summary (4-8 non-empty lines)
lines = [
    '# Mailer Policy Analysis Summary',
    '',
    f'- Total 45-day annual margin: ${total_margin_45}',
    f'- Total 90-day annual margin: ${total_margin_90}',
    f'- Absolute margin difference: ${abs_diff}',
    f'- Decision: {decision}',
    '',
    justification
]
with open('/root/mailer_policy_summary.md', 'w') as f:
    f.write('\n'.join(lines) + '\n')

print('Done. Decision:', decision)
print('Total margin 45:', total_margin_45)
print('Total margin 90:', total_margin_90)
print('Abs diff:', abs_diff)
```

Run it:
```bash
python3 /root/solve.py
```

## Step 4 – Validate outputs
```bash
python3 -c "
import json
with open('/root/mailer_policy_analysis.json') as f:
    d = json.load(f)
assert 'assumptions' in d
assert 'medications' in d and len(d['medications']) > 0
assert 'totals' in d
assert 'recommendation' in d
for m in d['medications']:
    for k in ['medication','price_per_1000_doses_usd','mailer_format','mailer_cost_usd',
              'base_payment_per_fill_150_patients_usd','service_fee_per_fill_150_patients_usd',
              'total_payment_per_fill_150_patients_usd',
              'annual_drug_cost_45_day_usd','annual_drug_cost_90_day_usd',
              'annual_mailer_cost_45_day_usd','annual_mailer_cost_90_day_usd',
              'annual_payment_45_day_usd','annual_payment_90_day_usd',
              'annual_margin_45_day_usd','annual_margin_90_day_usd',
              'annual_margin_difference_90_minus_45_usd']:
        assert k in m, f'Missing key {k} in medication {m.get(\"medication\",\"?\")}'  
print('JSON schema OK')
meds = [m['medication'] for m in d['medications']]
assert meds == sorted(meds), 'Medications not sorted alphabetically'
print('Sort OK')

with open('/root/mailer_policy_summary.md') as f:
    lines = [l for l in f.read().strip().splitlines() if l.strip()]
assert 4 <= len(lines) <= 8, f'Expected 4-8 non-empty lines, got {len(lines)}'
print(f'Markdown has {len(lines)} non-empty lines – OK')
print('All checks passed')
"
```

## Step 5 – Run the test suite
```bash
cd /root && python3 -m pytest test_output.py -v
```

If any test fails, read the error carefully, fix the logic in `solve.py`, re-run it, and re-run the tests. Common pitfalls from cross-task experience:
- Drug cost formula: `(doses_per_fill * patients * price_per_1000) / 1000 * fills_per_year`. The total annual doses are the same for 45-day and 90-day (both = 1 dose/day × 360 days ≈ same total), so annual drug costs should be equal or very close. Double-check the math.
- Mailer cost is per patient per fill (mailer_cost_usd × patients × fills_per_year).
- Payment per fill is already for 150 patients (as the column name says), so do NOT multiply by patients again.
- Ensure all numeric values in JSON are floats rounded to 2 decimal places, not strings.

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
Task metadata: author_email=gpt54@example.com, author_name=GPT-5.4, category=financial-analysis, difficulty=medium, tags=[mailer-program, csv, json, revenue-merge, decision-analysis].
Verifier config: timeout_sec=900.0.