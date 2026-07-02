# Task Instruction

## Task: Healthcare Cost-Benefit Analysis — 45-day vs 90-day Mailer Fills

### Step 1: Inspect all input files

Read and display the contents of each input CSV:
```
cat /root/compound_cost.csv
cat /root/mailer_cost.csv
cat /root/base_payment.csv
cat /root/service_fee.csv
```

Note the column names and values carefully. Identify how medications are keyed across files (likely a `medication` column) and how `mailer_format` links `compound_cost.csv` to `mailer_cost.csv`.

### Step 2: Write and run a Python script to produce both output files

Create `/root/solve.py` with the following logic:

```python
import csv, json, math

# Load CSVs
def load_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))

compound = load_csv('/root/compound_cost.csv')
mailer = load_csv('/root/mailer_cost.csv')
base_pay = load_csv('/root/base_payment.csv')
service = load_csv('/root/service_fee.csv')

# Build lookup dicts
# mailer_cost.csv is keyed by mailer_format
mailer_lookup = {row['mailer_format'].strip(): float(row['mailer_cost_usd']) for row in mailer}

# base_payment and service_fee are keyed by medication
base_lookup = {row['medication'].strip(): float(row['base_payment_per_fill_150_patients_usd']) for row in base_pay}
service_lookup = {row['medication'].strip(): float(row['service_fee_per_fill_150_patients_usd']) for row in service}

# Constants
patients = 150
fills_45 = 8
fills_90 = 4
doses_45 = 45
doses_90 = 90
threshold = 8500

medications = []
for row in compound:
    med = row['medication'].strip()
    price_per_1000 = float(row['price_per_1000_doses_usd'])
    fmt = row['mailer_format'].strip()
    mailer_cost = mailer_lookup[fmt]
    base_p = base_lookup[med]
    svc_fee = service_lookup[med]
    total_payment_per_fill = base_p + svc_fee

    # Drug cost per fill = (doses_per_fill * patients * price_per_1000) / 1000
    drug_cost_45 = (doses_45 * patients * price_per_1000) / 1000.0
    drug_cost_90 = (doses_90 * patients * price_per_1000) / 1000.0

    annual_drug_cost_45 = drug_cost_45 * fills_45
    annual_drug_cost_90 = drug_cost_90 * fills_90

    # Mailer cost per fill = mailer_cost * patients (per patient per fill)
    mailer_per_fill = mailer_cost * patients
    annual_mailer_45 = mailer_per_fill * fills_45
    annual_mailer_90 = mailer_per_fill * fills_90

    # Payment
    annual_payment_45 = total_payment_per_fill * fills_45
    annual_payment_90 = total_payment_per_fill * fills_90

    # Margin
    margin_45 = annual_payment_45 - annual_drug_cost_45 - annual_mailer_45
    margin_90 = annual_payment_90 - annual_drug_cost_90 - annual_mailer_90
    diff = margin_90 - margin_45

    medications.append({
        'medication': med,
        'price_per_1000_doses_usd': round(price_per_1000, 2),
        'mailer_format': fmt,
        'mailer_cost_usd': round(mailer_cost, 2),
        'base_payment_per_fill_150_patients_usd': round(base_p, 2),
        'service_fee_per_fill_150_patients_usd': round(svc_fee, 2),
        'total_payment_per_fill_150_patients_usd': round(total_payment_per_fill, 2),
        'annual_drug_cost_45_day_usd': round(annual_drug_cost_45, 2),
        'annual_drug_cost_90_day_usd': round(annual_drug_cost_90, 2),
        'annual_mailer_cost_45_day_usd': round(annual_mailer_45, 2),
        'annual_mailer_cost_90_day_usd': round(annual_mailer_90, 2),
        'annual_payment_45_day_usd': round(annual_payment_45, 2),
        'annual_payment_90_day_usd': round(annual_payment_90, 2),
        'annual_margin_45_day_usd': round(margin_45, 2),
        'annual_margin_90_day_usd': round(margin_90, 2),
        'annual_margin_difference_90_minus_45_usd': round(diff, 2)
    })

# Sort alphabetically by medication
medications.sort(key=lambda x: x['medication'])

total_margin_45 = round(sum(m['annual_margin_45_day_usd'] for m in medications), 2)
total_margin_90 = round(sum(m['annual_margin_90_day_usd'] for m in medications), 2)
total_diff = round(sum(m['annual_margin_difference_90_minus_45_usd'] for m in medications), 2)
abs_diff = round(abs(total_diff), 2)

if abs_diff < 8500:
    decision = 'shift_to_90_day'
    justification = f'The absolute total margin difference of ${abs_diff} is below the ${threshold} threshold, so shifting to 90-day fills is recommended.'
else:
    decision = 'keep_45_day'
    justification = f'The absolute total margin difference of ${abs_diff} meets or exceeds the ${threshold} threshold, so keeping 45-day fills is recommended.'

result = {
    'assumptions': {
        'patients_per_medication': patients,
        'fills_per_year_45_day': fills_45,
        'fills_per_year_90_day': fills_90,
        'doses_per_fill_45_day': doses_45,
        'doses_per_fill_90_day': doses_90,
        'switch_threshold_usd': threshold
    },
    'medications': medications,
    'totals': {
        'total_annual_margin_45_day_usd': total_margin_45,
        'total_annual_margin_90_day_usd': total_margin_90,
        'total_annual_margin_difference_90_minus_45_usd': total_diff,
        'absolute_total_margin_difference_usd': abs_diff
    },
    'recommendation': {
        'decision': decision,
        'justification': justification
    }
}

with open('/root/mailer_policy_analysis.json', 'w') as f:
    json.dump(result, f, indent=2)

# Write summary markdown (4-8 non-empty lines)
with open('/root/mailer_policy_summary.md', 'w') as f:
    f.write('# Mailer Policy Analysis Summary\n')
    f.write(f'\n')
    f.write(f'Total 45-day annual margin: ${total_margin_45:,.2f} USD\n')
    f.write(f'Total 90-day annual margin: ${total_margin_90:,.2f} USD\n')
    f.write(f'Absolute margin difference: ${abs_diff:,.2f} USD\n')
    f.write(f'\n')
    f.write(f'Decision: {decision}\n')
    f.write(f'Justification: {justification}\n')

print('Done. Files written.')
print(json.dumps(result, indent=2))
```

Run the script:
```
python3 /root/solve.py
```

### Step 3: Validate outputs

1. **Verify JSON is valid and matches schema:**
```
python3 -c "
import json
with open('/root/mailer_policy_analysis.json') as f:
    d = json.load(f)
assert 'assumptions' in d
assert 'medications' in d and isinstance(d['medications'], list)
assert 'totals' in d
assert 'recommendation' in d
assert d['recommendation']['decision'] in ('shift_to_90_day', 'keep_45_day')
# Check medications sorted
meds = [m['medication'] for m in d['medications']]
assert meds == sorted(meds), f'Not sorted: {meds}'
# Check all required fields in each medication
required = ['medication','price_per_1000_doses_usd','mailer_format','mailer_cost_usd','base_payment_per_fill_150_patients_usd','service_fee_per_fill_150_patients_usd','total_payment_per_fill_150_patients_usd','annual_drug_cost_45_day_usd','annual_drug_cost_90_day_usd','annual_mailer_cost_45_day_usd','annual_mailer_cost_90_day_usd','annual_payment_45_day_usd','annual_payment_90_day_usd','annual_margin_45_day_usd','annual_margin_90_day_usd','annual_margin_difference_90_minus_45_usd']
for m in d['medications']:
    for k in required:
        assert k in m, f'Missing {k} in {m[\"medication\"]}'
print('JSON validation passed.')
print(f'Medications: {len(d[\"medications\"])}')
print(f'Decision: {d[\"recommendation\"][\"decision\"]}')
print(f'Total diff: {d[\"totals\"][\"total_annual_margin_difference_90_minus_45_usd\"]}')
print(f'Abs diff: {d[\"totals\"][\"absolute_total_margin_difference_usd\"]}')
"
```

2. **Verify markdown summary:**
```
cat /root/mailer_policy_summary.md
```

Check it has 4-8 non-empty lines and includes: total 45-day margin, total 90-day margin, absolute difference, and the exact decision slug (`shift_to_90_day` or `keep_45_day`).

3. **Verify annual drug costs are consistent:**
   - Both 45-day and 90-day models should produce the same total annual doses per patient (360 doses = 45×8 = 90×4). So `annual_drug_cost_45_day` should equal `annual_drug_cost_90_day` for each medication. Confirm this:
```
python3 -c "
import json
with open('/root/mailer_policy_analysis.json') as f:
    d = json.load(f)
for m in d['medications']:
    assert m['annual_drug_cost_45_day_usd'] == m['annual_drug_cost_90_day_usd'], f'{m[\"medication\"]}: drug costs differ'
print('Drug cost consistency check passed.')
"
```

Since drug costs are the same for both models, the margin difference comes purely from payment and mailer cost differences. This is a good sanity check.

### Important Notes
- The `mailer_cost_usd` is per patient per fill (as stated: "per patient per fill, matched by mailer_format"). So annual mailer cost = mailer_cost_usd × patients × fills_per_year.
- The `base_payment_per_fill_150_patients_usd` and `service_fee_per_fill_150_patients_usd` are already for 150 patients (as the column name indicates). So annual payment = (base + service) × fills_per_year. Do NOT multiply by patients again.
- Drug cost: `price_per_1000_doses_usd` is per 1000 doses. For 150 patients × doses_per_fill doses per fill × fills_per_year fills: annual_drug_cost = (doses_per_fill × patients × fills_per_year × price_per_1000) / 1000. Equivalently: per-fill drug cost = (doses_per_fill × 150 × price_per_1000) / 1000, then multiply by fills.
- After inspecting the CSV files in Step 1, if column names differ slightly from what's assumed, adjust the script accordingly before running.

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