# Task Instruction

Execute the following steps in order:

## Step 1: Inspect all input files

Read and display the contents of:
- `/root/compound_cost.csv`
- `/root/mailer_cost.csv`
- `/root/base_payment.csv`
- `/root/service_fee.csv`

## Step 2: Create a Python script and run it

Create `/root/solve.py` with the following logic:

```python
import csv
import json
import os

# Read CSVs
def read_csv(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))

compound_cost = read_csv('/root/compound_cost.csv')
mailer_cost_data = read_csv('/root/mailer_cost.csv')
base_payment = read_csv('/root/base_payment.csv')
service_fee = read_csv('/root/service_fee.csv')

# Build lookup dicts
# mailer_cost.csv: keyed by mailer_format
mailer_cost_lookup = {row['mailer_format']: float(row['mailer_cost_usd']) for row in mailer_cost_data}

# base_payment.csv and service_fee.csv: keyed by medication
base_payment_lookup = {row['medication']: float(row['base_payment_per_fill_150_patients_usd']) for row in base_payment}
service_fee_lookup = {row['medication']: float(row['service_fee_per_fill_150_patients_usd']) for row in service_fee}

patients = 150
fills_45 = 8
fills_90 = 4
doses_per_fill_45 = 45
doses_per_fill_90 = 90
threshold = 8500

medications = []

for row in compound_cost:
    med = row['medication']
    price_per_1000 = float(row['price_per_1000_doses_usd'])
    mailer_format = row['mailer_format']
    mailer_cost_usd = mailer_cost_lookup[mailer_format]
    base_pay = base_payment_lookup[med]
    service = service_fee_lookup[med]
    total_payment_per_fill = base_pay + service

    # Drug cost per fill = (doses_per_fill / 1000) * price_per_1000 * patients
    # Annual drug cost = drug_cost_per_fill * fills_per_year
    annual_drug_cost_45 = round((doses_per_fill_45 / 1000.0) * price_per_1000 * patients * fills_45, 2)
    annual_drug_cost_90 = round((doses_per_fill_90 / 1000.0) * price_per_1000 * patients * fills_90, 2)

    # Mailer cost per fill = mailer_cost_usd * patients
    # Annual mailer cost = mailer_cost_per_fill * fills_per_year
    annual_mailer_cost_45 = round(mailer_cost_usd * patients * fills_45, 2)
    annual_mailer_cost_90 = round(mailer_cost_usd * patients * fills_90, 2)

    # Annual payment = total_payment_per_fill * fills_per_year
    annual_payment_45 = round(total_payment_per_fill * fills_45, 2)
    annual_payment_90 = round(total_payment_per_fill * fills_90, 2)

    # Annual margin = annual_payment - annual_drug_cost - annual_mailer_cost
    annual_margin_45 = round(annual_payment_45 - annual_drug_cost_45 - annual_mailer_cost_45, 2)
    annual_margin_90 = round(annual_payment_90 - annual_drug_cost_90 - annual_mailer_cost_90, 2)

    diff = round(annual_margin_90 - annual_margin_45, 2)

    medications.append({
        'medication': med,
        'price_per_1000_doses_usd': price_per_1000,
        'mailer_format': mailer_format,
        'mailer_cost_usd': mailer_cost_usd,
        'base_payment_per_fill_150_patients_usd': base_pay,
        'service_fee_per_fill_150_patients_usd': service,
        'total_payment_per_fill_150_patients_usd': round(total_payment_per_fill, 2),
        'annual_drug_cost_45_day_usd': annual_drug_cost_45,
        'annual_drug_cost_90_day_usd': annual_drug_cost_90,
        'annual_mailer_cost_45_day_usd': annual_mailer_cost_45,
        'annual_mailer_cost_90_day_usd': annual_mailer_cost_90,
        'annual_payment_45_day_usd': annual_payment_45,
        'annual_payment_90_day_usd': annual_payment_90,
        'annual_margin_45_day_usd': annual_margin_45,
        'annual_margin_90_day_usd': annual_margin_90,
        'annual_margin_difference_90_minus_45_usd': diff
    })

# Sort alphabetically by medication
medications.sort(key=lambda x: x['medication'])

total_margin_45 = round(sum(m['annual_margin_45_day_usd'] for m in medications), 2)
total_margin_90 = round(sum(m['annual_margin_90_day_usd'] for m in medications), 2)
total_diff = round(total_margin_90 - total_margin_45, 2)
abs_diff = round(abs(total_diff), 2)

if abs_diff < 8500:
    decision = 'shift_to_90_day'
    justification = f'The absolute total margin difference of ${abs_diff} is below the ${threshold} threshold, so shifting to 90-day fills is recommended.'
else:
    decision = 'keep_45_day'
    justification = f'The absolute total margin difference of ${abs_diff} meets or exceeds the ${threshold} threshold, so keeping 45-day fills is recommended.'

result = {
    'assumptions': {
        'patients_per_medication': 150,
        'fills_per_year_45_day': 8,
        'fills_per_year_90_day': 4,
        'doses_per_fill_45_day': 45,
        'doses_per_fill_90_day': 90,
        'switch_threshold_usd': 8500
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

print('JSON written.')
print(f'Total 45-day margin: ${total_margin_45}')
print(f'Total 90-day margin: ${total_margin_90}')
print(f'Absolute difference: ${abs_diff}')
print(f'Decision: {decision}')

# Write summary markdown
lines = [
    '# Mailer Policy Analysis Summary',
    '',
    f'Total annual margin under 45-day fills: ${total_margin_45:,.2f} USD',
    f'Total annual margin under 90-day fills: ${total_margin_90:,.2f} USD',
    f'Absolute margin difference (90-day minus 45-day): ${abs_diff:,.2f} USD',
    f'Decision threshold: ${threshold:,.2f} USD',
    f'Final decision: {decision}',
    f'Justification: {justification}'
]

with open('/root/mailer_policy_summary.md', 'w') as f:
    f.write('\n'.join(lines) + '\n')

print('Summary written.')
```

Run: `python3 /root/solve.py`

## Step 3: Validate outputs

1. Read `/root/mailer_policy_analysis.json` and confirm:
   - It parses as valid JSON.
   - The `assumptions` block has the exact keys and values specified.
   - The `medications` array is sorted alphabetically by `medication`.
   - Each medication object has all 16 required fields.
   - All currency values are rounded to 2 decimals.
   - `totals` has all 4 required fields.
   - `recommendation.decision` is one of `shift_to_90_day` or `keep_45_day`.
   - The decision rule is correctly applied: if `absolute_total_margin_difference_usd < 8500` then `shift_to_90_day`, otherwise `keep_45_day`.

2. Read `/root/mailer_policy_summary.md` and confirm:
   - It has 4-8 non-empty lines.
   - It includes the total 45-day margin, total 90-day margin, absolute difference, and the exact decision slug.

3. Spot-check one medication manually:
   - Pick the first medication alphabetically.
   - Verify: `annual_drug_cost_45 = (45/1000) * price_per_1000 * 150 * 8`
   - Verify: `annual_drug_cost_90 = (90/1000) * price_per_1000 * 150 * 4`
   - Verify: `annual_mailer_cost_45 = mailer_cost_usd * 150 * 8`
   - Verify: `annual_mailer_cost_90 = mailer_cost_usd * 150 * 4`
   - Verify: `annual_payment_45 = (base_payment + service_fee) * 8`
   - Verify: `annual_payment_90 = (base_payment + service_fee) * 4`
   - Verify margins and difference.

If any validation fails, fix the issue and re-run.

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