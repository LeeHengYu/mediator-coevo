# Task Instruction

Execute the following Python script to read the input CSVs, compute all required values, and produce both output files.

```python
import json, csv, os

# --- Read input files ---
def read_csv(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))

compound = read_csv('/root/compound_cost.csv')
mailer = read_csv('/root/mailer_cost.csv')
base_pay = read_csv('/root/base_payment.csv')
service = read_csv('/root/service_fee.csv')

# Print raw data for debugging
print('=== compound_cost.csv ===')
for r in compound: print(r)
print('=== mailer_cost.csv ===')
for r in mailer: print(r)
print('=== base_payment.csv ===')
for r in base_pay: print(r)
print('=== service_fee.csv ===')
for r in service: print(r)

# --- Build lookup dicts ---
# compound_cost: medication -> {price_per_1000_doses_usd, mailer_format}
med_info = {}
for r in compound:
    med = r['medication']
    med_info[med] = {
        'price_per_1000_doses_usd': float(r['price_per_1000_doses_usd']),
        'mailer_format': r['mailer_format']
    }

# mailer_cost: mailer_format -> mailer_cost_usd
mailer_lookup = {}
for r in mailer:
    mailer_lookup[r['mailer_format']] = float(r['mailer_cost_usd'])

# base_payment: medication -> base_payment_per_fill_150_patients_usd
base_lookup = {}
for r in base_pay:
    base_lookup[r['medication']] = float(r['base_payment_per_fill_150_patients_usd'])

# service_fee: medication -> service_fee_per_fill_150_patients_usd
service_lookup = {}
for r in service:
    service_lookup[r['medication']] = float(r['service_fee_per_fill_150_patients_usd'])

# --- Constants ---
patients = 150
fills_45 = 8
fills_90 = 4
doses_45 = 45
doses_90 = 90
threshold = 8500

# --- Compute per-medication ---
medications = []
for med in sorted(med_info.keys()):
    info = med_info[med]
    price_1000 = info['price_per_1000_doses_usd']
    mformat = info['mailer_format']
    mcost = mailer_lookup[mformat]
    bpay = base_lookup[med]
    sfee = service_lookup[med]
    total_payment_per_fill = round(bpay + sfee, 2)

    # Annual drug cost = (doses_per_fill * fills_per_year * patients) * price_per_1000_doses / 1000
    annual_drug_45 = round((doses_45 * fills_45 * patients) * price_1000 / 1000, 2)
    annual_drug_90 = round((doses_90 * fills_90 * patients) * price_1000 / 1000, 2)

    # Annual mailer cost = mailer_cost * patients * fills_per_year
    annual_mailer_45 = round(mcost * patients * fills_45, 2)
    annual_mailer_90 = round(mcost * patients * fills_90, 2)

    # Annual payment = total_payment_per_fill * fills_per_year
    annual_pay_45 = round(total_payment_per_fill * fills_45, 2)
    annual_pay_90 = round(total_payment_per_fill * fills_90, 2)

    # Annual margin = annual_payment - annual_drug_cost - annual_mailer_cost
    margin_45 = round(annual_pay_45 - annual_drug_45 - annual_mailer_45, 2)
    margin_90 = round(annual_pay_90 - annual_drug_90 - annual_mailer_90, 2)

    diff = round(margin_90 - margin_45, 2)

    medications.append({
        'medication': med,
        'price_per_1000_doses_usd': price_1000,
        'mailer_format': mformat,
        'mailer_cost_usd': mcost,
        'base_payment_per_fill_150_patients_usd': bpay,
        'service_fee_per_fill_150_patients_usd': sfee,
        'total_payment_per_fill_150_patients_usd': total_payment_per_fill,
        'annual_drug_cost_45_day_usd': annual_drug_45,
        'annual_drug_cost_90_day_usd': annual_drug_90,
        'annual_mailer_cost_45_day_usd': annual_mailer_45,
        'annual_mailer_cost_90_day_usd': annual_mailer_90,
        'annual_payment_45_day_usd': annual_pay_45,
        'annual_payment_90_day_usd': annual_pay_90,
        'annual_margin_45_day_usd': margin_45,
        'annual_margin_90_day_usd': margin_90,
        'annual_margin_difference_90_minus_45_usd': diff
    })

# --- Totals ---
total_margin_45 = round(sum(m['annual_margin_45_day_usd'] for m in medications), 2)
total_margin_90 = round(sum(m['annual_margin_90_day_usd'] for m in medications), 2)
total_diff = round(total_margin_90 - total_margin_45, 2)
abs_diff = round(abs(total_diff), 2)

# --- Decision ---
if abs_diff < 8500:
    decision = 'shift_to_90_day'
    justification = (f'The absolute total margin difference of ${abs_diff} is below '
                     f'the ${threshold} threshold, so shifting to 90-day fills is recommended '
                     f'to reduce operational complexity with minimal financial impact.')
else:
    decision = 'keep_45_day'
    justification = (f'The absolute total margin difference of ${abs_diff} exceeds '
                     f'the ${threshold} threshold, so keeping 45-day fills is recommended '
                     f'to preserve the current margin structure.')

# --- Build JSON output ---
output = {
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
    json.dump(output, f, indent=2)
print('\nJSON written.')

# --- Build Markdown summary ---
lines = [
    '# Mailer Policy Analysis Summary',
    '',
    f'Total annual margin under 45-day fills: ${total_margin_45:,.2f} USD',
    f'Total annual margin under 90-day fills: ${total_margin_90:,.2f} USD',
    f'Absolute margin difference: ${abs_diff:,.2f} USD',
    f'Recommendation: {decision}',
    '',
    f'{justification}'
]

with open('/root/mailer_policy_summary.md', 'w') as f:
    f.write('\n'.join(lines) + '\n')
print('Markdown written.')

# --- Verify outputs ---
with open('/root/mailer_policy_analysis.json') as f:
    check = json.load(f)
    print(f"\nJSON medications count: {len(check['medications'])}")
    print(f"Medications sorted: {[m['medication'] for m in check['medications']]}")
    print(f"Totals: {check['totals']}")
    print(f"Decision: {check['recommendation']['decision']}")

with open('/root/mailer_policy_summary.md') as f:
    md = f.read()
    non_empty = [l for l in md.split('\n') if l.strip()]
    print(f"\nMarkdown non-empty lines: {len(non_empty)}")
    print(md)
```

After the script completes, verify both output files exist and look correct. If there are any errors reading the CSVs (e.g., unexpected column names), inspect the CSV files first, adjust the column name references, and re-run.

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