# Task Instruction

Execute the following steps in order:

## Step 1: Inspect all input files

```bash
echo '=== compound_cost.csv ===' && cat /root/compound_cost.csv
echo '=== mailer_cost.csv ===' && cat /root/mailer_cost.csv
echo '=== base_payment.csv ===' && cat /root/base_payment.csv
echo '=== service_fee.csv ===' && cat /root/service_fee.csv
```

## Step 2: Create a Python script to compute and generate outputs

Create `/root/solve.py` with the following logic:

```python
import csv
import json
import os

# Read CSV helper
def read_csv(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))

compound = read_csv('/root/compound_cost.csv')
mailer = read_csv('/root/mailer_cost.csv')
base_pay = read_csv('/root/base_payment.csv')
service = read_csv('/root/service_fee.csv')

# Build lookup dicts by medication
def lookup(rows, key_col='medication'):
    d = {}
    for r in rows:
        d[r[key_col]] = r
    return d

compound_d = lookup(compound)
mailer_d = lookup(mailer)
base_d = lookup(base_pay)
service_d = lookup(service)

all_meds = sorted(compound_d.keys())  # alphabetical

patients = 150
fills_45 = 8
fills_90 = 4
doses_45 = 45
doses_90 = 90
threshold = 8500

medications = []
for med in all_meds:
    price_per_1000 = float(compound_d[med]['price_per_1000_doses_usd'])
    mformat = mailer_d[med]['mailer_format']
    mcost = float(mailer_d[med]['mailer_cost_usd'])
    bpay = float(base_d[med]['base_payment_per_fill_150_patients_usd'])
    sfee = float(service_d[med]['service_fee_per_fill_150_patients_usd'])
    total_payment_per_fill = bpay + sfee

    # Drug cost per fill = (doses_per_fill * patients * price_per_1000) / 1000
    drug_cost_per_fill_45 = (doses_45 * patients * price_per_1000) / 1000.0
    drug_cost_per_fill_90 = (doses_90 * patients * price_per_1000) / 1000.0

    annual_drug_45 = drug_cost_per_fill_45 * fills_45
    annual_drug_90 = drug_cost_per_fill_90 * fills_90

    # Mailer cost per fill = mcost * patients (per patient per fill)
    mailer_per_fill = mcost * patients
    annual_mailer_45 = mailer_per_fill * fills_45
    annual_mailer_90 = mailer_per_fill * fills_90

    # Payment
    annual_payment_45 = total_payment_per_fill * fills_45
    annual_payment_90 = total_payment_per_fill * fills_90

    # Margin
    margin_45 = annual_payment_45 - annual_drug_45 - annual_mailer_45
    margin_90 = annual_payment_90 - annual_drug_90 - annual_mailer_90
    diff = margin_90 - margin_45

    medications.append({
        'medication': med,
        'price_per_1000_doses_usd': round(price_per_1000, 2),
        'mailer_format': mformat,
        'mailer_cost_usd': round(mcost, 2),
        'base_payment_per_fill_150_patients_usd': round(bpay, 2),
        'service_fee_per_fill_150_patients_usd': round(sfee, 2),
        'total_payment_per_fill_150_patients_usd': round(total_payment_per_fill, 2),
        'annual_drug_cost_45_day_usd': round(annual_drug_45, 2),
        'annual_drug_cost_90_day_usd': round(annual_drug_90, 2),
        'annual_mailer_cost_45_day_usd': round(annual_mailer_45, 2),
        'annual_mailer_cost_90_day_usd': round(annual_mailer_90, 2),
        'annual_payment_45_day_usd': round(annual_payment_45, 2),
        'annual_payment_90_day_usd': round(annual_payment_90, 2),
        'annual_margin_45_day_usd': round(margin_45, 2),
        'annual_margin_90_day_usd': round(margin_90, 2),
        'annual_margin_difference_90_minus_45_usd': round(diff, 2)
    })

total_margin_45 = round(sum(m['annual_margin_45_day_usd'] for m in medications), 2)
total_margin_90 = round(sum(m['annual_margin_90_day_usd'] for m in medications), 2)
total_diff = round(sum(m['annual_margin_difference_90_minus_45_usd'] for m in medications), 2)
abs_diff = round(abs(total_diff), 2)

if abs_diff < threshold:
    decision = 'shift_to_90_day'
    justification = (f'The absolute total margin difference of ${abs_diff:,.2f} '
                     f'is below the ${threshold:,.2f} threshold, so switching to '
                     f'90-day fills is recommended.')
else:
    decision = 'keep_45_day'
    justification = (f'The absolute total margin difference of ${abs_diff:,.2f} '
                     f'exceeds the ${threshold:,.2f} threshold, so keeping '
                     f'45-day fills is recommended.')

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

print('JSON written.')
print(json.dumps(result, indent=2))

# Write markdown summary (4-8 non-empty lines)
lines = [
    '# Mailer Policy Analysis Summary',
    '',
    f'Total annual margin under 45-day fills: ${total_margin_45:,.2f} USD',
    f'Total annual margin under 90-day fills: ${total_margin_90:,.2f} USD',
    f'Absolute margin difference: ${abs_diff:,.2f} USD',
    f'Decision: {decision}',
    '',
    f'{justification}'
]

with open('/root/mailer_policy_summary.md', 'w') as f:
    f.write('\n'.join(lines) + '\n')

print('Markdown written.')
```

## Step 3: Run the script

```bash
python3 /root/solve.py
```

## Step 4: Validate outputs

```bash
echo '=== JSON output ===' && cat /root/mailer_policy_analysis.json
echo '=== Markdown output ===' && cat /root/mailer_policy_summary.md
```

Verify:
- JSON is valid and parseable
- `medications` array is sorted alphabetically by `medication`
- All currency values are rounded to 2 decimal places
- The markdown has 4-8 non-empty lines and includes: total 45-day margin, total 90-day margin, absolute difference, and the exact decision slug (`shift_to_90_day` or `keep_45_day`)
- The decision rule is correctly applied: if abs(total_difference) < 8500 → shift_to_90_day, else keep_45_day

## Step 5: Sanity-check the drug cost calculation

Manually verify one medication's drug cost to make sure the formula is correct:
- `annual_drug_cost_45_day = (45 * 150 * price_per_1000_doses / 1000) * 8`
- `annual_drug_cost_90_day = (90 * 150 * price_per_1000_doses / 1000) * 4`
- Both should equal `150 * 1 dose/day * 360 days * price_per_1000 / 1000` = same total doses (54,000 per year), so annual drug costs for 45-day and 90-day should be identical. Confirm this in the output.

If annual drug costs differ between 45 and 90 day for the same medication, something is wrong — re-check.

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