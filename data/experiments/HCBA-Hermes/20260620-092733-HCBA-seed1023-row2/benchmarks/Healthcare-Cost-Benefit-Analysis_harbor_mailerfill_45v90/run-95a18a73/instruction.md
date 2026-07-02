# Task Instruction

Execute the following Python script to read the input CSVs, compute the 45-day vs 90-day fill analysis, and produce both output files.

```python
import csv
import json
import os

# Read CSV helper
def read_csv(path):
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        return list(reader)

# Load input files
compound_rows = read_csv('/root/compound_cost.csv')
mailer_rows = read_csv('/root/mailer_cost.csv')
base_payment_rows = read_csv('/root/base_payment.csv')
service_fee_rows = read_csv('/root/service_fee.csv')

# Print raw data for debugging
print('=== compound_cost.csv ===')
for r in compound_rows: print(r)
print('=== mailer_cost.csv ===')
for r in mailer_rows: print(r)
print('=== base_payment.csv ===')
for r in base_payment_rows: print(r)
print('=== service_fee.csv ===')
for r in service_fee_rows: print(r)

# Build lookup dicts
compound_map = {r['medication']: r for r in compound_rows}

# mailer_cost keyed by mailer_format
mailer_map = {r['mailer_format']: float(r['mailer_cost_usd']) for r in mailer_rows}

# base_payment and service_fee keyed by medication
base_payment_map = {r['medication']: float(r['base_payment_per_fill_150_patients_usd']) for r in base_payment_rows}
service_fee_map = {r['medication']: float(r['service_fee_per_fill_150_patients_usd']) for r in service_fee_rows}

# Constants
patients = 150
fills_45 = 8
fills_90 = 4
doses_per_fill_45 = 45
doses_per_fill_90 = 90
threshold = 8500

medications_list = []

# Get all medication names from compound_cost (the primary list)
med_names = sorted(compound_map.keys())

for med in med_names:
    cr = compound_map[med]
    price_per_1000 = float(cr['price_per_1000_doses_usd'])
    mailer_format = cr['mailer_format']
    mailer_cost = mailer_map[mailer_format]
    base_pay = base_payment_map[med]
    service_fee = service_fee_map[med]
    total_payment_per_fill = base_pay + service_fee

    # Drug cost: (doses_per_fill * patients * fills * price_per_1000) / 1000
    annual_drug_cost_45 = round((doses_per_fill_45 * patients * fills_45 * price_per_1000) / 1000, 2)
    annual_drug_cost_90 = round((doses_per_fill_90 * patients * fills_90 * price_per_1000) / 1000, 2)

    # Mailer cost: mailer_cost_usd * patients * fills
    annual_mailer_cost_45 = round(mailer_cost * patients * fills_45, 2)
    annual_mailer_cost_90 = round(mailer_cost * patients * fills_90, 2)

    # Payment: total_payment_per_fill * fills
    annual_payment_45 = round(total_payment_per_fill * fills_45, 2)
    annual_payment_90 = round(total_payment_per_fill * fills_90, 2)

    # Margin: payment - drug_cost - mailer_cost
    annual_margin_45 = round(annual_payment_45 - annual_drug_cost_45 - annual_mailer_cost_45, 2)
    annual_margin_90 = round(annual_payment_90 - annual_drug_cost_90 - annual_mailer_cost_90, 2)

    margin_diff = round(annual_margin_90 - annual_margin_45, 2)

    medications_list.append({
        'medication': med,
        'price_per_1000_doses_usd': price_per_1000,
        'mailer_format': mailer_format,
        'mailer_cost_usd': mailer_cost,
        'base_payment_per_fill_150_patients_usd': base_pay,
        'service_fee_per_fill_150_patients_usd': service_fee,
        'total_payment_per_fill_150_patients_usd': round(total_payment_per_fill, 2),
        'annual_drug_cost_45_day_usd': annual_drug_cost_45,
        'annual_drug_cost_90_day_usd': annual_drug_cost_90,
        'annual_mailer_cost_45_day_usd': annual_mailer_cost_45,
        'annual_mailer_cost_90_day_usd': annual_mailer_cost_90,
        'annual_payment_45_day_usd': annual_payment_45,
        'annual_payment_90_day_usd': annual_payment_90,
        'annual_margin_45_day_usd': annual_margin_45,
        'annual_margin_90_day_usd': annual_margin_90,
        'annual_margin_difference_90_minus_45_usd': margin_diff
    })

# Totals
total_margin_45 = round(sum(m['annual_margin_45_day_usd'] for m in medications_list), 2)
total_margin_90 = round(sum(m['annual_margin_90_day_usd'] for m in medications_list), 2)
total_diff = round(total_margin_90 - total_margin_45, 2)
abs_diff = round(abs(total_diff), 2)

# Decision
if abs_diff < 8500:
    decision = 'shift_to_90_day'
    justification = f'The absolute total margin difference of ${abs_diff} is below the ${threshold} threshold, so shifting to 90-day fills is recommended.'
else:
    decision = 'keep_45_day'
    justification = f'The absolute total margin difference of ${abs_diff} exceeds the ${threshold} threshold, so keeping 45-day fills is recommended.'

output = {
    'assumptions': {
        'patients_per_medication': patients,
        'fills_per_year_45_day': fills_45,
        'fills_per_year_90_day': fills_90,
        'doses_per_fill_45_day': doses_per_fill_45,
        'doses_per_fill_90_day': doses_per_fill_90,
        'switch_threshold_usd': threshold
    },
    'medications': medications_list,
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

# Write JSON
with open('/root/mailer_policy_analysis.json', 'w') as f:
    json.dump(output, f, indent=2)

print('\n=== JSON written ===')
print(json.dumps(output, indent=2))

# Write Markdown summary
md_lines = [
    '# Mailer Policy Analysis Summary',
    '',
    f'- Total 45-day annual margin: ${total_margin_45:,.2f} USD',
    f'- Total 90-day annual margin: ${total_margin_90:,.2f} USD',
    f'- Absolute margin difference: ${abs_diff:,.2f} USD',
    f'- Recommendation: **{decision}**',
    '',
    justification
]

with open('/root/mailer_policy_summary.md', 'w') as f:
    f.write('\n'.join(md_lines) + '\n')

print('\n=== Markdown written ===')
for line in md_lines:
    print(line)

print('\nDone.')
```

After running the script, verify the outputs:
1. Check that `/root/mailer_policy_analysis.json` is valid JSON with `python3 -c "import json; d=json.load(open('/root/mailer_policy_analysis.json')); print('medications count:', len(d['medications'])); print('sorted:', [m['medication'] for m in d['medications']]); print('totals:', d['totals']); print('decision:', d['recommendation']['decision'])"`
2. Check that `/root/mailer_policy_summary.md` has 4-8 non-empty lines and contains the required elements with `grep -c '.' /root/mailer_policy_summary.md` and `cat /root/mailer_policy_summary.md`
3. Verify the Markdown contains the exact decision slug (either `shift_to_90_day` or `keep_45_day`) with `grep -E 'shift_to_90_day|keep_45_day' /root/mailer_policy_summary.md`

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