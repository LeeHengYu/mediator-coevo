# Task Instruction

Execute the following Python script to produce both output files. Before running the script, inspect the four input CSV files to understand their structure.

```bash
cd /root
cat compound_cost.csv
cat mailer_cost.csv
cat base_payment.csv
cat service_fee.csv
```

Then create and run this Python script:

```python
import csv
import json

# Read compound_cost.csv
compound_cost = {}
with open('/root/compound_cost.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        med = row['medication'].strip()
        compound_cost[med] = {
            'price_per_1000_doses_usd': float(row['price_per_1000_doses_usd']),
            'mailer_format': row['mailer_format'].strip()
        }

# Read mailer_cost.csv
mailer_cost = {}
with open('/root/mailer_cost.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        fmt = row['mailer_format'].strip()
        mailer_cost[fmt] = float(row['mailer_cost_usd'])

# Read base_payment.csv
base_payment = {}
with open('/root/base_payment.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        med = row['medication'].strip()
        base_payment[med] = float(row['base_payment_per_fill_150_patients_usd'])

# Read service_fee.csv
service_fee = {}
with open('/root/service_fee.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        med = row['medication'].strip()
        service_fee[med] = float(row['service_fee_per_fill_150_patients_usd'])

# Constants
patients = 150
fills_45 = 8
fills_90 = 4
doses_45 = 45
doses_90 = 90
threshold = 8500

# Build medications list sorted alphabetically
medications_list = []
for med in sorted(compound_cost.keys()):
    p1000 = compound_cost[med]['price_per_1000_doses_usd']
    mfmt = compound_cost[med]['mailer_format']
    mc = mailer_cost[mfmt]
    bp = base_payment[med]
    sf = service_fee[med]
    total_payment_per_fill = bp + sf

    # Drug cost = (doses_per_fill * patients * price_per_1000_doses / 1000) * fills_per_year
    annual_drug_cost_45 = round((doses_45 * patients * p1000 / 1000) * fills_45, 2)
    annual_drug_cost_90 = round((doses_90 * patients * p1000 / 1000) * fills_90, 2)

    # Mailer cost = mailer_cost_usd * patients * fills_per_year
    annual_mailer_cost_45 = round(mc * patients * fills_45, 2)
    annual_mailer_cost_90 = round(mc * patients * fills_90, 2)

    # Payment = total_payment_per_fill * fills_per_year
    annual_payment_45 = round(total_payment_per_fill * fills_45, 2)
    annual_payment_90 = round(total_payment_per_fill * fills_90, 2)

    # Margin = payment - drug_cost - mailer_cost
    annual_margin_45 = round(annual_payment_45 - annual_drug_cost_45 - annual_mailer_cost_45, 2)
    annual_margin_90 = round(annual_payment_90 - annual_drug_cost_90 - annual_mailer_cost_90, 2)

    diff = round(annual_margin_90 - annual_margin_45, 2)

    medications_list.append({
        'medication': med,
        'price_per_1000_doses_usd': p1000,
        'mailer_format': mfmt,
        'mailer_cost_usd': mc,
        'base_payment_per_fill_150_patients_usd': bp,
        'service_fee_per_fill_150_patients_usd': sf,
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

# Totals
total_margin_45 = round(sum(m['annual_margin_45_day_usd'] for m in medications_list), 2)
total_margin_90 = round(sum(m['annual_margin_90_day_usd'] for m in medications_list), 2)
total_diff = round(total_margin_90 - total_margin_45, 2)
abs_diff = round(abs(total_diff), 2)

# Decision
if abs_diff < 8500:
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
    json.dump(result, f, indent=2)

# Write Markdown summary with comma-formatted currency
md_lines = [
    '# Mailer Policy Summary',
    '',
    f'Total 45-day annual margin: ${total_margin_45:,.2f}',
    f'Total 90-day annual margin: ${total_margin_90:,.2f}',
    f'Absolute margin difference: ${abs_diff:,.2f}',
    f'Recommendation: {decision}',
]

with open('/root/mailer_policy_summary.md', 'w') as f:
    f.write('\n'.join(md_lines) + '\n')

print('Done. Files written.')
print(json.dumps(result, indent=2))
```

After running the script:
1. Verify the JSON is valid: `python3 -c "import json; json.load(open('/root/mailer_policy_analysis.json'))"`
2. Check the markdown has comma-formatted numbers: `cat /root/mailer_policy_summary.md`
3. Verify the markdown has between 4-8 non-empty lines: `grep -c '.' /root/mailer_policy_summary.md`
4. Run any available test suite: `cd /root && python -m pytest test_output.py -v` (if test file exists)

Key points from previous failure feedback:
- The markdown summary MUST use comma-formatted currency strings (e.g., `$27,000.00` not `$27000.00`). The script above uses `{value:,.2f}` formatting.
- The JSON must have the `recommendation` key (not separate `justification`/`decision` at top level).
- Medications must be sorted alphabetically by medication name.
- All currency values rounded to 2 decimal places.

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