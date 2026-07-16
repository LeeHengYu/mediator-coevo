# Task Instruction

Execute the following steps in order:

## Step 1: Inspect input files
Read and display the contents of all four input CSV files:
```
cat /root/compound_cost.csv
cat /root/mailer_cost.csv
cat /root/base_payment.csv
cat /root/service_fee.csv
```
Note the column names, data types, and how medications are identified across files. Identify the join key(s) — likely `medication` and/or `mailer_format`.

## Step 2: Write and run a Python script to produce both output files

Create `/root/solve.py` with the following logic:

```python
import csv, json, math

# 1. Read all CSVs into dicts keyed by medication
def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))

compound = {r['medication']: r for r in read_csv('/root/compound_cost.csv')}
mailer = {r['mailer_format']: r for r in read_csv('/root/mailer_cost.csv')}
base_pay = {r['medication']: r for r in read_csv('/root/base_payment.csv')}
service = {r['medication']: r for r in read_csv('/root/service_fee.csv')}

# 2. Constants
patients = 150
fills_45 = 8
fills_90 = 4
doses_45 = 45
doses_90 = 90
threshold = 8500

# 3. Build medication list (sorted alphabetically)
medications = []
for med_name in sorted(compound.keys()):
    c = compound[med_name]
    price_per_1000 = float(c['price_per_1000_doses_usd'])
    fmt = c['mailer_format']  # mailer_format should be in compound_cost or base_payment; adjust if needed
    mailer_cost = float(mailer[fmt]['mailer_cost_usd'])
    base_pf = float(base_pay[med_name]['base_payment_per_fill_150_patients_usd'])
    svc_pf = float(service[med_name]['service_fee_per_fill_150_patients_usd'])
    total_pf = round(base_pf + svc_pf, 2)

    # Drug cost = (doses_per_fill * patients * price_per_1000 / 1000) * fills
    drug_cost_per_fill_45 = doses_45 * patients * price_per_1000 / 1000
    drug_cost_per_fill_90 = doses_90 * patients * price_per_1000 / 1000
    annual_drug_45 = round(drug_cost_per_fill_45 * fills_45, 2)
    annual_drug_90 = round(drug_cost_per_fill_90 * fills_90, 2)

    # Mailer cost = mailer_cost * patients * fills
    annual_mailer_45 = round(mailer_cost * patients * fills_45, 2)
    annual_mailer_90 = round(mailer_cost * patients * fills_90, 2)

    # Payment = total_payment_per_fill * fills
    annual_pay_45 = round(total_pf * fills_45, 2)
    annual_pay_90 = round(total_pf * fills_90, 2)

    # Margin = payment - drug_cost - mailer_cost
    margin_45 = round(annual_pay_45 - annual_drug_45 - annual_mailer_45, 2)
    margin_90 = round(annual_pay_90 - annual_drug_90 - annual_mailer_90, 2)
    diff = round(margin_90 - margin_45, 2)

    medications.append({
        'medication': med_name,
        'price_per_1000_doses_usd': price_per_1000,
        'mailer_format': fmt,
        'mailer_cost_usd': mailer_cost,
        'base_payment_per_fill_150_patients_usd': base_pf,
        'service_fee_per_fill_150_patients_usd': svc_pf,
        'total_payment_per_fill_150_patients_usd': total_pf,
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

# 4. Totals
total_margin_45 = round(sum(m['annual_margin_45_day_usd'] for m in medications), 2)
total_margin_90 = round(sum(m['annual_margin_90_day_usd'] for m in medications), 2)
total_diff = round(total_margin_90 - total_margin_45, 2)
abs_diff = round(abs(total_diff), 2)

# 5. Decision
if abs_diff < 8500:
    decision = 'shift_to_90_day'
    justification = f'Absolute total margin difference ${abs_diff} is below the ${threshold} threshold, so shifting to 90-day fills is recommended.'
else:
    decision = 'keep_45_day'
    justification = f'Absolute total margin difference ${abs_diff} meets or exceeds the ${threshold} threshold, so keeping 45-day fills is recommended.'

# 6. Build JSON
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

# 7. Build markdown summary (4-8 non-empty lines)
md = f"""# Mailer Policy Analysis Summary

Total 45-day annual margin: ${total_margin_45:,.2f} USD
Total 90-day annual margin: ${total_margin_90:,.2f} USD
Absolute margin difference: ${abs_diff:,.2f} USD
Decision: {decision}
{justification}
"""

with open('/root/mailer_policy_summary.md', 'w') as f:
    f.write(md)

print('Done. Outputs written.')
print(json.dumps(result, indent=2))
```

**IMPORTANT**: Before running, after reading the CSVs in Step 1, verify:
- Which CSV contains the `mailer_format` column that links medications to mailer costs. It might be in `compound_cost.csv` or another file. Adjust the script accordingly.
- If `mailer_format` is NOT in `compound_cost.csv`, find which file has it and adjust the join logic.
- Check if column names match exactly (case, underscores, etc.).

Run: `python3 /root/solve.py`

## Step 3: Validate outputs
1. `cat /root/mailer_policy_analysis.json` — verify it parses as valid JSON, medications are sorted alphabetically, all currency values have at most 2 decimal places.
2. `cat /root/mailer_policy_summary.md` — verify 4-8 non-empty lines, contains total 45-day margin, total 90-day margin, absolute difference, and the exact decision slug.
3. Quick sanity: for one medication, manually verify the drug cost calculation: `doses_per_fill * patients * price_per_1000 / 1000 * fills_per_year`.
4. Verify the decision rule: if abs(total_difference) < 8500 → shift_to_90_day, else keep_45_day.

## Step 4: Fix any issues
If the script fails (e.g., KeyError on column names, missing mailer_format link), re-read the CSVs, identify the correct column names and join keys, fix the script, and re-run. Do not guess — always inspect the actual CSV contents first.

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