# Task Instruction

Execute the following steps in order:

## Step 1: Inspect the input files

Read and display the contents of:
- `/root/acquisition_cost.csv`
- `/root/packaging_cost.csv`
- `/root/reimbursement.csv`

Understand the column names, therapy names, and how they join together.

## Step 2: Write and run a Python script

Create `/root/solve.py` that does the following:

```python
import csv, json, math

# 1. Read acquisition_cost.csv -> dict keyed by therapy
#    Expected columns: therapy, price_per_1000_doses_usd, (possibly others)
with open('/root/acquisition_cost.csv') as f:
    acq = list(csv.DictReader(f))

# 2. Read packaging_cost.csv -> dict keyed by canister_size_units
#    Expected columns: canister_size_units, packaging_cost_usd
with open('/root/packaging_cost.csv') as f:
    pkg_list = list(csv.DictReader(f))
pkg = {int(row['canister_size_units']): float(row['packaging_cost_usd']) for row in pkg_list}

# 3. Read reimbursement.csv -> dict keyed by therapy
#    Expected columns: therapy, reimbursement_per_fill_240_patients_usd
with open('/root/reimbursement.csv') as f:
    reimb = list(csv.DictReader(f))
reimb_dict = {row['therapy']: float(row['reimbursement_per_fill_240_patients_usd']) for row in reimb}

# Constants
patients = 240
fills_30 = 12
fills_90 = 4
doses_per_fill_30 = 60
doses_per_fill_90 = 180
threshold = 12000

therapies = []
for row in acq:
    therapy = row['therapy']
    price_per_1000 = float(row['price_per_1000_doses_usd'])
    canister_size = int(row['canister_size_units'])
    packaging_cost_usd = pkg[canister_size]
    reimb_per_fill = reimb_dict[therapy]

    # Drug cost = (doses_per_fill / 1000) * price_per_1000 * patients * fills_per_year
    annual_drug_cost_30 = (doses_per_fill_30 / 1000.0) * price_per_1000 * patients * fills_30
    annual_drug_cost_90 = (doses_per_fill_90 / 1000.0) * price_per_1000 * patients * fills_90

    # Packaging cost = packaging_cost_usd * patients * fills_per_year
    annual_pkg_30 = packaging_cost_usd * patients * fills_30
    annual_pkg_90 = packaging_cost_usd * patients * fills_90

    # Reimbursement = reimb_per_fill * fills_per_year
    annual_reimb_30 = reimb_per_fill * fills_30
    annual_reimb_90 = reimb_per_fill * fills_90

    # Margin = reimbursement - drug_cost - packaging_cost
    margin_30 = annual_reimb_30 - annual_drug_cost_30 - annual_pkg_30
    margin_90 = annual_reimb_90 - annual_drug_cost_90 - annual_pkg_90
    diff = margin_90 - margin_30

    therapies.append({
        'therapy': therapy,
        'price_per_1000_doses_usd': price_per_1000,
        'canister_size_units': canister_size,
        'packaging_cost_usd': packaging_cost_usd,
        'reimbursement_per_fill_240_patients_usd': reimb_per_fill,
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

total_30 = round(sum(t['annual_margin_30_day_usd'] for t in therapies), 2)
total_90 = round(sum(t['annual_margin_90_day_usd'] for t in therapies), 2)
total_diff = round(sum(t['annual_margin_difference_90_minus_30_usd'] for t in therapies), 2)
abs_diff = round(abs(total_diff), 2)

if abs_diff < threshold:
    decision = 'adopt_90_day'
    justification = f'The absolute total margin difference of ${abs_diff:,.2f} is below the ${threshold:,} threshold, so switching to 90-day fills is recommended.'
else:
    decision = 'keep_30_day'
    justification = f'The absolute total margin difference of ${abs_diff:,.2f} exceeds the ${threshold:,} threshold, so keeping 30-day fills is recommended.'

result = {
    'assumptions': {
        'patients_per_therapy': patients,
        'fills_per_year_30_day': fills_30,
        'fills_per_year_90_day': fills_90,
        'doses_per_fill_30_day': doses_per_fill_30,
        'doses_per_fill_90_day': doses_per_fill_90,
        'switch_threshold_usd': threshold
    },
    'therapies': therapies,
    'totals': {
        'total_annual_margin_30_day_usd': total_30,
        'total_annual_margin_90_day_usd': total_90,
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

print(json.dumps(result, indent=2))

# Write summary markdown
lines = [
    '# Cycle Margin Analysis Summary',
    '',
    f'Total 30-day annual margin: ${total_30:,.2f} USD',
    f'Total 90-day annual margin: ${total_90:,.2f} USD',
    f'Absolute margin difference: ${abs_diff:,.2f} USD',
    f'Recommendation: {decision}',
    '',
    justification
]

with open('/root/cycle_margin_summary.md', 'w') as f:
    f.write('\n'.join(lines) + '\n')

print('\n--- Summary ---')
print('\n'.join(lines))
```

Run `python3 /root/solve.py`.

If there are column-name mismatches (e.g., the CSV uses different headers), fix them based on what you saw in Step 1 and re-run.

## Step 3: Validate outputs

1. Read `/root/cycle_margin_analysis.json` and confirm:
   - It parses as valid JSON.
   - `therapies` array is sorted alphabetically by `therapy`.
   - All currency values are rounded to 2 decimal places.
   - The `assumptions` block matches the constants.
   - The decision logic is correct: `abs(total_difference) < 12000` → `adopt_90_day`, else `keep_30_day`.

2. Read `/root/cycle_margin_summary.md` and confirm:
   - It has 4–8 non-empty lines.
   - Contains total 30-day margin, total 90-day margin, absolute difference, and the exact decision slug (`adopt_90_day` or `keep_30_day`).

## Step 4: Spot-check one therapy manually

Pick the first therapy alphabetically. Manually compute its annual_drug_cost_30_day, annual_packaging_cost_30_day, annual_reimbursement_30_day, and annual_margin_30_day using the raw CSV values. Compare to the JSON output. If they don't match, debug and fix.

## Important notes
- The drug cost formula: `(doses_per_fill / 1000) * price_per_1000_doses_usd * 240_patients * fills_per_year`. Both 30-day and 90-day models use the same price_per_1000; only doses_per_fill and fills_per_year differ. Note that `60 * 12 * 240 = 172800` total doses/year and `180 * 4 * 240 = 172800` total doses/year — so annual drug costs should be identical for 30-day vs 90-day. This is expected.
- The packaging cost differs because `12 fills * 240 patients = 2880` events vs `4 fills * 240 patients = 960` events. This is the key cost driver.
- The reimbursement also differs: `reimb_per_fill * 12` vs `reimb_per_fill * 4`.
- So the margin difference is driven by packaging savings minus reimbursement loss.

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