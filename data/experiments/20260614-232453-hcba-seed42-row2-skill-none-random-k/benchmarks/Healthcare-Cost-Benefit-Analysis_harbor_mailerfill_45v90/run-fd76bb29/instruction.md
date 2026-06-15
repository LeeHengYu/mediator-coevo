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

Note the column names and all rows carefully before proceeding.

### Step 2: Write and run a Python script to produce both output files

Create `/root/solve.py` with the following logic:

```python
import csv, json, math

# --- Load CSVs ---
def load_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))

compound_rows = load_csv('/root/compound_cost.csv')
mailer_rows = load_csv('/root/mailer_cost.csv')
base_rows = load_csv('/root/base_payment.csv')
service_rows = load_csv('/root/service_fee.csv')

# Build lookup dicts
# compound_cost.csv should have columns: medication, price_per_1000_doses_usd, and likely mailer_format
# Inspect and adapt column names as needed

# Print column names for debugging
print("compound_cost columns:", list(compound_rows[0].keys()))
print("mailer_cost columns:", list(mailer_rows[0].keys()))
print("base_payment columns:", list(base_rows[0].keys()))
print("service_fee columns:", list(service_rows[0].keys()))

# Build lookups keyed by medication
compound = {r['medication']: r for r in compound_rows}
base = {r['medication']: r for r in base_rows}
service = {r['medication']: r for r in service_rows}
# mailer_cost is keyed by mailer_format
mailer = {r['mailer_format']: float(r['mailer_cost_usd']) for r in mailer_rows}

patients = 150
fills_45 = 8
fills_90 = 4
doses_45 = 45
doses_90 = 90
threshold = 8500

medications_out = []

for med_name in sorted(compound.keys()):
    c = compound[med_name]
    b = base[med_name]
    s = service[med_name]
    
    price_per_1000 = float(c['price_per_1000_doses_usd'])
    mailer_format = c['mailer_format']
    mailer_cost_per = mailer[mailer_format]
    base_pay = float(b['base_payment_per_fill_150_patients_usd'])
    svc_fee = float(s['service_fee_per_fill_150_patients_usd'])
    
    total_payment_per_fill = round(base_pay + svc_fee, 2)
    
    # Drug cost per fill = (doses_per_fill * patients * price_per_1000) / 1000
    # Annual drug cost = drug_cost_per_fill * fills_per_year
    annual_drug_cost_45 = round((doses_45 * patients * price_per_1000 / 1000) * fills_45, 2)
    annual_drug_cost_90 = round((doses_90 * patients * price_per_1000 / 1000) * fills_90, 2)
    
    # Mailer cost per fill = mailer_cost_usd * patients (per patient per fill)
    # Annual mailer cost = mailer_cost_per_fill * fills_per_year
    annual_mailer_45 = round((mailer_cost_per * patients) * fills_45, 2)
    annual_mailer_90 = round((mailer_cost_per * patients) * fills_90, 2)
    
    # Annual payment = total_payment_per_fill * fills_per_year
    annual_payment_45 = round(total_payment_per_fill * fills_45, 2)
    annual_payment_90 = round(total_payment_per_fill * fills_90, 2)
    
    # Annual margin = annual_payment - annual_drug_cost - annual_mailer_cost
    annual_margin_45 = round(annual_payment_45 - annual_drug_cost_45 - annual_mailer_45, 2)
    annual_margin_90 = round(annual_payment_90 - annual_drug_cost_90 - annual_mailer_90, 2)
    
    diff = round(annual_margin_90 - annual_margin_45, 2)
    
    medications_out.append({
        "medication": med_name,
        "price_per_1000_doses_usd": price_per_1000,
        "mailer_format": mailer_format,
        "mailer_cost_usd": mailer_cost_per,
        "base_payment_per_fill_150_patients_usd": base_pay,
        "service_fee_per_fill_150_patients_usd": svc_fee,
        "total_payment_per_fill_150_patients_usd": total_payment_per_fill,
        "annual_drug_cost_45_day_usd": annual_drug_cost_45,
        "annual_drug_cost_90_day_usd": annual_drug_cost_90,
        "annual_mailer_cost_45_day_usd": annual_mailer_45,
        "annual_mailer_cost_90_day_usd": annual_mailer_90,
        "annual_payment_45_day_usd": annual_payment_45,
        "annual_payment_90_day_usd": annual_payment_90,
        "annual_margin_45_day_usd": annual_margin_45,
        "annual_margin_90_day_usd": annual_margin_90,
        "annual_margin_difference_90_minus_45_usd": diff
    })

total_margin_45 = round(sum(m['annual_margin_45_day_usd'] for m in medications_out), 2)
total_margin_90 = round(sum(m['annual_margin_90_day_usd'] for m in medications_out), 2)
total_diff = round(total_margin_90 - total_margin_45, 2)
abs_diff = round(abs(total_diff), 2)

if abs_diff < 8500:
    decision = "shift_to_90_day"
    justification = (f"The absolute total margin difference of ${abs_diff} is below "
                     f"the ${threshold} threshold, so shifting to 90-day fills is recommended.")
else:
    decision = "keep_45_day"
    justification = (f"The absolute total margin difference of ${abs_diff} meets or exceeds "
                     f"the ${threshold} threshold, so keeping 45-day fills is recommended.")

result = {
    "assumptions": {
        "patients_per_medication": 150,
        "fills_per_year_45_day": 8,
        "fills_per_year_90_day": 4,
        "doses_per_fill_45_day": 45,
        "doses_per_fill_90_day": 90,
        "switch_threshold_usd": 8500
    },
    "medications": medications_out,
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
    json.dump(result, f, indent=2)

print("JSON written.")
print(json.dumps(result, indent=2))

# --- Write summary markdown ---
lines = [
    "# Mailer Policy Analysis Summary",
    "",
    f"Total 45-day annual margin: ${total_margin_45:,.2f}",
    f"Total 90-day annual margin: ${total_margin_90:,.2f}",
    f"Absolute margin difference: ${abs_diff:,.2f}",
    f"Recommendation: {decision}",
    "",
    f"{justification}"
]

with open('/root/mailer_policy_summary.md', 'w') as f:
    f.write('\n'.join(lines) + '\n')

print("\nMarkdown written.")
```

Run it:
```
python3 /root/solve.py
```

### Step 3: Adapt if column names differ

After Step 1, if the CSV column names don't match what's assumed above (e.g., `mailer_format` might be in a different CSV, or column names have slight variations), adjust the script accordingly before running. The key lookups are:
- `compound_cost.csv`: must provide `medication`, `price_per_1000_doses_usd`, and `mailer_format`
- `mailer_cost.csv`: must provide `mailer_format` and `mailer_cost_usd`
- `base_payment.csv`: must provide `medication` and `base_payment_per_fill_150_patients_usd`
- `service_fee.csv`: must provide `medication` and `service_fee_per_fill_150_patients_usd`

If `mailer_format` is not in `compound_cost.csv` but in another file, adjust the join logic.

### Step 4: Validate outputs

1. Verify JSON is valid:
```
python3 -c "import json; d=json.load(open('/root/mailer_policy_analysis.json')); print('Keys:', list(d.keys())); print('Num meds:', len(d['medications'])); print('Sorted check:', [m['medication'] for m in d['medications']]); print('Totals:', d['totals']); print('Decision:', d['recommendation']['decision'])"
```

2. Verify markdown:
```
cat /root/mailer_policy_summary.md
```

Check:
- JSON has all required top-level keys: `assumptions`, `medications`, `totals`, `recommendation`
- `medications` array is sorted alphabetically by `medication`
- All currency values are rounded to 2 decimal places
- The decision slug is exactly `shift_to_90_day` or `keep_45_day`
- The markdown has 4-8 non-empty lines and includes: total 45-day margin, total 90-day margin, absolute difference, and the exact decision slug
- `annual_drug_cost` for 45-day and 90-day should be equal (since total annual doses = 150 patients × 365 doses... actually: 45×8=360 vs 90×4=360, so same total doses → same annual drug cost)

### Step 5: Sanity check the math

For any one medication, verify by hand:
- Annual doses = doses_per_fill × fills_per_year = 45×8 = 360 (same for 90×4 = 360)
- Annual drug cost = (360 × patients × price_per_1000) / 1000 — should be identical for both models
- Annual mailer cost differs: 8 fills × 150 patients × mailer_cost vs 4 fills × 150 patients × mailer_cost
- Annual payment differs: total_payment_per_fill × 8 vs × 4
- The margin difference should come from payment difference minus mailer cost difference (drug cost cancels out)

If drug costs are indeed equal for both models, the margin difference per med = (payment_per_fill × 4 - payment_per_fill × 8) - (mailer × 150 × 4 - mailer × 150 × 8) = -4 × payment_per_fill + 4 × 150 × mailer. This is a useful sanity check.

Confirm both output files exist and look correct, then the task is complete.

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