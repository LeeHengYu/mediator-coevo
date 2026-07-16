# Task Instruction

Execute the following steps in order:

1. **Inspect the input files** to understand their structure and column names:
```bash
cat /root/acquisition_cost.csv
cat /root/packaging_cost.csv
cat /root/reimbursement.csv
```

2. **Create and run `/root/solve.py`** with the following logic:

```python
import csv, json, math

# Read acquisition_cost.csv
with open('/root/acquisition_cost.csv') as f:
    acq = list(csv.DictReader(f))

# Read packaging_cost.csv
with open('/root/packaging_cost.csv') as f:
    pkg = list(csv.DictReader(f))

# Read reimbursement.csv
with open('/root/reimbursement.csv') as f:
    reimb = list(csv.DictReader(f))

# Build lookup dicts
pkg_lookup = {int(row['canister_size_units']): float(row['packaging_cost_usd']) for row in pkg}
reimb_lookup = {row['therapy'].strip(): float(row['reimbursement_per_fill_240_patients_usd']) for row in reimb}

patients = 240
fills_30 = 12
fills_90 = 4
doses_30 = 60
doses_90 = 180
threshold = 12000

therapies = []
for row in acq:
    therapy = row['therapy'].strip()
    price_per_1000 = float(row['price_per_1000_doses_usd'])
    canister_size = int(row['canister_size_units'])
    packaging_cost = pkg_lookup[canister_size]
    reimb_per_fill = reimb_lookup[therapy]

    # Drug cost: (doses_per_fill * patients * fills * price_per_1000) / 1000
    annual_drug_cost_30 = round((doses_30 * patients * fills_30 * price_per_1000) / 1000, 2)
    annual_drug_cost_90 = round((doses_90 * patients * fills_90 * price_per_1000) / 1000, 2)

    # Packaging cost: packaging_cost_usd per patient per fill
    annual_pkg_30 = round(packaging_cost * patients * fills_30, 2)
    annual_pkg_90 = round(packaging_cost * patients * fills_90, 2)

    # Reimbursement: reimbursement_per_fill_240_patients * fills
    annual_reimb_30 = round(reimb_per_fill * fills_30, 2)
    annual_reimb_90 = round(reimb_per_fill * fills_90, 2)

    margin_30 = round(annual_reimb_30 - annual_drug_cost_30 - annual_pkg_30, 2)
    margin_90 = round(annual_reimb_90 - annual_drug_cost_90 - annual_pkg_90, 2)
    diff = round(margin_90 - margin_30, 2)

    therapies.append({
        'therapy': therapy,
        'price_per_1000_doses_usd': price_per_1000,
        'canister_size_units': canister_size,
        'packaging_cost_usd': packaging_cost,
        'reimbursement_per_fill_240_patients_usd': reimb_per_fill,
        'annual_drug_cost_30_day_usd': annual_drug_cost_30,
        'annual_drug_cost_90_day_usd': annual_drug_cost_90,
        'annual_packaging_cost_30_day_usd': annual_pkg_30,
        'annual_packaging_cost_90_day_usd': annual_pkg_90,
        'annual_reimbursement_30_day_usd': annual_reimb_30,
        'annual_reimbursement_90_day_usd': annual_reimb_90,
        'annual_margin_30_day_usd': margin_30,
        'annual_margin_90_day_usd': margin_90,
        'annual_margin_difference_90_minus_30_usd': diff
    })

# Sort alphabetically by therapy
therapies.sort(key=lambda x: x['therapy'])

total_30 = round(sum(t['annual_margin_30_day_usd'] for t in therapies), 2)
total_90 = round(sum(t['annual_margin_90_day_usd'] for t in therapies), 2)
total_diff = round(sum(t['annual_margin_difference_90_minus_30_usd'] for t in therapies), 2)
abs_diff = round(abs(total_diff), 2)

if abs_diff < threshold:
    decision = 'adopt_90_day'
    justification = f'The absolute total margin difference of ${abs_diff:,.2f} is below the ${threshold:,.2f} threshold, so switching to 90-day fills is recommended.'
else:
    decision = 'keep_30_day'
    justification = f'The absolute total margin difference of ${abs_diff:,.2f} exceeds the ${threshold:,.2f} threshold, so keeping 30-day fills is recommended.'

result = {
    'assumptions': {
        'patients_per_therapy': patients,
        'fills_per_year_30_day': fills_30,
        'fills_per_year_90_day': fills_90,
        'doses_per_fill_30_day': doses_30,
        'doses_per_fill_90_day': doses_90,
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

# Format currency with comma thousands separators
def fmt(v):
    if v < 0:
        return f'-${abs(v):,.2f}'
    return f'${v:,.2f}'

lines = [
    '# Cycle Margin Analysis Summary',
    '',
    f'Total 30-day annual margin: {fmt(total_30)}',
    f'Total 90-day annual margin: {fmt(total_90)}',
    f'Absolute margin difference: {fmt(abs_diff)}',
    f'Recommendation: {decision}',
]

with open('/root/cycle_margin_summary.md', 'w') as f:
    f.write('\n'.join(lines) + '\n')

print('Done')
print(json.dumps(result, indent=2))
```

3. Run the script:
```bash
python3 /root/solve.py
```

4. **Validate outputs**:
```bash
cat /root/cycle_margin_analysis.json
cat /root/cycle_margin_summary.md
python3 -c "
import json
with open('/root/cycle_margin_analysis.json') as f:
    d = json.load(f)
assert 'assumptions' in d
assert 'therapies' in d and len(d['therapies']) > 0
assert 'totals' in d
assert 'recommendation' in d
assert d['recommendation']['decision'] in ('adopt_90_day', 'keep_30_day')
assert d['therapies'] == sorted(d['therapies'], key=lambda x: x['therapy'])
for t in d['therapies']:
    for k in ['therapy','price_per_1000_doses_usd','canister_size_units','packaging_cost_usd','reimbursement_per_fill_240_patients_usd','annual_drug_cost_30_day_usd','annual_drug_cost_90_day_usd','annual_packaging_cost_30_day_usd','annual_packaging_cost_90_day_usd','annual_reimbursement_30_day_usd','annual_reimbursement_90_day_usd','annual_margin_30_day_usd','annual_margin_90_day_usd','annual_margin_difference_90_minus_30_usd']:
        assert k in t, f'Missing key {k}'
print('JSON validation passed')
"

# Verify summary has required content
python3 -c "
with open('/root/cycle_margin_summary.md') as f:
    text = f.read()
non_empty = [l for l in text.strip().split('\n') if l.strip()]
assert 4 <= len(non_empty) <= 8, f'Expected 4-8 non-empty lines, got {len(non_empty)}'
assert 'adopt_90_day' in text or 'keep_30_day' in text
print('Summary validation passed')
"
```

5. If any step fails, inspect the error, fix the issue, and re-run. Pay special attention to:
   - Column name mismatches in CSVs (inspect headers first)
   - Matching therapies across files (strip whitespace, check case)
   - Currency formatting in summary must use comma thousands separators (e.g., `$12,345.67` not `$12345.67`)
   - JSON keys must match the schema exactly (use the long-form names as specified)

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