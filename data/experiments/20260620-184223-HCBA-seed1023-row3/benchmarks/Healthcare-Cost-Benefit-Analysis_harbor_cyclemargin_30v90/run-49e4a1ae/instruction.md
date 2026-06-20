# Task Instruction

Execute the following steps in order:

## Step 1 – Inspect the input files
```bash
cat /root/acquisition_cost.csv
cat /root/packaging_cost.csv
cat /root/reimbursement.csv
```
Understand the column names, therapy names, and data types before writing any code.

## Step 2 – Write and run the solver script
Create `/root/solve.py` with the following logic:

```python
import csv, json, pathlib

# ── read inputs ──────────────────────────────────────────────
def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))

acq = {r['therapy'].strip(): r for r in read_csv('/root/acquisition_cost.csv')}
pkg = {int(r['canister_size_units']): float(r['packaging_cost_usd']) for r in read_csv('/root/packaging_cost.csv')}
reimb = {r['therapy'].strip(): float(r['reimbursement_per_fill_240_patients_usd']) for r in read_csv('/root/reimbursement.csv')}

# ── constants ────────────────────────────────────────────────
PATIENTS = 240
FILLS_30 = 12
FILLS_90 = 4
DOSES_30 = 60
DOSES_90 = 180
THRESHOLD = 12000

therapies_out = []
for therapy_name in sorted(acq.keys()):
    row = acq[therapy_name]
    price_per_1000 = float(row['price_per_1000_doses_usd'])
    canister_size = int(row['canister_size_units'])
    packaging_cost = pkg[canister_size]
    reimb_per_fill = reimb[therapy_name]

    # drug cost = (doses_per_fill / 1000) * price_per_1000 * patients * fills
    annual_drug_cost_30 = round((DOSES_30 / 1000) * price_per_1000 * PATIENTS * FILLS_30, 2)
    annual_drug_cost_90 = round((DOSES_90 / 1000) * price_per_1000 * PATIENTS * FILLS_90, 2)

    # packaging cost = packaging_cost_usd * patients * fills
    annual_pkg_30 = round(packaging_cost * PATIENTS * FILLS_30, 2)
    annual_pkg_90 = round(packaging_cost * PATIENTS * FILLS_90, 2)

    # reimbursement = reimb_per_fill * fills
    annual_reimb_30 = round(reimb_per_fill * FILLS_30, 2)
    annual_reimb_90 = round(reimb_per_fill * FILLS_90, 2)

    margin_30 = round(annual_reimb_30 - annual_drug_cost_30 - annual_pkg_30, 2)
    margin_90 = round(annual_reimb_90 - annual_drug_cost_90 - annual_pkg_90, 2)
    diff = round(margin_90 - margin_30, 2)

    therapies_out.append({
        "therapy": therapy_name,
        "price_per_1000_doses_usd": price_per_1000,
        "canister_size_units": canister_size,
        "packaging_cost_usd": packaging_cost,
        "reimbursement_per_fill_240_patients_usd": reimb_per_fill,
        "annual_drug_cost_30_day_usd": annual_drug_cost_30,
        "annual_drug_cost_90_day_usd": annual_drug_cost_90,
        "annual_packaging_cost_30_day_usd": annual_pkg_30,
        "annual_packaging_cost_90_day_usd": annual_pkg_90,
        "annual_reimbursement_30_day_usd": annual_reimb_30,
        "annual_reimbursement_90_day_usd": annual_reimb_90,
        "annual_margin_30_day_usd": margin_30,
        "annual_margin_90_day_usd": margin_90,
        "annual_margin_difference_90_minus_30_usd": diff
    })

total_30 = round(sum(t["annual_margin_30_day_usd"] for t in therapies_out), 2)
total_90 = round(sum(t["annual_margin_90_day_usd"] for t in therapies_out), 2)
total_diff = round(total_90 - total_30, 2)
abs_diff = round(abs(total_diff), 2)

if abs_diff < THRESHOLD:
    decision = "adopt_90_day"
    justification = (f"The absolute total margin difference of ${abs_diff:,.2f} "
                     f"is below the ${THRESHOLD:,.2f} threshold, so switching to "
                     f"90-day fills is recommended.")
else:
    decision = "keep_30_day"
    justification = (f"The absolute total margin difference of ${abs_diff:,.2f} "
                     f"exceeds the ${THRESHOLD:,.2f} threshold, so the clinic "
                     f"should keep 30-day fills.")

result = {
    "assumptions": {
        "patients_per_therapy": PATIENTS,
        "fills_per_year_30_day": FILLS_30,
        "fills_per_year_90_day": FILLS_90,
        "doses_per_fill_30_day": DOSES_30,
        "doses_per_fill_90_day": DOSES_90,
        "switch_threshold_usd": THRESHOLD
    },
    "therapies": therapies_out,
    "totals": {
        "total_annual_margin_30_day_usd": total_30,
        "total_annual_margin_90_day_usd": total_90,
        "total_annual_margin_difference_90_minus_30_usd": total_diff,
        "absolute_total_margin_difference_usd": abs_diff
    },
    "recommendation": {
        "decision": decision,
        "justification": justification
    }
}

with open('/root/cycle_margin_analysis.json', 'w') as f:
    json.dump(result, f, indent=2)

# ── markdown summary ─────────────────────────────────────────
lines = [
    "# Cycle Margin Summary",
    "",
    f"Total 30-day annual margin: ${total_30:,.2f}",
    f"Total 90-day annual margin: ${total_90:,.2f}",
    f"Absolute margin difference: ${abs_diff:,.2f}",
    f"Decision: {decision}",
    "",
    justification
]
with open('/root/cycle_margin_summary.md', 'w') as f:
    f.write('\n'.join(lines) + '\n')

print("Done.")
print(json.dumps(result, indent=2))
```

Run it:
```bash
python3 /root/solve.py
```

## Step 3 – Validate outputs
```bash
# Confirm JSON is valid and has required top-level keys
python3 -c "
import json
with open('/root/cycle_margin_analysis.json') as f:
    d = json.load(f)
assert set(d.keys()) == {'assumptions','therapies','totals','recommendation'}, f'Bad top keys: {d.keys()}'
assert 'patients_per_therapy' in d['assumptions']
assert 'fills_per_year_30_day' in d['assumptions']
assert 'fills_per_year_90_day' in d['assumptions']
assert 'doses_per_fill_30_day' in d['assumptions']
assert 'doses_per_fill_90_day' in d['assumptions']
assert 'switch_threshold_usd' in d['assumptions']
for t in d['therapies']:
    for k in ['therapy','annual_drug_cost_30_day_usd','annual_margin_30_day_usd','annual_margin_difference_90_minus_30_usd']:
        assert k in t, f'Missing {k}'
assert isinstance(d['recommendation'], dict)
assert d['recommendation']['decision'] in ('adopt_90_day','keep_30_day')
print('JSON schema OK')
"

# Confirm markdown has required content
python3 -c "
with open('/root/cycle_margin_summary.md') as f:
    text = f.read()
lines = [l for l in text.strip().split('\n') if l.strip()]
assert 4 <= len(lines) <= 8, f'Line count {len(lines)} not in 4-8'
assert 'adopt_90_day' in text or 'keep_30_day' in text
print('Markdown OK')
"
```

## Step 4 – Display final outputs
```bash
cat /root/cycle_margin_analysis.json
echo '---'
cat /root/cycle_margin_summary.md
```

If any step fails, diagnose from the error message, fix, and re-run before proceeding.

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