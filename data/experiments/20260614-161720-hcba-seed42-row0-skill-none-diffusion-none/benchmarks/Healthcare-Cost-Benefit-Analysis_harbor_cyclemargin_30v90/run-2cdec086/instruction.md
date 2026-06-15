# Task Instruction

Execute the following steps to produce the two required output files.

## Step 1 – Inspect the input files
```bash
cat /root/acquisition_cost.csv
cat /root/packaging_cost.csv
cat /root/reimbursement.csv
```
Understand the column names and the set of therapies.

## Step 2 – Write and run a Python script

Create `/root/solve.py` with the logic below, then run it with `python3 /root/solve.py`.

```python
import csv, json, math

# ── Read inputs ──────────────────────────────────────────────
def read_csv(path):
 with open(path) as f:
 return list(csv.DictReader(f))

acq_rows = read_csv('/root/acquisition_cost.csv')
pkg_rows = read_csv('/root/packaging_cost.csv')
reimb_rows = read_csv('/root/reimbursement.csv')

# Build lookup dicts keyed by therapy name
acq = {r['therapy']: float(r['price_per_1000_doses_usd']) for r in acq_rows}
pkg_by_canister = {int(r['canister_size_units']): float(r['packaging_cost_usd']) for r in pkg_rows}
canister_for = {r['therapy']: int(r['canister_size_units']) for r in acq_rows}
reimb = {r['therapy']: float(r['reimbursement_per_fill_240_patients_usd']) for r in reimb_rows}

# ── Constants ────────────────────────────────────────────────
PATIENTS = 240
FILLS_30 = 12
FILLS_90 = 4
DOSES_30 = 60
DOSES_90 = 180
THRESHOLD = 12000
DOSES_PER_PATIENT_PER_YEAR = 2 * 365  # 730 doses

# ── Per-therapy calculations ─────────────────────────────────
therapies = sorted(acq.keys())  # alphabetical
therapy_results = []
for t in therapies:
 price_1000 = acq[t]
 cs = canister_for[t]
 pkg_cost = pkg_by_canister[cs]
 reimb_fill = reimb[t]

 # Drug cost: same for both models (same annual doses)
 # annual doses per patient = 2 * 365 = 730
 # total doses = 730 * 240
 # BUT the fills model: 30-day -> 60 doses/fill * 12 fills = 720 doses/patient/year
 #                       90-day -> 180 doses/fill * 4 fills = 720 doses/patient/year
 # So annual doses per patient = 720 (from fill model), total = 720 * 240
 annual_doses = 720 * PATIENTS  # 172800
 annual_drug_cost = round(annual_doses * price_1000 / 1000, 2)
 # Drug cost is the same for 30 and 90 day
 annual_drug_cost_30 = annual_drug_cost
 annual_drug_cost_90 = annual_drug_cost

 # Packaging cost: per patient per fill
 annual_pkg_30 = round(pkg_cost * PATIENTS * FILLS_30, 2)
 annual_pkg_90 = round(pkg_cost * PATIENTS * FILLS_90, 2)

 # Reimbursement
 annual_reimb_30 = round(reimb_fill * FILLS_30, 2)
 annual_reimb_90 = round(reimb_fill * FILLS_90, 2)

 # Margins
 margin_30 = round(annual_reimb_30 - annual_drug_cost_30 - annual_pkg_30, 2)
 margin_90 = round(annual_reimb_90 - annual_drug_cost_90 - annual_pkg_90, 2)
 diff = round(margin_90 - margin_30, 2)

 therapy_results.append({
 "therapy": t,
 "price_per_1000_doses_usd": price_1000,
 "canister_size_units": cs,
 "packaging_cost_usd": pkg_cost,
 "reimbursement_per_fill_240_patients_usd": reimb_fill,
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

# ── Totals ───────────────────────────────────────────────────
total_30 = round(sum(t["annual_margin_30_day_usd"] for t in therapy_results), 2)
total_90 = round(sum(t["annual_margin_90_day_usd"] for t in therapy_results), 2)
total_diff = round(total_90 - total_30, 2)
abs_diff = round(abs(total_diff), 2)

# ── Decision ─────────────────────────────────────────────────
if abs_diff < THRESHOLD:
 decision = "adopt_90_day"
 justification = (f"The absolute total margin difference of ${abs_diff:,.2f} "
 f"is below the ${THRESHOLD:,.2f} threshold, so adopting "
 f"90-day fills is recommended.")
else:
 decision = "keep_30_day"
 justification = (f"The absolute total margin difference of ${abs_diff:,.2f} "
 f"exceeds the ${THRESHOLD:,.2f} threshold, so keeping "
 f"30-day fills is recommended.")

# ── Build JSON ───────────────────────────────────────────────
output = {
 "assumptions": {
 "patients_per_therapy": PATIENTS,
 "fills_per_year_30_day": FILLS_30,
 "fills_per_year_90_day": FILLS_90,
 "doses_per_fill_30_day": DOSES_30,
 "doses_per_fill_90_day": DOSES_90,
 "switch_threshold_usd": THRESHOLD
 },
 "therapies": therapy_results,
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
 json.dump(output, f, indent=2)

print("JSON written.")

# ── Build Markdown summary ───────────────────────────────────
md_lines = [
 "# Cycle Margin Summary",
 "",
 f"- Total 30-day margin: ${total_30:,.2f} USD",
 f"- Total 90-day margin: ${total_90:,.2f} USD",
 f"- Absolute difference: ${abs_diff:,.2f} USD",
 f"- Decision: {decision}",
 "",
 justification
]

with open('/root/cycle_margin_summary.md', 'w') as f:
 f.write('\n'.join(md_lines) + '\n')

print("Markdown written.")
```

## Step 3 – Validate outputs
```bash
python3 -c "
import json
with open('/root/cycle_margin_analysis.json') as f:
 d = json.load(f)
assert set(d.keys()) == {'assumptions','therapies','totals','recommendation'}
assert isinstance(d['therapies'], list) and len(d['therapies']) > 0
for t in d['therapies']:
 for k in ['therapy','annual_margin_30_day_usd','annual_margin_90_day_usd','annual_margin_difference_90_minus_30_usd']:
 assert k in t, f'Missing {k}'
assert d['therapies'] == sorted(d['therapies'], key=lambda x: x['therapy'])
assert 'total_annual_margin_30_day_usd' in d['totals']
assert d['recommendation']['decision'] in ('adopt_90_day','keep_30_day')
print('JSON schema OK')
"

wc -l /root/cycle_margin_summary.md
grep -c '.' /root/cycle_margin_summary.md   # non-empty line count
grep 'adopt_90_day\|keep_30_day' /root/cycle_margin_summary.md
cat /root/cycle_margin_analysis.json
cat /root/cycle_margin_summary.md
```

Confirm:
- JSON parses cleanly, has all required keys, therapies sorted alphabetically.
- Markdown has 4-8 non-empty lines, includes total 30-day margin, total 90-day margin, absolute difference, and the exact decision slug.
- Drug costs are identical for 30-day and 90-day (720 doses/patient/year in both models).
- Packaging and reimbursement scale with number of fills (12 vs 4).

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