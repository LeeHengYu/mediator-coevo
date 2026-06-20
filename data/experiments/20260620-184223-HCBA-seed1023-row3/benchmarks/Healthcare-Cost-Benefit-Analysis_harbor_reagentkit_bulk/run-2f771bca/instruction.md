# Task Instruction

Execute the following steps in order:

## Step 1 – Inspect all input files

```bash
cat /root/assay_manifest.json
cat /root/carrier_cost.csv
cat /root/billing.csv
cat /root/lab_overrides.csv
cat /root/report_template.json
```

Read and understand every field before writing any code.

## Step 2 – Write and run the solver script

Create `/root/solve.py` with the logic below. Follow every rule precisely.

```python
import json, csv, math
from collections import defaultdict

# ── Load inputs ──────────────────────────────────────────────
with open('/root/assay_manifest.json') as f:
    manifest = json.load(f)

with open('/root/carrier_cost.csv') as f:
    carrier_rows = list(csv.DictReader(f))

with open('/root/billing.csv') as f:
    billing_rows = list(csv.DictReader(f))

with open('/root/lab_overrides.csv') as f:
    override_rows = list(csv.DictReader(f))

with open('/root/report_template.json') as f:
    template = json.load(f)

# ── Carrier cost lookup ──────────────────────────────────────
carrier_cost_map = {}
for row in carrier_rows:
    carrier_cost_map[row['carrier_type'].strip()] = float(row['carrier_cost_usd'])

# ── Identify in-scope assays ─────────────────────────────────
assays = manifest['assays']
in_scope = [a for a in assays if a['in_scope'] is True]

# ── Build alias → assay_id map ───────────────────────────────
label_to_assay = {}
for a in in_scope:
    label_to_assay[a['assay_name'].strip().lower()] = a['assay_id']
    for alias in a.get('aliases', []):
        label_to_assay[alias.strip().lower()] = a['assay_id']

# ── Resolve billing rows ────────────────────────────────────
# Keep only active rows that match an in-scope assay
billing_candidates = defaultdict(list)
for row in billing_rows:
    if row['is_active'].strip().lower() != 'true':
        continue
    label = row['assay_label'].strip().lower()
    aid = label_to_assay.get(label)
    if aid is None:
        continue
    billing_candidates[aid].append(row)

# For each assay keep the row with the latest effective_month
billing_final = {}
for aid, rows in billing_candidates.items():
    best = max(rows, key=lambda r: r['effective_month'])
    billing_final[aid] = best

# ── Resolve lab overrides ────────────────────────────────────
override_candidates = defaultdict(list)
for row in override_rows:
    if row['status'].strip().lower() != 'approved':
        continue
    override_candidates[row['assay_id'].strip()].append(row)

override_final = {}
for aid, rows in override_candidates.items():
    best = max(rows, key=lambda r: int(r['revision']))
    override_final[aid] = int(best['active_labs'])

# ── Build per-assay analysis ─────────────────────────────────
results = []
for a in in_scope:
    aid = a['assay_id']
    aname = a['assay_name']

    # Active labs
    active_labs = override_final.get(aid, a['default_active_labs'])

    # Billing
    brow = billing_final[aid]
    payment_per_run_per_lab = float(brow['payment_per_run_per_lab_usd'])

    # Reagent
    reagent_price = float(a['reagent_price_per_1000_tests_usd'])
    tests_small = int(a['tests_per_lab_per_run_small'])
    tests_bulk  = int(a['tests_per_lab_per_run_bulk'])

    # Carrier
    ctype = a['carrier_type'].strip()
    carrier_cost = carrier_cost_map[ctype]

    # Runs per year
    runs_small = 24
    runs_bulk  = 12

    # Annual revenue
    rev_small = payment_per_run_per_lab * active_labs * runs_small
    rev_bulk  = payment_per_run_per_lab * active_labs * runs_bulk

    # Annual reagent cost
    rc_small = reagent_price * active_labs * tests_small * runs_small / 1000
    rc_bulk  = reagent_price * active_labs * tests_bulk  * runs_bulk  / 1000

    # Annual carrier cost
    cc_small = carrier_cost * runs_small
    cc_bulk  = carrier_cost * runs_bulk

    # Annual margin
    margin_small = rev_small - rc_small - cc_small
    margin_bulk  = rev_bulk  - rc_bulk  - cc_bulk

    diff = margin_bulk - margin_small

    results.append({
        'assay_id': aid,
        'assay_name': aname,
        'active_labs': active_labs,
        'reagent_price_per_1000_tests_usd': round(reagent_price, 2),
        'carrier_type': ctype,
        'carrier_cost_usd': round(carrier_cost, 2),
        'payment_per_run_per_lab_usd': round(payment_per_run_per_lab, 2),
        'tests_per_lab_per_run_small': tests_small,
        'tests_per_lab_per_run_bulk': tests_bulk,
        'annual_reagent_cost_small_kit_usd': round(rc_small, 2),
        'annual_reagent_cost_bulk_kit_usd': round(rc_bulk, 2),
        'annual_carrier_cost_small_kit_usd': round(cc_small, 2),
        'annual_carrier_cost_bulk_kit_usd': round(cc_bulk, 2),
        'annual_revenue_small_kit_usd': round(rev_small, 2),
        'annual_revenue_bulk_kit_usd': round(rev_bulk, 2),
        'annual_margin_small_kit_usd': round(margin_small, 2),
        'annual_margin_bulk_kit_usd': round(margin_bulk, 2),
        'annual_margin_difference_bulk_minus_small_usd': round(diff, 2),
    })

# Sort by assay_id ascending
results.sort(key=lambda x: x['assay_id'])

# ── Totals ───────────────────────────────────────────────────
total_margin_small = sum(r['annual_margin_small_kit_usd'] for r in results)
total_margin_bulk  = sum(r['annual_margin_bulk_kit_usd'] for r in results)
total_diff         = sum(r['annual_margin_difference_bulk_minus_small_usd'] for r in results)
abs_diff           = abs(total_diff)

# Decision
if abs_diff < 7000:
    decision = 'adopt_bulk_kit'
    justification = (f'The absolute total margin difference is ${abs_diff:,.2f}, '
                     f'which is below the $7,000 threshold, so bulk-kit adoption is recommended.')
else:
    decision = 'keep_small_kit'
    justification = (f'The absolute total margin difference is ${abs_diff:,.2f}, '
                     f'which exceeds the $7,000 threshold, so keeping the small-kit cadence is recommended.')

# ── Build JSON report ────────────────────────────────────────
report = {
    'metadata': template['metadata'],
    'analysis': {
        'assumptions': {
            'runs_per_year_small_kit': 24,
            'runs_per_year_bulk_kit': 12,
            'switch_threshold_usd': 7000,
            'lab_override_rule': 'highest approved revision per assay_id, else default_active_labs',
            'billing_rule': 'latest active effective_month per assay',
        },
        'assays': results,
        'totals': {
            'total_annual_margin_small_kit_usd': round(total_margin_small, 2),
            'total_annual_margin_bulk_kit_usd': round(total_margin_bulk, 2),
            'total_annual_margin_difference_bulk_minus_small_usd': round(total_diff, 2),
            'absolute_total_margin_difference_usd': round(abs_diff, 2),
        },
        'recommendation': {
            'decision': decision,
            'justification': justification,
        },
    },
}

with open('/root/reagent_policy_report.json', 'w') as f:
    json.dump(report, f, indent=2)

print('JSON report written.')

# ── Build markdown summary ───────────────────────────────────
lines = [
    '# Reagent Policy Summary',
    '',
    f'- Total small-kit margin (USD): {total_margin_small:,.2f}',
    f'- Total bulk-kit margin (USD): {total_margin_bulk:,.2f}',
    f'- Absolute difference (USD): {abs_diff:,.2f}',
    f'- Total difference (bulk minus small, USD): {total_diff:,.2f}',
    f'- Decision: {decision}',
    '',
    justification,
]

with open('/root/reagent_policy_summary.md', 'w') as f:
    f.write('\n'.join(lines) + '\n')

print('Markdown summary written.')
```

Run:
```bash
python3 /root/solve.py
```

## Step 3 – Validate outputs

```bash
cat /root/reagent_policy_report.json
cat /root/reagent_policy_summary.md
```

Check:
1. The `metadata` block matches `/root/report_template.json` exactly.
2. `analysis.assumptions` has exactly five keys: `runs_per_year_small_kit`, `runs_per_year_bulk_kit`, `switch_threshold_usd`, `lab_override_rule`, `billing_rule`.
3. `assays` are sorted by `assay_id` ascending.
4. All currency values are rounded to 2 decimal places.
5. The markdown has 4-8 non-empty lines and includes total small-kit margin, total bulk-kit margin, absolute difference, and the decision slug.
6. Currency values in the markdown use comma-separated thousands formatting (e.g., `1,234.56` or `-7,106.39`).

If any check fails, diagnose and fix before finishing.

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
Task metadata: author_email=gpt54@example.com, author_name=GPT-5.4, category=financial-analysis, difficulty=medium, tags=[lab-operations, json, csv, template-update, decision-analysis].
Verifier config: timeout_sec=900.0.