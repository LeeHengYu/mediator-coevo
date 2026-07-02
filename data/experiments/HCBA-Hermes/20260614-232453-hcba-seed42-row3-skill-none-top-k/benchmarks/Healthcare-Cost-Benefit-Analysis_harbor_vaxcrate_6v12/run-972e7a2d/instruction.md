# Task Instruction

Perform the following steps exactly:

## Step 1 – Inspect all input files

Read and display the full contents of every input file:
```
cat /root/campaign_manifest.json
cat /root/crate_cost.csv
cat /root/billing.csv
cat /root/location_overrides.csv
cat /root/suspensions.csv
```

## Step 2 – Write and run a Python script

Create `/root/solve.py` with the logic below, then run it with `python3 /root/solve.py`.

```python
import json, csv, math, os

# ── Load inputs ──────────────────────────────────────────────
with open('/root/campaign_manifest.json') as f:
    manifest = json.load(f)

def read_csv(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))

crate_cost_rows = read_csv('/root/crate_cost.csv')
billing_rows = read_csv('/root/billing.csv')
override_rows = read_csv('/root/location_overrides.csv')
suspension_rows = read_csv('/root/suspensions.csv')

# ── Build lookup structures ──────────────────────────────────
# Crate cost lookup  (strip whitespace from keys)
crate_cost_map = {}
for r in crate_cost_rows:
    tier = r.get('crate_tier','').strip()
    cost = r.get('crate_cost_usd','').strip()
    if tier and cost:
        crate_cost_map[tier] = float(cost)

# Suspended campaign_ids with hold
suspended_ids = set()
for r in suspension_rows:
    if r.get('suspension_status','').strip().lower() == 'hold':
        suspended_ids.add(r.get('campaign_id','').strip())

# ── Filter campaigns ────────────────────────────────────────
# manifest can be a list or dict; normalise
if isinstance(manifest, dict):
    campaigns_list = manifest.get('campaigns', [manifest])
else:
    campaigns_list = manifest

retained = []
for c in campaigns_list:
    if str(c.get('analysis_flag','')).strip().lower() != 'review':
        continue
    cid = str(c.get('campaign_id','')).strip()
    if cid in suspended_ids:
        continue
    retained.append(c)

# ── Resolve billing rows ────────────────────────────────────
# Build map: campaign_id -> (campaign_name, alias_labels list)
for c in retained:
    cid = c['campaign_id']
    cname = str(c.get('campaign_name','')).strip()
    aliases = c.get('alias_labels', [])
    if isinstance(aliases, str):
        aliases = [a.strip() for a in aliases.split(',') if a.strip()]
    all_labels = set()
    if cname:
        all_labels.add(cname)
    for a in aliases:
        all_labels.add(str(a).strip())
    c['_all_labels'] = all_labels

def resolve_billing(campaign):
    labels = campaign['_all_labels']
    candidates = []
    for br in billing_rows:
        if br.get('status','').strip().lower() != 'active':
            continue
        bl = br.get('campaign_label','').strip()
        if bl in labels:
            candidates.append(br)
    if not candidates:
        return None
    # keep latest cycle_tag
    candidates.sort(key=lambda r: r.get('cycle_tag',''), reverse=True)
    return candidates[0]

# ── Resolve active_clinics via overrides ────────────────────
def resolve_clinics(campaign):
    cid = str(campaign['campaign_id']).strip()
    candidates = []
    for r in override_rows:
        if r.get('campaign_id','').strip() != cid:
            continue
        if r.get('state','').strip().lower() != 'approved':
            continue
        rev = r.get('revision','').strip()
        ac = r.get('active_clinics','').strip()
        if rev == '' or ac == '':
            continue
        try:
            rev_num = float(rev)
            ac_num = float(ac)
        except ValueError:
            continue
        candidates.append((rev_num, ac_num, r))
    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        return int(candidates[0][1])
    # fallback
    return int(campaign.get('default_active_clinics', 0))

# ── Compute per-campaign numbers ────────────────────────────
results = []
for c in retained:
    cid = str(c['campaign_id']).strip()
    cname = str(c.get('campaign_name','')).strip()
    active_clinics = resolve_clinics(c)
    drug_cost_per_1000 = float(c.get('drug_cost_per_1000_doses_usd', 0))
    doses_per_day = float(c.get('doses_per_day', 0))
    crate_tier = str(c.get('crate_tier','')).strip()
    crate_cost = crate_cost_map.get(crate_tier, 0.0)

    billing = resolve_billing(c)
    if billing is None:
        payment = 0.0
    else:
        payment = float(billing.get('payment_per_dispatch_per_clinic_usd', 0))

    # 6-day model
    d6, disp6 = 6, 60
    rev_6 = payment * active_clinics * disp6
    drug_6 = drug_cost_per_1000 * active_clinics * doses_per_day * d6 * disp6 / 1000
    crate_6 = crate_cost * disp6   # crate cost is per dispatch (one crate per dispatch)
    # Wait – the instructions say "annual_crate_cost" but don't specify formula.
    # The crate_cost_usd is the cost of a crate. Each dispatch uses one crate.
    # So annual_crate_cost = crate_cost_usd * dispatches_per_year
    margin_6 = rev_6 - drug_6 - crate_6

    # 12-day model
    d12, disp12 = 12, 30
    rev_12 = payment * active_clinics * disp12
    drug_12 = drug_cost_per_1000 * active_clinics * doses_per_day * d12 * disp12 / 1000
    crate_12 = crate_cost * disp12
    margin_12 = rev_12 - drug_12 - crate_12

    diff = margin_12 - margin_6

    results.append({
        'campaign_id': cid,
        'campaign_name': cname,
        'active_clinics': active_clinics,
        'drug_cost_per_1000_doses_usd': round(drug_cost_per_1000, 2),
        'doses_per_day': round(doses_per_day, 2),
        'crate_tier': crate_tier,
        'crate_cost_usd': round(crate_cost, 2),
        'payment_per_dispatch_per_clinic_usd': round(payment, 2),
        'annual_drug_cost_6_day_usd': round(drug_6, 2),
        'annual_drug_cost_12_day_usd': round(drug_12, 2),
        'annual_crate_cost_6_day_usd': round(crate_6, 2),
        'annual_crate_cost_12_day_usd': round(crate_12, 2),
        'annual_revenue_6_day_usd': round(rev_6, 2),
        'annual_revenue_12_day_usd': round(rev_12, 2),
        'annual_margin_6_day_usd': round(margin_6, 2),
        'annual_margin_12_day_usd': round(margin_12, 2),
        'annual_margin_difference_12_minus_6_usd': round(diff, 2)
    })

# Sort by campaign_id ascending
results.sort(key=lambda x: x['campaign_id'])

total_margin_6 = round(sum(r['annual_margin_6_day_usd'] for r in results), 2)
total_margin_12 = round(sum(r['annual_margin_12_day_usd'] for r in results), 2)
total_diff = round(sum(r['annual_margin_difference_12_minus_6_usd'] for r in results), 2)
abs_diff = round(abs(total_diff), 2)

if abs_diff < 11000:
    decision = 'move_to_12_day'
else:
    decision = 'keep_6_day'

justification = (f'The absolute total margin difference is ${abs_diff:,.2f}, '
                 f'which is {"below" if abs_diff < 11000 else "at or above"} '
                 f'the $11,000 threshold, so the recommendation is {decision}.')

output = {
    'assumptions': {
        'dispatches_per_year_6_day': 60,
        'dispatches_per_year_12_day': 30,
        'days_per_dispatch_6_day': 6,
        'days_per_dispatch_12_day': 12,
        'switch_threshold_usd': 11000,
        'override_rule': 'highest numeric approved revision with non-empty active_clinics, else default_active_clinics',
        'suspension_rule': 'exclude hold campaigns'
    },
    'campaigns': results,
    'totals': {
        'total_annual_margin_6_day_usd': total_margin_6,
        'total_annual_margin_12_day_usd': total_margin_12,
        'total_annual_margin_difference_12_minus_6_usd': total_diff,
        'absolute_total_margin_difference_usd': abs_diff
    },
    'recommendation': {
        'decision': decision,
        'justification': justification
    }
}

with open('/root/vaxcrate_analysis.json', 'w') as f:
    json.dump(output, f, indent=2)

print('=== JSON written ===')
print(json.dumps(output, indent=2))

# ── Write summary markdown ──────────────────────────────────
lines = [
    '# VaxCrate 6-Day vs 12-Day Dispatch Analysis Summary',
    '',
    f'- Total 6-day annual margin: ${total_margin_6:,.2f} USD',
    f'- Total 12-day annual margin: ${total_margin_12:,.2f} USD',
    f'- Absolute difference: ${abs_diff:,.2f} USD',
    f'- Decision: {decision}',
    '',
    justification
]

with open('/root/vaxcrate_summary.md', 'w') as f:
    f.write('\n'.join(lines) + '\n')

print('\n=== Summary written ===')
for l in lines:
    print(l)
```

## Step 3 – Validate outputs

After the script completes:

1. `cat /root/vaxcrate_analysis.json` – confirm it is valid JSON matching the required schema, campaigns sorted by campaign_id, all currency values rounded to 2 decimals.
2. `cat /root/vaxcrate_summary.md` – confirm 4-8 non-empty lines, includes total 6-day margin, total 12-day margin, absolute difference, and the exact decision slug.
3. `python3 -c "import json; d=json.load(open('/root/vaxcrate_analysis.json')); print('campaigns:', len(d['campaigns'])); print('decision:', d['recommendation']['decision'])"` – quick sanity check.

## Step 4 – Inspect and fix if needed

If the script fails or the outputs look wrong:
- Re-read the input files carefully, paying attention to column names (whitespace, casing).
- Check whether `campaign_manifest.json` is a top-level list or has a wrapper object.
- Check whether `alias_labels` is a list or comma-separated string.
- Re-examine the crate cost formula: the task says `annual_crate_cost` uses `crate_cost_usd` from `crate_cost.csv` matched by `crate_tier`. The simplest interpretation is `crate_cost_usd * dispatches_per_year` (one crate per dispatch). If results seem off, consider whether crate cost might be per-clinic-per-dispatch (`crate_cost_usd * active_clinics * dispatches_per_year`) and re-check by looking at whether the numbers make sense relative to revenue.
- Fix and re-run until both output files are correct.

Do NOT skip any step. Show all intermediate output so issues can be diagnosed.

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
Task metadata: author_email=gpt54@example.com, author_name=GPT-5.4, category=financial-analysis, difficulty=medium, tags=[vaccination, json, csv, distractor-handling, decision-analysis].
Verifier config: timeout_sec=900.0.