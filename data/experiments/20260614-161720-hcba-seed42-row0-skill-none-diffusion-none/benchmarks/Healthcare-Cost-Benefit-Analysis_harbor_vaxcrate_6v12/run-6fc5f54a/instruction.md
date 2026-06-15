# Task Instruction

Execute the following Python script to produce both output files. Before running, inspect all five input files to understand their structure.

```bash
cat /root/campaign_manifest.json
cat /root/crate_cost.csv
cat /root/billing.csv
cat /root/location_overrides.csv
cat /root/suspensions.csv
```

Then run this Python script:

```python
import json
import csv

# Load campaign manifest
with open('/root/campaign_manifest.json', 'r') as f:
    manifest = json.load(f)

# Load crate costs
crate_costs = {}
with open('/root/crate_cost.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        crate_costs[row['crate_tier'].strip()] = float(row['crate_cost_usd'].strip())

# Load billing
billing_rows = []
with open('/root/billing.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        billing_rows.append({k.strip(): v.strip() for k, v in row.items()})

# Load location overrides
overrides = []
with open('/root/location_overrides.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        overrides.append({k.strip(): v.strip() for k, v in row.items()})

# Load suspensions
suspended = set()
with open('/root/suspensions.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        row = {k.strip(): v.strip() for k, v in row.items()}
        if row.get('suspension_status', '').lower() == 'hold':
            suspended.add(row['campaign_id'].strip())

# Build campaign lookup
campaigns_list = manifest if isinstance(manifest, list) else manifest.get('campaigns', [])

# Filter: analysis_flag == 'review' and not suspended
retained = []
for c in campaigns_list:
    if c.get('analysis_flag', '').strip().lower() != 'review':
        continue
    cid = c['campaign_id'].strip()
    if cid in suspended:
        continue
    retained.append(c)

# Build name/alias -> campaign mapping for billing resolution
name_to_campaign = {}
for c in retained:
    cid = c['campaign_id'].strip()
    cname = c['campaign_name'].strip()
    name_to_campaign[cname] = c
    aliases = c.get('alias_labels', [])
    if isinstance(aliases, str):
        aliases = [a.strip() for a in aliases.split(',')]
    for a in aliases:
        a = a.strip()
        if a:
            name_to_campaign[a] = c

# Resolve billing: match campaign_label to campaign_name or alias_labels
# Only active rows, keep latest cycle_tag per campaign
billing_by_campaign = {}  # campaign_id -> billing row
for br in billing_rows:
    if br.get('status', '').lower() != 'active':
        continue
    label = br.get('campaign_label', '').strip()
    if label in name_to_campaign:
        c = name_to_campaign[label]
        cid = c['campaign_id'].strip()
        cycle = br.get('cycle_tag', '').strip()
        if cid not in billing_by_campaign or cycle > billing_by_campaign[cid]['cycle_tag']:
            billing_by_campaign[cid] = br

# Resolve active clinics from location_overrides
# approved state, non-blank revision and active_clinics, highest numeric revision
override_by_campaign = {}  # campaign_id -> override row
for o in overrides:
    if o.get('state', '').lower() != 'approved':
        continue
    rev = o.get('revision', '').strip()
    ac = o.get('active_clinics', '').strip()
    if rev == '' or ac == '':
        continue
    try:
        rev_num = float(rev)
    except ValueError:
        continue
    cid = o.get('campaign_id', '').strip()
    if cid not in override_by_campaign:
        override_by_campaign[cid] = (rev_num, o)
    else:
        if rev_num > override_by_campaign[cid][0]:
            override_by_campaign[cid] = (rev_num, o)

# Build output campaigns
result_campaigns = []
for c in retained:
    cid = c['campaign_id'].strip()
    cname = c['campaign_name'].strip()
    
    # Active clinics
    if cid in override_by_campaign:
        active_clinics = int(float(override_by_campaign[cid][1]['active_clinics'].strip()))
    else:
        active_clinics = int(float(c.get('default_active_clinics', 0)))
    
    drug_cost_per_1000 = float(c.get('drug_cost_per_1000_doses_usd', 0))
    doses_per_day = float(c.get('doses_per_day', 0))
    crate_tier = c.get('crate_tier', '').strip()
    crate_cost = crate_costs.get(crate_tier, 0.0)
    
    # Billing
    if cid in billing_by_campaign:
        payment = float(billing_by_campaign[cid].get('payment_per_dispatch_per_clinic_usd', 0))
    else:
        payment = 0.0
    
    # 6-day model
    days_6 = 6
    disp_6 = 60
    rev_6 = payment * active_clinics * disp_6
    drug_6 = drug_cost_per_1000 * active_clinics * doses_per_day * days_6 * disp_6 / 1000
    crate_6 = crate_cost * active_clinics * disp_6
    margin_6 = rev_6 - drug_6 - crate_6
    
    # 12-day model
    days_12 = 12
    disp_12 = 30
    rev_12 = payment * active_clinics * disp_12
    drug_12 = drug_cost_per_1000 * active_clinics * doses_per_day * days_12 * disp_12 / 1000
    crate_12 = crate_cost * active_clinics * disp_12
    margin_12 = rev_12 - drug_12 - crate_12
    
    diff = margin_12 - margin_6
    
    result_campaigns.append({
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
result_campaigns.sort(key=lambda x: x['campaign_id'])

# Totals
total_margin_6 = round(sum(c['annual_margin_6_day_usd'] for c in result_campaigns), 2)
total_margin_12 = round(sum(c['annual_margin_12_day_usd'] for c in result_campaigns), 2)
total_diff = round(sum(c['annual_margin_difference_12_minus_6_usd'] for c in result_campaigns), 2)
abs_diff = round(abs(total_diff), 2)

# Decision
if abs_diff < 11000:
    decision = 'move_to_12_day'
    justification = f'Absolute total margin difference of ${abs_diff:.2f} is below the $11,000 threshold, recommending move to 12-day dispatch.'
else:
    decision = 'keep_6_day'
    justification = f'Absolute total margin difference of ${abs_diff:.2f} exceeds the $11,000 threshold, recommending keeping 6-day dispatch.'

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
    'campaigns': result_campaigns,
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

print('JSON written successfully.')
print(json.dumps(output, indent=2))

# Write summary markdown - NO comma formatting for numbers
with open('/root/vaxcrate_summary.md', 'w') as f:
    f.write('# VaxCrate 6-Day vs 12-Day Dispatch Analysis Summary\n')
    f.write('\n')
    f.write(f'Total 6-day margin USD: {total_margin_6:.2f}\n')
    f.write(f'Total 12-day margin USD: {total_margin_12:.2f}\n')
    f.write(f'Total margin difference (12 minus 6): {total_diff:.2f}\n')
    f.write(f'Absolute difference USD: {abs_diff:.2f}\n')
    f.write(f'Decision: {decision}\n')
    f.write(f'Justification: {justification}\n')

print('\nSummary markdown written successfully.')
with open('/root/vaxcrate_summary.md', 'r') as f:
    print(f.read())
```

After running the script, verify both output files exist and display their contents:
```bash
cat /root/vaxcrate_analysis.json
cat /root/vaxcrate_summary.md
```

CRITICAL FORMATTING NOTES:
- In the markdown summary, do NOT use comma-separated number formatting. Use `f'{value:.2f}'` not `f'{value:,.2f}'`. This is essential - the previous run failed exactly because commas were used.
- The cross-task feedback is CONTRADICTORY: one task wanted commas, another didn't. For THIS specific task (harbor_vaxcrate_6v12), the direct feedback says NO commas. Follow the direct feedback.
- Round all currency values to 2 decimal places.
- Sort campaigns array by campaign_id ascending.
- The annual_crate_cost formula is: `crate_cost_usd * active_clinics * dispatches_per_year` (similar to the successful oncocooler task pattern).
- Make sure to inspect the input files first to understand their exact column names and structure before running the computation script. If column names differ from expectations, adjust the script accordingly.

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