# Task Instruction

Execute the following steps carefully to produce `/root/vaxcrate_analysis.json` and `/root/vaxcrate_summary.md`.

## Step 1: Inspect all input files

Read and display the full contents of each input file:
```
cat /root/campaign_manifest.json
cat /root/crate_cost.csv
cat /root/billing.csv
cat /root/location_overrides.csv
cat /root/suspensions.csv
```

## Step 2: Write and run a Python script

Create `/root/solve.py` with the following logic. Be very precise with every formula.

```python
import json, csv, math

# Load campaign_manifest.json
with open('/root/campaign_manifest.json') as f:
    manifest = json.load(f)

# Load crate_cost.csv
crate_costs = {}
with open('/root/crate_cost.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        crate_costs[row['crate_tier'].strip()] = float(row['crate_cost_usd'].strip())

# Load billing.csv
billing_rows = []
with open('/root/billing.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        billing_rows.append({k.strip(): v.strip() for k, v in row.items()})

# Load location_overrides.csv
override_rows = []
with open('/root/location_overrides.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        override_rows.append({k.strip(): v.strip() for k, v in row.items()})

# Load suspensions.csv
suspended_ids = set()
with open('/root/suspensions.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        row = {k.strip(): v.strip() for k, v in row.items()}
        if row.get('suspension_status', '').lower() == 'hold':
            suspended_ids.add(row['campaign_id'].strip())

# Get campaigns list from manifest (could be a list or dict)
if isinstance(manifest, list):
    campaigns_list = manifest
else:
    # might be {"campaigns": [...]} or similar
    campaigns_list = manifest.get('campaigns', [manifest] if 'campaign_id' in manifest else list(manifest.values()) if isinstance(list(manifest.values())[0], list) else [manifest])
    if not isinstance(campaigns_list, list):
        campaigns_list = [campaigns_list]

print("All campaign IDs:", [c.get('campaign_id') for c in campaigns_list])
print("Suspended IDs:", suspended_ids)

# Filter: analysis_flag == 'review' and not suspended
retained = []
for c in campaigns_list:
    if str(c.get('analysis_flag', '')).strip().lower() != 'review':
        continue
    if c['campaign_id'].strip() in suspended_ids:
        continue
    retained.append(c)

print("Retained campaign IDs:", [c['campaign_id'] for c in retained])

# Build name->campaign and alias->campaign mappings
name_to_campaign = {}
alias_to_campaign = {}
for c in retained:
    cid = c['campaign_id'].strip()
    cname = c['campaign_name'].strip()
    name_to_campaign[cname.lower()] = c
    aliases = c.get('alias_labels', [])
    if isinstance(aliases, str):
        aliases = [a.strip() for a in aliases.split(',')]
    for a in aliases:
        alias_to_campaign[a.strip().lower()] = c

# Resolve billing rows to campaigns
# For each retained campaign, find all active billing rows that match, keep latest cycle_tag
campaign_billing = {}  # campaign_id -> billing row
for brow in billing_rows:
    if brow.get('status', '').lower() != 'active':
        continue
    label = brow.get('campaign_label', '').strip().lower()
    matched_campaign = name_to_campaign.get(label) or alias_to_campaign.get(label)
    if matched_campaign is None:
        continue
    cid = matched_campaign['campaign_id'].strip()
    existing = campaign_billing.get(cid)
    if existing is None or brow['cycle_tag'].strip() > existing['cycle_tag'].strip():
        campaign_billing[cid] = brow

print("Billing matches:", {k: v.get('cycle_tag') for k, v in campaign_billing.items()})

# Resolve active_clinics from location_overrides
campaign_clinics = {}  # campaign_id -> active_clinics
for orow in override_rows:
    if orow.get('state', '').lower() != 'approved':
        continue
    rev = orow.get('revision', '').strip()
    ac = orow.get('active_clinics', '').strip()
    if rev == '' or ac == '':
        continue
    cid = orow.get('campaign_id', '').strip()
    # Check if this campaign is retained
    if cid not in [c['campaign_id'].strip() for c in retained]:
        continue
    rev_num = float(rev)
    if cid not in campaign_clinics or rev_num > campaign_clinics[cid][0]:
        campaign_clinics[cid] = (rev_num, int(float(ac)))

print("Override clinics:", campaign_clinics)

# Build results
results = []
for c in retained:
    cid = c['campaign_id'].strip()
    cname = c['campaign_name'].strip()
    
    # Active clinics
    if cid in campaign_clinics:
        active_clinics = campaign_clinics[cid][1]
    else:
        active_clinics = int(float(c['default_active_clinics']))
    
    drug_cost_per_1000 = float(c['drug_cost_per_1000_doses_usd'])
    doses_per_day = float(c['doses_per_day'])
    crate_tier = c['crate_tier'].strip()
    crate_cost = crate_costs[crate_tier]
    
    brow = campaign_billing.get(cid)
    if brow is None:
        print(f"WARNING: No billing row for {cid}")
        continue
    payment_per_dispatch = float(brow['payment_per_dispatch_per_clinic_usd'])
    
    # 6-day model
    days_6 = 6
    disp_6 = 60
    revenue_6 = payment_per_dispatch * active_clinics * disp_6
    drug_cost_6 = drug_cost_per_1000 * active_clinics * doses_per_day * days_6 * disp_6 / 1000.0
    crate_cost_6 = crate_cost * disp_6  # NOTE: crate cost per dispatch, NOT per clinic
    margin_6 = revenue_6 - drug_cost_6 - crate_cost_6
    
    # 12-day model
    days_12 = 12
    disp_12 = 30
    revenue_12 = payment_per_dispatch * active_clinics * disp_12
    drug_cost_12 = drug_cost_per_1000 * active_clinics * doses_per_day * days_12 * disp_12 / 1000.0
    crate_cost_12 = crate_cost * disp_12
    margin_12 = revenue_12 - drug_cost_12 - crate_cost_12
    
    diff = margin_12 - margin_6
    
    results.append({
        'campaign_id': cid,
        'campaign_name': cname,
        'active_clinics': active_clinics,
        'drug_cost_per_1000_doses_usd': round(drug_cost_per_1000, 2),
        'doses_per_day': round(doses_per_day, 2),
        'crate_tier': crate_tier,
        'crate_cost_usd': round(crate_cost, 2),
        'payment_per_dispatch_per_clinic_usd': round(payment_per_dispatch, 2),
        'annual_drug_cost_6_day_usd': round(drug_cost_6, 2),
        'annual_drug_cost_12_day_usd': round(drug_cost_12, 2),
        'annual_crate_cost_6_day_usd': round(crate_cost_6, 2),
        'annual_crate_cost_12_day_usd': round(crate_cost_12, 2),
        'annual_revenue_6_day_usd': round(revenue_6, 2),
        'annual_revenue_12_day_usd': round(revenue_12, 2),
        'annual_margin_6_day_usd': round(margin_6, 2),
        'annual_margin_12_day_usd': round(margin_12, 2),
        'annual_margin_difference_12_minus_6_usd': round(diff, 2)
    })

# Sort by campaign_id ascending
results.sort(key=lambda x: x['campaign_id'])

# Print per-campaign for debugging
for r in results:
    print(f"Campaign {r['campaign_id']}: clinics={r['active_clinics']}, rev6={r['annual_revenue_6_day_usd']}, drug6={r['annual_drug_cost_6_day_usd']}, crate6={r['annual_crate_cost_6_day_usd']}, margin6={r['annual_margin_6_day_usd']}, margin12={r['annual_margin_12_day_usd']}, diff={r['annual_margin_difference_12_minus_6_usd']}")

total_margin_6 = round(sum(r['annual_margin_6_day_usd'] for r in results), 2)
total_margin_12 = round(sum(r['annual_margin_12_day_usd'] for r in results), 2)
total_diff = round(sum(r['annual_margin_difference_12_minus_6_usd'] for r in results), 2)
abs_diff = round(abs(total_diff), 2)

if abs_diff < 11000:
    decision = 'move_to_12_day'
    justification = f'Absolute total margin difference of ${abs_diff} is below the $11,000 threshold, recommending move to 12-day dispatch.'
else:
    decision = 'keep_6_day'
    justification = f'Absolute total margin difference of ${abs_diff} exceeds the $11,000 threshold, recommending keeping 6-day dispatch.'

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

print("\nTotals: margin6=", total_margin_6, "margin12=", total_margin_12, "diff=", total_diff, "abs_diff=", abs_diff)
print("Decision:", decision)

# Write summary
with open('/root/vaxcrate_summary.md', 'w') as f:
    f.write(f'# VaxCrate 6-Day vs 12-Day Dispatch Analysis Summary\n')
    f.write(f'\n')
    f.write(f'Total 6-day annual margin: ${total_margin_6:,.2f} USD\n')
    f.write(f'Total 12-day annual margin: ${total_margin_12:,.2f} USD\n')
    f.write(f'Absolute margin difference: ${abs_diff:,.2f} USD\n')
    f.write(f'\n')
    f.write(f'Final decision: {decision}\n')
    f.write(f'\n')
    f.write(f'{justification}\n')

print("Done.")
```

Run: `python3 /root/solve.py`

## Step 3: Examine the debug output

Look carefully at the printed debug output. The previous run failed because:
- A campaign had `annual_margin_6_day_usd` of 468 instead of expected 7956. The difference (7956 - 468 = 7488) is exactly 60 * 124.8, which suggests `crate_cost_6_day` was wrong.
- The total margin was -45894.84 instead of -83406.84, a difference of 37512 which is 5 * 7488 + some remainder.

This strongly suggests the **crate cost should be multiplied by active_clinics**. If the first run's output shows numbers that don't match the expected values, we need to change the crate cost formula.

Specifically, check: does `annual_crate_cost = crate_cost * dispatches_per_year` give the expected margin, or does `annual_crate_cost = crate_cost * active_clinics * dispatches_per_year` give the expected margin?

From the feedback: expected margin_6 = 7956, got 468. If the difference is due to crate cost being too low (not multiplied by clinics), then crate_cost_6 should be higher.

Actually wait - if margin was 468 instead of 7956, margin was TOO LOW, meaning costs were too HIGH or revenue too low. Let me reconsider: 468 < 7956 means the script computed a lower margin. If crate cost was NOT multiplied by clinics, the cost would be LOWER and margin HIGHER. So the issue might be the opposite - maybe the script already multiplied by clinics when it shouldn't have, OR there's a different issue.

Let me reconsider: the previous feedback says got 468 expected 7956. The difference is 7488. If we look at this differently, maybe the billing join was wrong (wrong payment amount) or active_clinics was wrong.

**IMPORTANT**: After examining the debug output, if the numbers don't match the expected values from the test (margin_6 of 7956 for some campaign, total of -83406.84), try the alternative crate cost formula:

`crate_cost_annual = crate_cost * active_clinics * dispatches_per_year`

Modify the script accordingly and re-run.

## Step 4: Validate

After the script produces output:
1. `cat /root/vaxcrate_analysis.json` - verify the JSON is valid and has all required keys
2. `cat /root/vaxcrate_summary.md` - verify 4-8 non-empty lines with required content
3. Run the test if available: `cd /root && python -m pytest test_output.py -v 2>&1 | head -80`

If the test fails, read the error messages carefully, identify which values are wrong, trace back through the formulas, fix the script, and re-run. Pay special attention to:
- Whether crate cost should be per-dispatch or per-clinic-per-dispatch
- Whether the billing join matched the right rows
- Whether active_clinics values are correct
- Whether cycle_tag comparison is string-based (lexicographic) and correct

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