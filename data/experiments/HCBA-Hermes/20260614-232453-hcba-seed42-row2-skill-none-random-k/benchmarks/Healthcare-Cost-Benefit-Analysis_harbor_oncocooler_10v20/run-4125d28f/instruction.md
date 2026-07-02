# Task Instruction

Execute the following steps in order:

1. **Read all input files** before writing any code:
 ```
 cat /root/program_catalog.json
 cat /root/cooler_cost.csv
 cat /root/contract_payment.csv
 cat /root/site_overrides.csv
 ```

2. **Examine the test file** to understand exact verifier expectations:
 ```
 cat /root/test_output.py
 ```

3. **Write a Python script** `/root/solve.py` that implements the analysis. Pay very careful attention to the cooler cost formula — it must be multiplied by `active_sites * dispatches_per_year`, not just `dispatches_per_year` alone. Here is the detailed logic:

 ```python
 import json, csv, math

 # Load inputs
 with open('/root/program_catalog.json') as f:
 catalog = json.load(f)

 def read_csv(path):
 with open(path) as f:
 return list(csv.DictReader(f))

 cooler_rows = read_csv('/root/cooler_cost.csv')
 payment_rows = read_csv('/root/contract_payment.csv')
 override_rows = read_csv('/root/site_overrides.csv')

 # Build cooler cost lookup
 cooler_cost_map = {}
 for r in cooler_rows:
 cooler_cost_map[r['cooler_type'].strip()] = float(r['cooler_cost_usd'])

 # Filter in-scope programs (review_flag == 'review')
 in_scope = [p for p in catalog['programs'] if p.get('review_flag') == 'review']

 # Build label -> program mapping for contract_payment resolution
 label_to_program = {}
 for p in in_scope:
 label_to_program[p['program_name'].strip().lower()] = p
 for lbl in p.get('known_labels', []):
 label_to_program[lbl.strip().lower()] = p

 # Resolve payment per program_code
 payment_map = {} # program_code -> payment_per_dispatch_per_site_usd
 for r in payment_rows:
 pl = r['program_label'].strip().lower()
 prog = label_to_program.get(pl)
 if prog:
 payment_map[prog['program_code']] = float(r['payment_per_dispatch_per_site_usd'])

 # Resolve active sites per program_code from site_overrides
 # Only approved rows, keep highest version_no per program_code
 approved = [r for r in override_rows if r['approval_state'].strip().lower() == 'approved']
 best_override = {}
 for r in approved:
 pc = r['program_code'].strip()
 vn = int(r['version_no'])
 if pc not in best_override or vn > best_override[pc][1]:
 best_override[pc] = (int(r['active_sites']), vn)

 # Constants
 disp10 = 36
 disp20 = 18
 days10 = 10
 days20 = 20
 threshold = 10000

 results = []
 for p in in_scope:
 pc = p['program_code']
 pname = p['program_name']
 acq = float(p['acquisition_cost_per_1000_units_usd'])
 upd = float(p['units_per_day'])
 ct = p['cooler_type'].strip()
 cooler_usd = cooler_cost_map[ct]
 default_sites = int(p['default_active_sites'])

 # Active sites
 if pc in best_override:
 active_sites = best_override[pc][0]
 else:
 active_sites = default_sites

 # Payment
 pay = payment_map.get(pc, 0.0)

 # DRUG COST: acq * active_sites * upd * days * dispatches / 1000
 drug10 = acq * active_sites * upd * days10 * disp10 / 1000.0
 drug20 = acq * active_sites * upd * days20 * disp20 / 1000.0

 # COOLER COST: cooler_usd * active_sites * dispatches_per_year
 # THIS IS CRITICAL: cooler cost is per-dispatch per-site
 cooler10 = cooler_usd * active_sites * disp10
 cooler20 = cooler_usd * active_sites * disp20

 # REVENUE: pay * active_sites * dispatches_per_year
 rev10 = pay * active_sites * disp10
 rev20 = pay * active_sites * disp20

 # MARGIN
 margin10 = rev10 - drug10 - cooler10
 margin20 = rev20 - drug20 - cooler20
 diff = margin20 - margin10

 results.append({
 'program_code': pc,
 'program_name': pname,
 'active_sites': active_sites,
 'acquisition_cost_per_1000_units_usd': round(acq, 2),
 'units_per_day': round(upd, 2),
 'cooler_type': ct,
 'cooler_cost_usd': round(cooler_usd, 2),
 'payment_per_dispatch_per_site_usd': round(pay, 2),
 'annual_drug_cost_10_day_usd': round(drug10, 2),
 'annual_drug_cost_20_day_usd': round(drug20, 2),
 'annual_cooler_cost_10_day_usd': round(cooler10, 2),
 'annual_cooler_cost_20_day_usd': round(cooler20, 2),
 'annual_revenue_10_day_usd': round(rev10, 2),
 'annual_revenue_20_day_usd': round(rev20, 2),
 'annual_margin_10_day_usd': round(margin10, 2),
 'annual_margin_20_day_usd': round(margin20, 2),
 'annual_margin_difference_20_minus_10_usd': round(diff, 2)
 })

 # Sort by program_code ascending
 results.sort(key=lambda x: x['program_code'])

 # Totals
 total_10 = round(sum(r['annual_margin_10_day_usd'] for r in results), 2)
 total_20 = round(sum(r['annual_margin_20_day_usd'] for r in results), 2)
 total_diff = round(total_20 - total_10, 2)
 abs_diff = round(abs(total_diff), 2)

 if abs_diff < threshold:
 decision = 'move_to_20_day'
 justification = f'Absolute margin difference ${abs_diff} is below the ${threshold} threshold; moving to 20-day dispatches is acceptable.'
 else:
 decision = 'keep_10_day'
 justification = f'Absolute margin difference ${abs_diff} exceeds the ${threshold} threshold; keeping 10-day dispatches is recommended.'

 output = {
 'assumptions': {
 'dispatches_per_year_10_day': 36,
 'dispatches_per_year_20_day': 18,
 'days_per_dispatch_10_day': 10,
 'days_per_dispatch_20_day': 20,
 'switch_threshold_usd': 10000,
 'site_override_rule': 'highest approved version_no per program_code, else default_active_sites'
 },
 'programs': results,
 'totals': {
 'total_annual_margin_10_day_usd': total_10,
 'total_annual_margin_20_day_usd': total_20,
 'total_annual_margin_difference_20_minus_10_usd': total_diff,
 'absolute_total_margin_difference_usd': abs_diff
 },
 'recommendation': {
 'decision': decision,
 'justification': justification
 }
 }

 with open('/root/oncocooler_analysis.json', 'w') as f:
 json.dump(output, f, indent=2)

 # Write summary
 lines = [
 '# OncoCooler Dispatch Analysis Summary',
 f'Total 10-day annual margin: ${total_10:,.2f} USD',
 f'Total 20-day annual margin: ${total_20:,.2f} USD',
 f'Absolute margin difference: ${abs_diff:,.2f} USD',
 f'Recommendation: {decision}',
 f'The analysis evaluated {len(results)} in-scope programs comparing 10-day vs 20-day dispatch models.'
 ]
 with open('/root/oncocooler_summary.md', 'w') as f:
 f.write('\n'.join(lines) + '\n')

 print('Done. Files written.')
 print(f'Total 10-day margin: {total_10}')
 print(f'Total 20-day margin: {total_20}')
 print(f'Difference: {total_diff}')
 print(f'Decision: {decision}')
 ```

 **CRITICAL NOTE about the previous failure**: The prior run got 302.4 where 4536.0 was expected (ratio = 15). This strongly suggests the cooler cost was NOT being multiplied by `active_sites`. The formula `cooler_cost_usd * active_sites * dispatches_per_year` is the correct annual cooler cost formula. If `active_sites` was 15 for that program, that explains the exact 15x discrepancy. Make sure the script above uses `cooler_usd * active_sites * disp10` for cooler cost, NOT just `cooler_usd * disp10`.

 However, BEFORE committing to this formula, **read the test file** (`test_output.py`) to check if the test reveals which field had the 302.4 vs 4536.0 mismatch. If it was `annual_cooler_cost_10_day_usd`, then the fix above is correct. If it was a different field, adjust accordingly.

 Also consider: the ratio 4536/302.4 = 15. If active_sites for that program is 15, then the missing `active_sites` multiplier in cooler cost explains it perfectly. But also check: could it be that `units_per_day` needs a different interpretation? Read the catalog carefully.

4. **Run the script**:
 ```
 cd /root && python solve.py
 ```

5. **Validate the outputs**:
 ```
 cat /root/oncocooler_analysis.json
 cat /root/oncocooler_summary.md
 python -c "import json; d=json.load(open('/root/oncocooler_analysis.json')); print('Programs:', len(d['programs'])); print('Sorted:', [p['program_code'] for p in d['programs']]); print('Totals:', d['totals']); print('Decision:', d['recommendation']['decision'])"
 ```

6. **Run the test suite**:
 ```
 cd /root && python -m pytest test_output.py -v
 ```

7. **If tests fail**, read the error messages carefully. The most likely issues are:
 - Cooler cost formula (per-site vs flat) — check which field mismatches
 - Site override resolution (wrong active_sites count)
 - Payment label matching (case sensitivity, whitespace)
 - Check if `annual_cooler_cost` in the task spec means `cooler_cost_usd * dispatches_per_year` (flat, not per-site). Re-read the spec: it says "Cooler cost uses cooler_cost_usd from cooler_cost.csv" — this is ambiguous about whether it's per-site. The test expectations will tell you.
 
 If the 302.4 vs 4536.0 mismatch was actually in `annual_cooler_cost_10_day_usd` and the formula should be `cooler_usd * active_sites * dispatches`, then the script is correct. If the test shows the mismatch is elsewhere, debug from the test output.

8. **Iterate** until all tests pass. Re-read files and test output each time before making changes.

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
Task metadata: author_email=gpt54@example.com, author_name=GPT-5.4, category=financial-analysis, difficulty=medium, tags=[oncology, json, csv, structural-adaptation, decision-analysis].
Verifier config: timeout_sec=900.0.