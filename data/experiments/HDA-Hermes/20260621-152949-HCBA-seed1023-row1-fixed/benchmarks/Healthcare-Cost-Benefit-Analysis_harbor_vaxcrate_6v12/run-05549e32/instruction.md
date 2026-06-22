# Task Instruction

Execute the following steps in order:

1. **Read all input files** to understand the data:
 ```
 cat /root/campaign_manifest.json
 cat /root/crate_cost.csv
 cat /root/billing.csv
 cat /root/location_overrides.csv
 cat /root/suspensions.csv
 ```

2. **Examine the test/verifier file** to understand exact expectations:
 ```
 find /root -name '*.py' -path '*/test*' | head -20
 ```
 Then read any test files found (e.g., `cat /root/tests/test_output.py` or similar).

3. **Write a Python script** `/root/solve.py` that implements the full analysis. The script must:

 a. Load `campaign_manifest.json`. Filter campaigns where `analysis_flag == "review"`.
 
 b. Load `suspensions.csv`. Exclude any campaign whose `campaign_id` appears with `suspension_status == "hold"`.
 
 c. Load `billing.csv`. For each retained campaign, match billing rows by checking if `campaign_label` equals the campaign's `campaign_name` OR if `campaign_label` appears in the campaign's `alias_labels` list. Keep only rows where `status == "active"`. If multiple active rows match, keep the one with the latest `cycle_tag` (sort lexicographically/chronologically as appropriate).
 
 d. Load `location_overrides.csv`. For each retained campaign, filter rows matching `campaign_id` where `state == "approved"`. Discard rows where `revision` is blank/empty or `active_clinics` is blank/empty. Among remaining rows, keep the one with the highest numeric `revision`. If no valid row exists, use `default_active_clinics` from the manifest.
 
 e. Look up `crate_cost_usd` from `crate_cost.csv` by matching the campaign's `crate_tier`.
 
 f. Compute for each campaign:
 - `annual_revenue_6_day = payment_per_dispatch_per_clinic_usd * active_clinics * 60`
 - `annual_revenue_12_day = payment_per_dispatch_per_clinic_usd * active_clinics * 30`
 - `annual_drug_cost_6_day = drug_cost_per_1000_doses_usd * active_clinics * doses_per_day * 6 * 60 / 1000`
 - `annual_drug_cost_12_day = drug_cost_per_1000_doses_usd * active_clinics * doses_per_day * 12 * 30 / 1000`
 - `annual_crate_cost_6_day = crate_cost_usd * 60`
 - `annual_crate_cost_12_day = crate_cost_usd * 30`
 - `annual_margin_6_day = annual_revenue_6_day - annual_drug_cost_6_day - annual_crate_cost_6_day`
 - `annual_margin_12_day = annual_revenue_12_day - annual_drug_cost_12_day - annual_crate_cost_12_day`
 - `annual_margin_difference_12_minus_6 = annual_margin_12_day - annual_margin_6_day`
 
 g. Round ALL currency values to 2 decimal places.
 
 h. Sort campaigns by `campaign_id` ascending.
 
 i. Compute totals:
 - `total_annual_margin_6_day_usd` = sum of all `annual_margin_6_day_usd`
 - `total_annual_margin_12_day_usd` = sum of all `annual_margin_12_day_usd`
 - `total_annual_margin_difference_12_minus_6_usd` = sum of all per-campaign differences
 - `absolute_total_margin_difference_usd` = abs(total_annual_margin_difference_12_minus_6_usd)
 
 j. Decision: if `abs(total_difference) < 11000` → `move_to_12_day`, else `keep_6_day`.
 
 k. Write `/root/vaxcrate_analysis.json` with the exact schema specified (including the `assumptions` block with exact keys and values).
 
 l. Write `/root/vaxcrate_summary.md` with 4-8 non-empty lines including: total 6-day margin (USD), total 12-day margin (USD), absolute difference (USD), and the exact decision slug (`move_to_12_day` or `keep_6_day`).

4. **Run the script:**
 ```
 cd /root && python solve.py
 ```

5. **Validate the outputs:**
 ```
 cat /root/vaxcrate_analysis.json
 cat /root/vaxcrate_summary.md
 python -c "import json; d=json.load(open('/root/vaxcrate_analysis.json')); print('campaigns:', len(d['campaigns'])); print('decision:', d['recommendation']['decision']); print('totals:', d['totals'])"
 ```

6. **Run the verifier tests** if they exist:
 ```
 cd /root && python -m pytest tests/ -v 2>&1 | head -80
 ```
 If tests fail, read the error messages carefully, fix the issue in `solve.py`, re-run, and re-verify.

**Critical implementation notes:**
- When matching `campaign_label` from billing to campaigns, be careful: `alias_labels` in the manifest is a list. Check if `campaign_label` equals `campaign_name` OR is contained in `alias_labels`.
- `cycle_tag` comparison: inspect the actual format in billing.csv. If it looks like dates or version strings, sort appropriately (lexicographic sort usually works for ISO dates or `YYYY-QN` style tags).
- For `revision` in location_overrides.csv, convert to numeric (int or float) for comparison. Blank means the string is empty after stripping.
- For `active_clinics`, blank means the string is empty after stripping. Convert valid values to int/float.
- `crate_tier` matching must be exact string match.
- The `annual_crate_cost` formulas use `crate_cost_usd * dispatches_per_year` (NOT multiplied by active_clinics — crate cost is per-dispatch, not per-clinic-per-dispatch). Double-check this against the task description: the task says `crate_cost_usd` from `crate_cost.csv` without mentioning clinics in the crate cost formula. The drug cost and revenue formulas explicitly mention `active_clinics` but the crate cost formula is just listed as `annual_crate_cost` without a formula. Look at the test expectations to confirm. **IMPORTANT**: The task does NOT give an explicit formula for annual_crate_cost. Look at the test file to see what's expected. If the test checks specific values, reverse-engineer whether crate cost is `crate_cost_usd * dispatches_per_year` or `crate_cost_usd * active_clinics * dispatches_per_year`. Start with `crate_cost_usd * dispatches_per_year` (no clinics multiplier) as the simpler interpretation, but if tests fail, try the other.
- Round each individual currency field to 2 decimals. Also round totals to 2 decimals.
- The `justification` string should briefly explain the decision referencing the threshold and the absolute difference.

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