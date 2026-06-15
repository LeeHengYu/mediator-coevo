# Task Instruction

Execute the following steps in order:

1. **Read all input files** before writing any code:
   - `cat /root/campaign_manifest.json`
   - `cat /root/crate_cost.csv`
   - `cat /root/billing.csv`
   - `cat /root/location_overrides.csv`
   - `cat /root/suspensions.csv`

2. **Write a Python script** `/root/solve.py` that performs the full analysis. The script must follow these exact rules:

   **a. Load data:**
   - Load `campaign_manifest.json` as JSON (it contains a list of campaign objects).
   - Load `crate_cost.csv`, `billing.csv`, `location_overrides.csv`, `suspensions.csv` as CSV files.

   **b. Filter campaigns:**
   - Keep only campaigns where `analysis_flag == "review"`.
   - Exclude any campaign whose `campaign_id` appears in `suspensions.csv` with `suspension_status == "hold"`.

   **c. Resolve billing rows:**
   - For each retained campaign, find rows in `billing.csv` where `campaign_label` matches either `campaign_name` or any entry in the campaign's `alias_labels` list.
   - Keep only billing rows with `status == "active"`.
   - If multiple active rows match, keep the one with the latest (lexicographically greatest) `cycle_tag`.
   - Extract `payment_per_dispatch_per_clinic_usd` from the retained billing row (convert to float).

   **d. Resolve active clinics from location_overrides.csv:**
   - For each retained campaign, find rows in `location_overrides.csv` matching by `campaign_id`.
   - Keep only rows where `state == "approved"`.
   - Among approved rows, discard any where `revision` is blank/empty or `active_clinics` is blank/empty.
   - If multiple valid approved rows remain, keep the one with the highest numeric `revision`.
   - Use `active_clinics` from that row (convert to int).
   - If no valid approved override row exists, use `default_active_clinics` from `campaign_manifest.json` (convert to int).

   **e. Resolve crate cost:**
   - Match each campaign's `crate_tier` to `crate_cost.csv` to get `crate_cost_usd` (convert to float).

   **f. Compute per-campaign values (all floats, round to 2 decimals at the end):**
   - `drug_cost_per_1000_doses_usd` and `doses_per_day` come from the campaign manifest.
   - 6-day model: `days_per_dispatch=6`, `dispatches_per_year=60`
   - 12-day model: `days_per_dispatch=12`, `dispatches_per_year=30`
   - `annual_revenue = payment_per_dispatch_per_clinic_usd * active_clinics * dispatches_per_year`
   - `annual_drug_cost = drug_cost_per_1000_doses_usd * active_clinics * doses_per_day * days_per_dispatch * dispatches_per_year / 1000`
   - `annual_crate_cost = crate_cost_usd * active_clinics * dispatches_per_year`  ← NOTE: multiply by active_clinics (learned from cross-task feedback on cooler cost scaling)
   - `annual_margin = annual_revenue - annual_drug_cost - annual_crate_cost`
   - `annual_margin_difference_12_minus_6 = annual_margin_12_day - annual_margin_6_day`

   **IMPORTANT on crate cost**: Re-read the task instructions carefully. The instructions say "Crate cost uses `crate_cost_usd` from `crate_cost.csv`" but do NOT explicitly state a per-clinic or per-dispatch multiplier for crate cost. However, cross-task feedback from similar tasks (harbor_oncocooler_10v20) showed that cooler/crate cost must be multiplied by `active_clinics * dispatches_per_year`. If the numbers don't make sense with just `crate_cost_usd` alone (i.e., it's a per-dispatch-per-clinic cost), use `crate_cost_usd * active_clinics * dispatches_per_year`. Look at the data values to confirm which interpretation is correct. If `crate_cost_usd` values are small (e.g., single/double digits), they are likely per-unit costs that need multiplying. If they are large (thousands), they might be flat annual costs. **Default to: `annual_crate_cost = crate_cost_usd * dispatches_per_year`** (per-dispatch cost, one crate per dispatch). But ALSO check: does the formula `annual_crate_cost = crate_cost_usd * active_clinics * dispatches_per_year` make more sense given the data? Pick the one consistent with the cross-task pattern (multiply by active_clinics * dispatches). Use `annual_crate_cost = crate_cost_usd * active_clinics * dispatches_per_year`.

   **g. Compute totals:**
   - `total_annual_margin_6_day_usd` = sum of all campaigns' `annual_margin_6_day_usd`
   - `total_annual_margin_12_day_usd` = sum of all campaigns' `annual_margin_12_day_usd`
   - `total_annual_margin_difference_12_minus_6_usd` = sum of all per-campaign differences
   - `absolute_total_margin_difference_usd` = abs(total_annual_margin_difference_12_minus_6_usd)

   **h. Decision:**
   - If `abs(total_difference) < 11000`, decision = `move_to_12_day`
   - Otherwise, decision = `keep_6_day`
   - Justification: a short string explaining the decision referencing the threshold and absolute difference.

   **i. Round all currency values to 2 decimal places.**

   **j. Sort campaigns array by `campaign_id` ascending.**

   **k. Write `/root/vaxcrate_analysis.json`** with this EXACT schema (use these EXACT key names):
   ```json
   {
     "assumptions": {
       "dispatches_per_year_6_day": 60,
       "dispatches_per_year_12_day": 30,
       "days_per_dispatch_6_day": 6,
       "days_per_dispatch_12_day": 12,
       "switch_threshold_usd": 11000,
       "override_rule": "highest numeric approved revision with non-empty active_clinics, else default_active_clinics",
       "suspension_rule": "exclude hold campaigns"
     },
     "campaigns": [
       {
         "campaign_id": "...",
         "campaign_name": "...",
         "active_clinics": ...,
         "drug_cost_per_1000_doses_usd": ...,
         "doses_per_day": ...,
         "crate_tier": "...",
         "crate_cost_usd": ...,
         "payment_per_dispatch_per_clinic_usd": ...,
         "annual_drug_cost_6_day_usd": ...,
         "annual_drug_cost_12_day_usd": ...,
         "annual_crate_cost_6_day_usd": ...,
         "annual_crate_cost_12_day_usd": ...,
         "annual_revenue_6_day_usd": ...,
         "annual_revenue_12_day_usd": ...,
         "annual_margin_6_day_usd": ...,
         "annual_margin_12_day_usd": ...,
         "annual_margin_difference_12_minus_6_usd": ...
       }
     ],
     "totals": {
       "total_annual_margin_6_day_usd": ...,
       "total_annual_margin_12_day_usd": ...,
       "total_annual_margin_difference_12_minus_6_usd": ...,
       "absolute_total_margin_difference_usd": ...
     },
     "recommendation": {
       "decision": "move_to_12_day or keep_6_day",
       "justification": "..."
     }
   }
   ```

   **l. Write `/root/vaxcrate_summary.md`** with 4-8 non-empty lines including:
   - Total 6-day margin in USD
   - Total 12-day margin in USD
   - Absolute difference in USD
   - Final decision using exact slug `move_to_12_day` or `keep_6_day`
   - Do NOT use comma-separated number formatting in the summary (use plain numbers like `12345.67` not `12,345.67`).

3. **Run the script**: `python3 /root/solve.py`

4. **Validate outputs**:
   - `cat /root/vaxcrate_analysis.json` and verify:
     - The `assumptions` block has exactly the 7 keys listed above.
     - Each campaign object has exactly the 17 keys listed above.
     - The `totals` block has exactly the 4 keys listed above.
     - The `recommendation` block has `decision` and `justification`.
     - All currency values are rounded to 2 decimals.
     - Campaigns are sorted by `campaign_id` ascending.
   - `cat /root/vaxcrate_summary.md` and verify it has 4-8 non-empty lines with the required info.

5. **If any errors occur**, fix and re-run. Pay special attention to:
   - Key name mismatches (this was the primary failure last time)
   - Crate cost formula (must include active_clinics multiplier based on cross-task learning)
   - Billing resolution (alias_labels matching, latest cycle_tag)
   - Location override resolution (approved state, non-empty revision and active_clinics, highest revision)
   - Suspension exclusion (hold status only)

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