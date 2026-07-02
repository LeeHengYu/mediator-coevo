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

2. **Read the test/verifier file** to understand exact validation expectations:
 ```
 cat /root/test_output.py
 ```

3. **Write and run a Python script** at `/root/solve.py` that performs the full analysis. The script must:

 a. **Load all inputs:**
 - `campaign_manifest.json` — a JSON file with campaign entries.
 - `crate_cost.csv` — maps `crate_tier` to `crate_cost_usd`.
 - `billing.csv` — has `campaign_label`, `status`, `cycle_tag`, `payment_per_dispatch_per_clinic_usd`.
 - `location_overrides.csv` — has `campaign_id`, `state`, `revision`, `active_clinics`.
 - `suspensions.csv` — has `campaign_id`, `suspension_status`.

 b. **Filter campaigns:**
 - From the manifest, keep only campaigns where `analysis_flag` == `"review"`.
 - Exclude any campaign whose `campaign_id` appears in `suspensions.csv` with `suspension_status` == `"hold"`.

 c. **Resolve billing rows:**
 - For each retained campaign, find billing rows where `campaign_label` matches either `campaign_name` or any entry in `alias_labels` (which may be a list or comma-separated string — inspect the data to determine).
 - Keep only rows where `status` == `"active"`.
 - If multiple active rows match the same campaign, keep the one with the latest (lexicographically largest) `cycle_tag`.
 - Extract `payment_per_dispatch_per_clinic_usd` from the retained billing row.

 d. **Resolve active clinics:**
 - From `location_overrides.csv`, filter to rows where `state` == `"approved"`.
 - Discard rows where `revision` is blank/empty or `active_clinics` is blank/empty.
 - Among remaining valid rows for a given `campaign_id`, keep the one with the highest numeric `revision`.
 - If no valid override row exists for a campaign, use `default_active_clinics` from the manifest.

 e. **Compute per-campaign metrics** using these formulas:
 - `annual_revenue = payment_per_dispatch_per_clinic_usd * active_clinics * dispatches_per_year`
 - `annual_drug_cost = drug_cost_per_1000_doses_usd * active_clinics * doses_per_day * days_per_dispatch * dispatches_per_year / 1000`
 - `annual_crate_cost = crate_cost_usd * dispatches_per_year` (crate_cost_usd from crate_cost.csv matched by crate_tier)
 - `annual_margin = annual_revenue - annual_drug_cost - annual_crate_cost`
 - `annual_margin_difference = margin_12_day - margin_6_day`
 - 6-day model: 6 days/dispatch, 60 dispatches/year
 - 12-day model: 12 days/dispatch, 30 dispatches/year

 f. **Compute totals:**
 - Sum all per-campaign margins for 6-day and 12-day.
 - `total_difference = total_12_day_margin - total_6_day_margin`
 - `absolute_total_margin_difference = abs(total_difference)`

 g. **Decision rule:**
 - If `abs(total_difference) < 11000`, recommend `"move_to_12_day"`.
 - Otherwise, recommend `"keep_6_day"`.

 h. **Round all currency values to 2 decimal places** using Python's `round(value, 2)`.

 i. **Sort the campaigns array by `campaign_id` ascending.**

 j. **Write `/root/vaxcrate_analysis.json`** with the exact schema from the task. The `assumptions` object must contain exactly these keys:
 - `dispatches_per_year_6_day`: 60
 - `dispatches_per_year_12_day`: 30
 - `days_per_dispatch_6_day`: 6
 - `days_per_dispatch_12_day`: 12
 - `switch_threshold_usd`: 11000
 - `override_rule`: `"highest numeric approved revision with non-empty active_clinics, else default_active_clinics"`
 - `suspension_rule`: `"exclude hold campaigns"`

   The `recommendation` object must have both `decision` (the slug) and `justification` (a brief explanatory string).

 k. **Write `/root/vaxcrate_summary.md`** with 4–8 non-empty lines including:
 - Total 6-day margin in USD with comma formatting (e.g., `1,234.56`)
 - Total 12-day margin in USD with comma formatting
 - Absolute difference in USD with comma formatting
 - The exact decision slug (`move_to_12_day` or `keep_6_day`)

   **Important:** Format currency numbers with commas as thousands separators (use `f"{value:,.2f}"` in Python). This was a failure mode in a similar task.

4. **Run the script:**
 ```
 cd /root && python solve.py
 ```

5. **Validate the outputs:**
 ```
 cat /root/vaxcrate_analysis.json
 cat /root/vaxcrate_summary.md
 python -c "import json; d=json.load(open('/root/vaxcrate_analysis.json')); print('Keys in assumptions:', list(d['assumptions'].keys())); print('Num campaigns:', len(d['campaigns'])); print('Recommendation:', d['recommendation']); print('Totals:', d['totals'])"
 ```

6. **Run the verifier test if available:**
 ```
 cd /root && python -m pytest test_output.py -v 2>&1 | head -80
 ```
   If any tests fail, read the error messages carefully, fix the script, re-run, and re-validate. Pay special attention to:
   - Missing or extra keys in the JSON schema
   - Incorrect numeric formatting in the summary (must use comma-separated thousands)
   - Sort order of campaigns
   - Rounding precision
   - The `justification` field must exist in `recommendation`

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