# Task Instruction

Execute the following steps in order:

1. **Inspect all input files** to understand their structure:
 ```
 cat /root/therapy_catalog.json
 cat /root/bag_supply_cost.csv
 cat /root/delivery_payment.csv
 cat /root/patient_overrides.csv
 ```

2. **Write a Python script** `/root/solve.py` that implements the full analysis. The script must:

 **a. Load data:**
 - Parse `/root/therapy_catalog.json` (list of therapy objects).
 - Parse `/root/bag_supply_cost.csv` (columns include `bag_size_ml` and `bag_supply_cost_usd`).
 - Parse `/root/delivery_payment.csv` (columns include `therapy_label` and `payment_per_delivery_per_patient_usd`).
 - Parse `/root/patient_overrides.csv` (columns include `therapy_code`, `status`, `revision`, `active_patients`).

 **b. Filter in-scope therapies:**
 - From `therapy_catalog.json`, keep only entries where `include_in_review` is `true`.
 - Build a lookup: for each in-scope therapy, map its `therapy_name` AND every alias in its `aliases` list (if present) to the therapy record.

 **c. Resolve delivery payments:**
 - For each row in `delivery_payment.csv`, match `therapy_label` against the alias/name lookup built above.
 - Ignore rows that don't map to any in-scope therapy.
 - Store `payment_per_delivery_per_patient_usd` keyed by `therapy_code`.

 **d. Resolve patient overrides:**
 - Filter `patient_overrides.csv` to rows where `status` == `approved`.
 - Among approved rows, keep only those whose `therapy_code` matches an in-scope therapy.
 - If multiple approved rows exist for the same `therapy_code`, keep only the one with the highest `revision`.
 - The `active_patients` value from the kept row is used.

 **e. Resolve bag supply cost:**
 - For each in-scope therapy, look up `bag_supply_cost_usd` from `bag_supply_cost.csv` by matching `bag_size_ml`.

 **f. Compute per-therapy financials** (for both 7-day and 14-day models):
 - Constants: 7-day → 52 deliveries/year, 7 days/delivery; 14-day → 26 deliveries/year, 14 days/delivery.
 - `annual_drug_cost = drug_cost_per_1000_mg_usd * active_patients * dose_mg_per_day * days_per_delivery * deliveries_per_year / 1000`
 - `annual_supply_cost = bag_supply_cost_usd * active_patients * deliveries_per_year`
 - `annual_revenue = payment_per_delivery_per_patient_usd * active_patients * deliveries_per_year`
 - `annual_margin = annual_revenue - annual_drug_cost - annual_supply_cost`
 - `annual_margin_difference_14_minus_7 = annual_margin_14_day - annual_margin_7_day`
 - Round ALL currency values to 2 decimal places.

 **g. Compute totals:**
 - Sum all per-therapy `annual_margin_7_day_usd` → `total_annual_margin_7_day_usd`
 - Sum all per-therapy `annual_margin_14_day_usd` → `total_annual_margin_14_day_usd`
 - `total_annual_margin_difference_14_minus_7_usd = total_14 - total_7` (round to 2 decimals)
 - `absolute_total_margin_difference_usd = abs(total_annual_margin_difference_14_minus_7_usd)` (round to 2 decimals)

 **h. Decision rule:**
 - If `absolute_total_margin_difference_usd < 15000`, decision = `move_to_14_day`.
 - Otherwise, decision = `keep_7_day`.
 - Write a justification string that mentions the absolute difference and the threshold.

 **i. Build JSON output** matching the exact schema from the task. Sort the `therapies` array by `therapy_code` ascending. Write to `/root/infusion_batch_analysis.json` with `indent=2`.

 **j. Build markdown summary** `/root/infusion_batch_summary.md`:
 - 4–8 non-empty lines.
 - Must include: total 7-day margin (USD with commas as thousands separators), total 14-day margin (USD), absolute difference (USD), and the exact decision slug (`move_to_14_day` or `keep_7_day`).
 - Example format:
   ```
   # Infusion Batch Analysis Summary\n\nTotal 7-Day Annual Margin: $X,XXX.XX\nTotal 14-Day Annual Margin: $X,XXX.XX\nAbsolute Margin Difference: $X,XXX.XX\nRecommendation: move_to_14_day
   ```

3. **Run the script:**
 ```
 python3 /root/solve.py
 ```

4. **Validate outputs:**
 - `cat /root/infusion_batch_analysis.json` — confirm it parses as valid JSON, has the `assumptions`, `therapies`, `totals`, and `recommendation` top-level keys, therapies are sorted by `therapy_code`, all currency values have at most 2 decimal places.
 - `cat /root/infusion_batch_summary.md` — confirm 4–8 non-empty lines, includes the required figures and decision slug.
 - Verify the `recommendation` object has both `decision` and `justification` keys (avoid the schema error seen in the vaxcrate failure where these were placed at the wrong level).

5. If any test or verifier script exists (e.g., `test_output.py`), run it:
 ```
 ls /root/test_output.py 2>/dev/null && python3 -m pytest /root/test_output.py -v
 ```

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
Task metadata: author_email=gpt54@example.com, author_name=GPT-5.4, category=financial-analysis, difficulty=medium, tags=[home-infusion, json, csv, alias-resolution, decision-analysis].
Verifier config: timeout_sec=900.0.