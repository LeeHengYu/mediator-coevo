# Task Instruction

Execute the following steps in order:

1. **Inspect all input files** to understand their structure and contents:
   - `cat /root/therapy_catalog.json`
   - `cat /root/bag_supply_cost.csv`
   - `cat /root/delivery_payment.csv`
   - `cat /root/patient_overrides.csv`

2. **Write and run a Python script** (`/root/solve.py`) that performs the full analysis. The script must:

   a. **Load inputs:**
      - Parse `therapy_catalog.json` (list or dict of therapies, each with fields like `therapy_code`, `therapy_name`, `aliases`, `include_in_review`, `drug_cost_per_1000_mg_usd`, `dose_mg_per_day`, `bag_size_ml`, etc.).
      - Parse `bag_supply_cost.csv` (columns include `bag_size_ml` and `bag_supply_cost_usd`).
      - Parse `delivery_payment.csv` (columns include `therapy_label` and `payment_per_delivery_per_patient_usd`).
      - Parse `patient_overrides.csv` (columns include `therapy_code`, `status`, `revision`, `active_patients` or `patient_count` — inspect to confirm the exact column name).

   b. **Filter in-scope therapies:** Only therapies where `include_in_review` is `true` (boolean or string — handle both).

   c. **Resolve delivery payments:** For each in-scope therapy, find the row in `delivery_payment.csv` where `therapy_label` matches either the therapy's `therapy_name` or any of its `aliases`. Ignore payment rows that don't match any in-scope therapy. The match should be case-sensitive as given in the data (but inspect data first; if labels differ only by case, do case-insensitive matching).

   d. **Resolve active patient counts from `patient_overrides.csv`:**
      - Use only rows where `status` == `approved` (case-insensitive check to be safe, but inspect data).
      - If multiple approved rows exist for the same `therapy_code`, keep the one with the highest `revision`.
      - Ignore approved rows whose `therapy_code` is not in scope.
      - The active patient count field might be named `active_patients`, `patient_count`, or similar — inspect the CSV header.

   e. **Resolve bag supply cost:** For each therapy, look up `bag_supply_cost_usd` from `bag_supply_cost.csv` by matching on `bag_size_ml`.

   f. **Compute per-therapy figures using these exact formulas:**
      - `annual_drug_cost = drug_cost_per_1000_mg_usd * active_patients * dose_mg_per_day * days_per_delivery * deliveries_per_year / 1000`
      - `annual_supply_cost = bag_supply_cost_usd * active_patients * deliveries_per_year`
        (Each delivery uses 1 bag per patient. The supply cost formula is: `bag_supply_cost_usd * active_patients * deliveries_per_year`.)
      - `annual_revenue = payment_per_delivery_per_patient_usd * active_patients * deliveries_per_year`
      - `annual_margin = annual_revenue - annual_drug_cost - annual_supply_cost`
      - Compute all of the above for both the 7-day model (days_per_delivery=7, deliveries_per_year=52) and the 14-day model (days_per_delivery=14, deliveries_per_year=26).
      - `annual_margin_difference_14_minus_7 = annual_margin_14_day - annual_margin_7_day`

   g. **Compute totals:**
      - `total_annual_margin_7_day_usd` = sum of all per-therapy `annual_margin_7_day_usd`
      - `total_annual_margin_14_day_usd` = sum of all per-therapy `annual_margin_14_day_usd`
      - `total_annual_margin_difference_14_minus_7_usd` = sum of all per-therapy `annual_margin_difference_14_minus_7_usd`
      - `absolute_total_margin_difference_usd` = `abs(total_annual_margin_difference_14_minus_7_usd)`

   h. **Decision rule:**
      - If `absolute_total_margin_difference_usd < 15000`, decision = `move_to_14_day`
      - Otherwise, decision = `keep_7_day`

   i. **Round all currency values to 2 decimal places** using Python's `round(value, 2)`.

   j. **Sort the therapies array by `therapy_code` ascending** (alphabetical/lexicographic).

   k. **Write `/root/infusion_batch_analysis.json`** with the exact schema specified:
      ```json
      {
        "assumptions": {
          "deliveries_per_year_7_day": 52,
          "deliveries_per_year_14_day": 26,
          "days_per_delivery_7_day": 7,
          "days_per_delivery_14_day": 14,
          "switch_threshold_usd": 15000,
          "patient_override_rule": "highest approved revision per therapy_code"
        },
        "therapies": [ ... ],
        "totals": { ... },
        "recommendation": {
          "decision": "move_to_14_day" or "keep_7_day",
          "justification": "<a brief sentence explaining the decision referencing the absolute difference and threshold>"
        }
      }
      ```
      Use `json.dump` with `indent=2` for readability.

   l. **Write `/root/infusion_batch_summary.md`** with 4–8 non-empty lines that include:
      - Total 7-day margin (USD) with the numeric value
      - Total 14-day margin (USD) with the numeric value
      - Absolute difference (USD) with the numeric value
      - The exact decision slug (`move_to_14_day` or `keep_7_day`)
      Example format:
      ```
      # Infusion Batch Analysis Summary

      - Total 7-day annual margin: $X.XX USD
      - Total 14-day annual margin: $X.XX USD
      - Absolute margin difference: $X.XX USD
      - Recommendation: move_to_14_day
      ```

3. **Run the script:** `python3 /root/solve.py`

4. **Validate outputs:**
   - `cat /root/infusion_batch_analysis.json` — confirm it parses as valid JSON, has the correct schema with `assumptions`, `therapies` (sorted by therapy_code), `totals`, and `recommendation`.
   - `cat /root/infusion_batch_summary.md` — confirm 4–8 non-empty lines, contains total margins, absolute difference, and the decision slug.
   - Verify that the `therapies` array only contains in-scope therapies.
   - Spot-check one therapy's numbers manually: pick the first therapy, compute its 7-day and 14-day drug cost, supply cost, revenue, and margin by hand from the raw data to confirm the script is correct.

5. **If any validation fails**, fix the script and re-run until both output files are correct.

Key pitfalls to avoid:
- Do NOT include therapies where `include_in_review` is false.
- Do NOT include patient override rows that are not `approved` or whose therapy is not in scope.
- The `annual_supply_cost` formula is `bag_supply_cost_usd * active_patients * deliveries_per_year` — do not accidentally include `days_per_delivery` in this formula (supply cost is per delivery, not per day).
- Drug cost formula DOES include `days_per_delivery` — the drug is consumed daily.
- Make sure `therapy_label` matching considers aliases from the catalog.
- Make sure numeric types are correct when reading CSVs (convert strings to float/int as needed).
- The `absolute_total_margin_difference_usd` field must be the absolute value of the total difference, not the sum of absolute per-therapy differences.

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