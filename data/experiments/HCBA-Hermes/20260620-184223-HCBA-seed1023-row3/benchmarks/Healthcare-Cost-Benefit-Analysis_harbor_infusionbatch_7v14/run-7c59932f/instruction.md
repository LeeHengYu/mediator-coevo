# Task Instruction

Execute the following steps in order:

1. **Read all input files** to understand the data:
   - `cat /root/therapy_catalog.json`
   - `cat /root/bag_supply_cost.csv`
   - `cat /root/delivery_payment.csv`
   - `cat /root/patient_overrides.csv`

2. **Read the test/verifier file** (likely at `/root/tests/test_outputs.py` or similar) to understand exact validation expectations:
   - `find /root -name '*.py' -path '*/test*' | head -20`
   - Read whatever test files you find.

3. **Write a Python script** `/root/solve.py` that does the following:

   a. Load `therapy_catalog.json`. Filter to therapies where `include_in_review` is `true`. Build a lookup from therapy_name and all aliases to the therapy record.

   b. Load `delivery_payment.csv`. For each row, match `therapy_label` to either `therapy_name` or any alias in the catalog. Ignore rows that don't match an in-scope therapy. Store the `payment_per_delivery_per_patient_usd` for each matched therapy.

   c. Load `patient_overrides.csv`. Keep only rows where `status` is `approved`. For rows sharing the same `therapy_code`, keep only the one with the highest `revision`. Ignore rows for therapy_codes not in scope.

   d. Load `bag_supply_cost.csv`. Build a lookup from `bag_size_ml` to `bag_supply_cost_usd`.

   e. For each in-scope therapy, compute:
      - **annual_drug_cost** = `drug_cost_per_1000_mg_usd * active_patients * dose_mg_per_day * days_per_delivery * deliveries_per_year / 1000`
      - **annual_supply_cost** = `bag_supply_cost_usd * active_patients * deliveries_per_year`
      - **annual_revenue** = `payment_per_delivery_per_patient_usd * active_patients * deliveries_per_year`
      - **annual_margin** = `annual_revenue - annual_drug_cost - annual_supply_cost`
      - Compute all of these for both 7-day (days_per_delivery=7, deliveries=52) and 14-day (days_per_delivery=14, deliveries=26) models.
      - `annual_margin_difference_14_minus_7_usd` = margin_14 - margin_7
      - Round ALL currency values to 2 decimal places.

   f. Sort therapies by `therapy_code` ascending.

   g. Compute totals:
      - `total_annual_margin_7_day_usd` = sum of all therapy 7-day margins
      - `total_annual_margin_14_day_usd` = sum of all therapy 14-day margins
      - `total_annual_margin_difference_14_minus_7_usd` = sum of per-therapy differences
      - `absolute_total_margin_difference_usd` = abs(total_difference)
      - Round all to 2 decimals.

   h. Decision: if `abs(total_difference) < 15000`, decision = `move_to_14_day`; else `keep_7_day`.

   i. Build the JSON output with **exactly** these root keys and no others: `assumptions`, `therapies`, `totals`, `recommendation`. Do NOT include `warnings`, `model_constants`, `service_line`, or any extra keys at any level. Each therapy object must have exactly the keys from the schema.

   j. Write `/root/infusion_batch_analysis.json` with `json.dump(..., indent=2)`.

   k. Write `/root/infusion_batch_summary.md` with 4-8 non-empty lines. **Critical**: format all currency values using Python's `f'{value:,.2f}'` format (with comma thousands separators). Include:
      - Total 7-day margin (USD) with the exact number
      - Total 14-day margin (USD) with the exact number
      - Absolute difference (USD) with the exact number
      - The decision slug exactly: `move_to_14_day` or `keep_7_day`

4. **Run the script**: `cd /root && python solve.py`

5. **Validate**: 
   - `cat /root/infusion_batch_analysis.json` and verify no extra keys exist.
   - `cat /root/infusion_batch_summary.md` and verify comma-formatted currency values.
   - Run the test suite if available: `cd /root && python -m pytest tests/ -v` or similar.

6. **Fix any issues** found by the tests and re-run until all tests pass.

Key pitfalls to avoid (from prior feedback):
- Do NOT add extra keys like `warnings`, `model_constants`, or `service_line` anywhere in the JSON.
- Do NOT output plain floats in the markdown summary; always use `:,.2f` formatting with comma separators.
- Match therapy_label in delivery_payment.csv against BOTH therapy_name and aliases from the catalog.
- Ensure the `assumptions` block has exactly the keys shown in the schema.

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