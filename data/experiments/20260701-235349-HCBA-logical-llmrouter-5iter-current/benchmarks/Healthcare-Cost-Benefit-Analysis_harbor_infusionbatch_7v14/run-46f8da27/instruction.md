# Task Instruction

Execute the following steps in order:

1. **Read all input files** before writing any code:
   - `cat /root/therapy_catalog.json`
   - `cat /root/bag_supply_cost.csv`
   - `cat /root/delivery_payment.csv`
   - `cat /root/patient_overrides.csv`

2. **Write and run a Python script** `/root/solve.py` that does the following:

   a. **Load inputs:**
      - Parse `therapy_catalog.json` (list or dict of therapy entries).
      - Parse `bag_supply_cost.csv`, `delivery_payment.csv`, `patient_overrides.csv` with the `csv` module.

   b. **Filter therapies:** Keep only entries where `include_in_review` is `true` (boolean True or string "true").

   c. **Resolve delivery payments:** For each row in `delivery_payment.csv`, match its `therapy_label` against each in-scope therapy's `therapy_name` OR any alias in the therapy's aliases list (check for keys like `aliases`, `alias`, etc. in the catalog). Ignore payment rows that don't match any in-scope therapy. Store `payment_per_delivery_per_patient_usd` (as float) keyed by `therapy_code`.

   d. **Resolve active patients from `patient_overrides.csv`:**
      - Keep only rows where `status` == `approved`.
      - Among approved rows sharing the same `therapy_code`, keep only the one with the highest `revision` (numeric comparison).
      - Ignore rows whose `therapy_code` is not in scope.
      - The kept row's `active_patients` (or `patient_count` — inspect the CSV header) gives the count.

   e. **Compute per-therapy metrics** for each in-scope therapy:
      - `drug_cost_per_1000_mg_usd` and `dose_mg_per_day` from the catalog.
      - `bag_size_ml` from the catalog; look up `bag_supply_cost_usd` from `bag_supply_cost.csv` by matching `bag_size_ml`.
      - For model in {7-day, 14-day}:
        - `days_per_delivery` = 7 or 14
        - `deliveries_per_year` = 52 or 26
        - `annual_drug_cost = drug_cost_per_1000_mg_usd * active_patients * dose_mg_per_day * days_per_delivery * deliveries_per_year / 1000`
        - `annual_supply_cost = bag_supply_cost_usd * active_patients * deliveries_per_year`
        - `annual_revenue = payment_per_delivery_per_patient_usd * active_patients * deliveries_per_year`
        - `annual_margin = annual_revenue - annual_drug_cost - annual_supply_cost`
      - `annual_margin_difference_14_minus_7_usd = annual_margin_14_day - annual_margin_7_day`
      - Round ALL currency values to 2 decimal places.

   f. **Sort** the therapies list by `therapy_code` ascending (string sort).

   g. **Compute totals:**
      - `total_annual_margin_7_day_usd` = sum of all per-therapy 7-day margins (round to 2 dp).
      - `total_annual_margin_14_day_usd` = sum of all per-therapy 14-day margins (round to 2 dp).
      - `total_annual_margin_difference_14_minus_7_usd` = sum of all per-therapy differences (round to 2 dp).
      - `absolute_total_margin_difference_usd` = `abs(total_annual_margin_difference_14_minus_7_usd)` (round to 2 dp).

   h. **Decision:**
      - If `absolute_total_margin_difference_usd < 15000`, decision = `"move_to_14_day"`.
      - Otherwise, decision = `"keep_7_day"`.
      - Justification: a brief string explaining the numbers.

   i. **Write `/root/infusion_batch_analysis.json`** with EXACTLY this structure (use `json.dump` with `indent=2`):
      ```
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
          "decision": "...",
          "justification": "..."
        }
      }
      ```
      Make sure `assumptions` includes ALL six keys exactly as shown, especially `switch_threshold_usd` and `patient_override_rule`.

   j. **Write `/root/infusion_batch_summary.md`** with 4-8 non-empty lines. Currency values MUST use comma thousands separators and 2 decimal places. Use Python's `"{:,.2f}".format(value)` for formatting. Example line: `Total 7-day margin: $-455,619.31`. Include:
      - Total 7-day margin (USD)
      - Total 14-day margin (USD)
      - Absolute difference (USD)
      - Final decision using the exact slug `move_to_14_day` or `keep_7_day`.

3. **Run the script:** `python3 /root/solve.py`

4. **Validate outputs:**
   - `cat /root/infusion_batch_analysis.json` — confirm the `assumptions` block has all 6 keys, `therapies` is sorted by `therapy_code`, all currency fields have exactly 2 decimal places, and `recommendation` block exists with `decision` and `justification`.
   - `cat /root/infusion_batch_summary.md` — confirm 4-8 non-empty lines, comma-formatted currency values, and the exact decision slug.

5. **If a test file exists** (e.g., `/root/test_outputs.py` or similar), run it: `python3 -m pytest /root/test_outputs.py -v` and fix any failures.

Key pitfalls to avoid (from prior feedback):
- Do NOT omit `switch_threshold_usd` or `patient_override_rule` from the `assumptions` block.
- Do NOT output raw floats like `-455619.31` in the summary markdown — use comma formatting like `-455,619.31`.
- Make sure the decision rule threshold comparison uses `abs(total_difference) < 15000` (strict less-than).

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