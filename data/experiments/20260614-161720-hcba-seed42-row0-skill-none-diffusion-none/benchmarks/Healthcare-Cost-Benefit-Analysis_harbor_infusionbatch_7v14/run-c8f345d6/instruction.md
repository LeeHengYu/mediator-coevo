# Task Instruction

## Task: Healthcare Cost-Benefit Analysis – Infusion Batch 7-day vs 14-day

You must produce two output files:
1. `/root/infusion_batch_analysis.json`
2. `/root/infusion_batch_summary.md`

### Step-by-step instructions

#### Step 1: Inspect all input files
Read and display the contents of:
- `/root/therapy_catalog.json`
- `/root/bag_supply_cost.csv`
- `/root/delivery_payment.csv`
- `/root/patient_overrides.csv`

Understand the structure before writing any code.

#### Step 2: Write a Python script `/root/solve.py` that does the following:

**A. Load data:**
- Load `therapy_catalog.json` as a list/dict of therapy records.
- Load `bag_supply_cost.csv`, `delivery_payment.csv`, `patient_overrides.csv` as CSV.

**B. Filter in-scope therapies:**
- Keep only therapies where `include_in_review` is `true` (boolean True).

**C. Resolve delivery payments:**
- For each row in `delivery_payment.csv`, match `therapy_label` to either the `therapy_name` or any entry in the `aliases` list of an in-scope therapy from `therapy_catalog.json`.
- Ignore payment rows that don't map to any in-scope therapy.
- Use the matched `payment_per_delivery_per_patient_usd` value.

**D. Resolve active patient counts:**
- From `patient_overrides.csv`, keep only rows where `status` is `approved`.
- If multiple approved rows exist for the same `therapy_code`, keep only the one with the highest `revision`.
- Ignore approved rows for therapy codes not in scope.
- The resulting `active_patients` count comes from the kept row's `active_patients` (or `patient_count` — check the actual column name).

**E. Compute per-therapy metrics (for each in-scope therapy):**

Constants:
- 7-day model: `days_per_delivery=7`, `deliveries_per_year=52`
- 14-day model: `days_per_delivery=14`, `deliveries_per_year=26`

From therapy catalog: `drug_cost_per_1000_mg_usd`, `dose_mg_per_day`, `bag_size_ml`
From bag_supply_cost.csv: match `bag_size_ml` to get `bag_supply_cost_usd`
From delivery_payment.csv (resolved): `payment_per_delivery_per_patient_usd`
From patient_overrides.csv (resolved): `active_patients`

For each model (7-day and 14-day):
- `annual_drug_cost = drug_cost_per_1000_mg_usd * active_patients * dose_mg_per_day * days_per_delivery * deliveries_per_year / 1000`
- `annual_supply_cost = bag_supply_cost_usd * active_patients * deliveries_per_year`
- `annual_revenue = payment_per_delivery_per_patient_usd * active_patients * deliveries_per_year`
- `annual_margin = annual_revenue - annual_drug_cost - annual_supply_cost`
- `annual_margin_difference_14_minus_7_usd = annual_margin_14_day - annual_margin_7_day`

Round ALL currency values to 2 decimal places.

**F. Compute totals:**
- `total_annual_margin_7_day_usd` = sum of all per-therapy `annual_margin_7_day_usd`
- `total_annual_margin_14_day_usd` = sum of all per-therapy `annual_margin_14_day_usd`
- `total_annual_margin_difference_14_minus_7_usd` = sum of all per-therapy `annual_margin_difference_14_minus_7_usd`
- `absolute_total_margin_difference_usd` = abs(total_annual_margin_difference_14_minus_7_usd)

Round all to 2 decimals.

**G. Decision:**
- If `absolute_total_margin_difference_usd < 15000`, decision = `move_to_14_day`
- Otherwise, decision = `keep_7_day`
- Write a short justification string.

**H. Build JSON output with EXACT key names:**

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
  "therapies": [
    {
      "therapy_code": "...",
      "therapy_name": "...",
      "active_patients": ...,
      "drug_cost_per_1000_mg_usd": ...,
      "dose_mg_per_day": ...,
      "bag_size_ml": ...,
      "bag_supply_cost_usd": ...,
      "payment_per_delivery_per_patient_usd": ...,
      "annual_drug_cost_7_day_usd": ...,
      "annual_drug_cost_14_day_usd": ...,
      "annual_supply_cost_7_day_usd": ...,
      "annual_supply_cost_14_day_usd": ...,
      "annual_revenue_7_day_usd": ...,
      "annual_revenue_14_day_usd": ...,
      "annual_margin_7_day_usd": ...,
      "annual_margin_14_day_usd": ...,
      "annual_margin_difference_14_minus_7_usd": ...
    }
  ],
  "totals": {
    "total_annual_margin_7_day_usd": ...,
    "total_annual_margin_14_day_usd": ...,
    "total_annual_margin_difference_14_minus_7_usd": ...,
    "absolute_total_margin_difference_usd": ...
  },
  "recommendation": {
    "decision": "move_to_14_day or keep_7_day",
    "justification": "..."
  }
}
```

**CRITICAL**: Every key name must match EXACTLY as shown above. Do NOT use nested structures for assumptions. Do NOT omit the `_usd` suffix. Use `switch_threshold_usd` (not `decision_threshold_usd`). Sort the `therapies` array by `therapy_code` ascending (alphabetical).

**I. Build markdown summary `/root/infusion_batch_summary.md`:**
- 4-8 non-empty lines
- Must include: total 7-day margin (USD), total 14-day margin (USD), absolute difference (USD), and the exact decision slug (`move_to_14_day` or `keep_7_day`)
- Format currency without commas in the number to avoid formatting issues, OR use standard formatting — just ensure the values are present.

#### Step 3: Run the script
```bash
python3 /root/solve.py
```

#### Step 4: Validate outputs
- Read `/root/infusion_batch_analysis.json` and verify:
  - Top-level keys are exactly: `assumptions`, `therapies`, `totals`, `recommendation`
  - `assumptions` has exactly the 6 keys listed above
  - Each therapy object has exactly the 17 keys listed above
  - `totals` has exactly the 4 keys listed above
  - `recommendation` has `decision` and `justification`
  - All currency values are rounded to 2 decimal places
  - Therapies are sorted by `therapy_code` ascending
- Read `/root/infusion_batch_summary.md` and verify it has 4-8 non-empty lines with the required values and slug.

If any validation fails, fix and re-run.

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