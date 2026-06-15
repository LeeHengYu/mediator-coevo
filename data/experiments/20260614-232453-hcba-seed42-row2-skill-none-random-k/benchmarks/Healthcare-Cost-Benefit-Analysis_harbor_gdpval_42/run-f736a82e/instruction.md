# Task Instruction

Create a Python script `/root/solve.py` that performs the following steps, then execute it.

## Step 1: Read Input Files

Read the three CSV files:
- `/root/wholesale_price.csv` — contains medication names and `price_per_1000_tablets_usd`
- `/root/vial_price.csv` — contains medication names, `vial_size_drams`, and `vial_price_usd`
- `/root/reimbursement.csv` — contains medication names and `reimbursement_per_fill_300_patients_usd`

First, inspect each CSV file (print the first few lines and column names) to understand the exact column names and medication name formats. Then merge them on the medication name column.

## Step 2: Perform Calculations

For each of the 10 medications, compute:

```
annual_drug_cost_90_day = (price_per_1000_tablets_usd / 1000) * 90 * 4 * 300
annual_drug_cost_100_day = (price_per_1000_tablets_usd / 1000) * 100 * 3 * 300

annual_supply_cost_90_day = vial_price_usd * 300 * 4
annual_supply_cost_100_day = vial_price_usd * 300 * 3

annual_reimbursement_90_day = reimbursement_per_fill_300_patients_usd * 4
annual_reimbursement_100_day = reimbursement_per_fill_300_patients_usd * 3

annual_revenue_90_day = annual_reimbursement_90_day - annual_drug_cost_90_day - annual_supply_cost_90_day
annual_revenue_100_day = annual_reimbursement_100_day - annual_drug_cost_100_day - annual_supply_cost_100_day

annual_revenue_difference_100_minus_90 = annual_revenue_100_day - annual_revenue_90_day
```

All currency values must be rounded to 2 decimal places.

## Step 3: Compute Totals and Decision

```
total_annual_revenue_90_day = sum of all annual_revenue_90_day values
total_annual_revenue_100_day = sum of all annual_revenue_100_day values
total_annual_revenue_difference = sum of all per-medication differences
absolute_total_revenue_difference = abs(total_annual_revenue_difference)
```

Decision rule:
- If `absolute_total_revenue_difference < 16000`: decision = `"switch_to_100_day"`
- Otherwise: decision = `"keep_90_day"`

## Step 4: Write `/root/refill_analysis.json`

The JSON must have EXACTLY this top-level structure with these exact keys:

```json
{
  "assumptions": {
    "patients_per_medication": 300,
    "fills_per_year_90_day": 4,
    "fills_per_year_100_day": 3,
    "tablets_per_fill_90_day": 90,
    "tablets_per_fill_100_day": 100,
    "switch_threshold_usd": 16000
  },
  "medications": [
    {
      "medication": "<name>",
      "price_per_1000_tablets_usd": <float>,
      "vial_size_drams": <int>,
      "vial_price_usd": <float>,
      "reimbursement_per_fill_300_patients_usd": <float>,
      "annual_drug_cost_90_day_usd": <float>,
      "annual_drug_cost_100_day_usd": <float>,
      "annual_supply_cost_90_day_usd": <float>,
      "annual_supply_cost_100_day_usd": <float>,
      "annual_reimbursement_90_day_usd": <float>,
      "annual_reimbursement_100_day_usd": <float>,
      "annual_revenue_90_day_usd": <float>,
      "annual_revenue_100_day_usd": <float>,
      "annual_revenue_difference_100_minus_90_usd": <float>
    }
  ],
  "totals": {
    "total_annual_revenue_90_day_usd": <float>,
    "total_annual_revenue_100_day_usd": <float>,
    "total_annual_revenue_difference_100_minus_90_usd": <float>,
    "absolute_total_revenue_difference_usd": <float>
  },
  "recommendation": {
    "decision": "switch_to_100_day" or "keep_90_day",
    "justification": "<1-2 sentence explanation referencing the absolute difference and threshold>"
  }
}
```

CRITICAL: Every medication object MUST include ALL 14 fields listed above, especially `annual_revenue_difference_100_minus_90_usd`. The `totals` and `recommendation` keys MUST be at the top level of the JSON, NOT nested inside any other key. Do NOT add extra keys to `assumptions`.

## Step 5: Write `/root/refill_summary.md`

Write a markdown file with 4-8 lines that includes:
- The total 90-day revenue as a plain number (no commas in the number — use format like `12345.67` not `12,345.67`)
- The total 100-day revenue
- The absolute difference
- The exact decision slug: either `switch_to_100_day` or `keep_90_day`

Example format:
```
## Refill Policy Analysis Summary

Total annual revenue under 90-day fills: $XXXXX.XX
Total annual revenue under 100-day fills: $XXXXX.XX
Absolute revenue difference: $XXXXX.XX
Recommendation: switch_to_100_day
```

Do NOT use commas as thousands separators in the summary numbers.

## Step 6: Validate

After writing both files:
1. Re-read `/root/refill_analysis.json` and verify it parses as valid JSON.
2. Verify the top-level keys are exactly: `assumptions`, `medications`, `totals`, `recommendation`.
3. Verify each medication entry has the `annual_revenue_difference_100_minus_90_usd` field.
4. Verify `totals` contains all 4 required fields.
5. Verify `recommendation` contains `decision` with one of the two valid slugs.
6. Print the JSON and summary contents for inspection.

If any CSV column names differ from expected, adapt accordingly (print columns first to check). The medication names in the output JSON should match exactly what appears in the CSV files.

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

Task-local resources are available under `environment/skills`: business-model-math-validation, loyalty-modeling, pharmacy-supply-chain, recursive-generosity-protocol, value-analysis.

# Verifier Contract

Success is judged by the SkillFlow verifier for this task.
Do not bypass, remove, or weaken verifier scripts, tests, fixtures, or expected-output checks.
Run the provided tests or verifier command when practical before finalizing.
Task metadata: author_email=codex@example.com, author_name=Codex, category=financial-analysis, difficulty=medium, tags=[pharmacy, unit-economics, cost-analysis, json, verification].
Verifier config: timeout_sec=900.0.