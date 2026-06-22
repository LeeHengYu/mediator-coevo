# Task Instruction

Execute the following steps exactly:

1. Read the three input CSV files:
   - `/root/acquisition_cost.csv`
   - `/root/packaging_cost.csv`
   - `/root/reimbursement.csv`

   Inspect their contents first (print them) so you understand the column names and how therapies are identified. The files should share a therapy name column.

2. Write and run a Python script that:

   a. Loads all three CSVs with pandas.
   b. Merges them on the therapy name column. If columns differ slightly (e.g., `therapy` vs `therapy_name`), normalize to `therapy`.
   c. For each therapy, computes:
      - `annual_drug_cost_30_day_usd = (price_per_1000_doses_usd / 1000) * 60 * 12 * 240`
      - `annual_drug_cost_90_day_usd = (price_per_1000_doses_usd / 1000) * 180 * 4 * 240`
        (Note: both equal the same total annual doses = 172800, so drug cost is identical for 30 vs 90.)
      - `annual_packaging_cost_30_day_usd = packaging_cost_usd * 240 * 12`
      - `annual_packaging_cost_90_day_usd = packaging_cost_usd * 240 * 4`
      - `annual_reimbursement_30_day_usd = reimbursement_per_fill_240_patients_usd * 12`
        (The reimbursement CSV gives reimbursement per fill for 240 patients already.)
      - `annual_reimbursement_90_day_usd = reimbursement_per_fill_240_patients_usd * 4`
      - `annual_margin_30_day_usd = annual_reimbursement_30_day_usd - annual_drug_cost_30_day_usd - annual_packaging_cost_30_day_usd`
      - `annual_margin_90_day_usd = annual_reimbursement_90_day_usd - annual_drug_cost_90_day_usd - annual_packaging_cost_90_day_usd`
      - `annual_margin_difference_90_minus_30_usd = annual_margin_90_day_usd - annual_margin_30_day_usd`
   d. Round ALL currency values to 2 decimal places.
   e. Sort therapies alphabetically by `therapy`.
   f. Compute totals:
      - `total_annual_margin_30_day_usd` = sum of all `annual_margin_30_day_usd`
      - `total_annual_margin_90_day_usd` = sum of all `annual_margin_90_day_usd`
      - `total_annual_margin_difference_90_minus_30_usd` = sum of all `annual_margin_difference_90_minus_30_usd`
      - `absolute_total_margin_difference_usd` = abs(total_annual_margin_difference_90_minus_30_usd)
      Round all totals to 2 decimals.
   g. Decision rule:
      - If `absolute_total_margin_difference_usd < 12000`, decision = `adopt_90_day`
      - Otherwise, decision = `keep_30_day`
   h. Build the JSON object with EXACTLY these keys (no extras, no missing):

```json
{
  "assumptions": {
    "patients_per_therapy": 240,
    "fills_per_year_30_day": 12,
    "fills_per_year_90_day": 4,
    "doses_per_fill_30_day": 60,
    "doses_per_fill_90_day": 180,
    "switch_threshold_usd": 12000
  },
  "therapies": [ ... ],
  "totals": {
    "total_annual_margin_30_day_usd": ...,
    "total_annual_margin_90_day_usd": ...,
    "total_annual_margin_difference_90_minus_30_usd": ...,
    "absolute_total_margin_difference_usd": ...
  },
  "recommendation": {
    "decision": "adopt_90_day or keep_30_day",
    "justification": "<a short sentence explaining the decision>"
  }
}
```

   Each therapy object in the `therapies` array must have EXACTLY these keys:
   `therapy`, `price_per_1000_doses_usd`, `canister_size_units`, `packaging_cost_usd`, `reimbursement_per_fill_240_patients_usd`, `annual_drug_cost_30_day_usd`, `annual_drug_cost_90_day_usd`, `annual_packaging_cost_30_day_usd`, `annual_packaging_cost_90_day_usd`, `annual_reimbursement_30_day_usd`, `annual_reimbursement_90_day_usd`, `annual_margin_30_day_usd`, `annual_margin_90_day_usd`, `annual_margin_difference_90_minus_30_usd`

   i. Write the JSON to `/root/cycle_margin_analysis.json` with `indent=2`.

   j. Write `/root/cycle_margin_summary.md` with 4-8 non-empty lines containing:
      - The total 30-day margin in USD
      - The total 90-day margin in USD
      - The absolute difference in USD
      - The exact decision slug (`adopt_90_day` or `keep_30_day`)

3. After the script runs, cat both output files to verify correctness:
   - Verify the JSON parses correctly and has the exact schema.
   - Verify the summary has 4-8 non-empty lines with the required values.

CRITICAL: Match the CSV column names exactly when reading. Print the CSV headers first. The reimbursement column in the CSV may be named `reimbursement_per_fill_240_patients_usd` or similar — use whatever name appears. Map it to the exact JSON key `reimbursement_per_fill_240_patients_usd` in the output. Similarly for `canister_size_units` from packaging CSV and `price_per_1000_doses_usd` from acquisition CSV. Do NOT add extra keys or omit required keys.

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
Task metadata: author_email=gpt54@example.com, author_name=GPT-5.4, category=financial-analysis, difficulty=medium, tags=[healthcare, unit-economics, csv, json, decision-analysis].
Verifier config: timeout_sec=900.0.