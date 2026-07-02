# Task Instruction

Execute the following steps in order:

1. **Inspect input files**
```bash
cat /root/ingredient_cost.csv
cat /root/card_cost.csv
cat /root/reimbursement.csv
```

2. **Inspect the test/verifier file** to understand exactly what keys, values, and structure it checks:
```bash
cat /tests/test_outputs.py
```
Also check for any other test files:
```bash
find /tests -type f 2>/dev/null
ls /root/tests/ 2>/dev/null
```

3. **Create `/root/syncpack_analysis.json`** using a Python script. The script must:

   a. Read the three CSV files.
   b. For each medication, compute:
      - `annual_drug_cost_28_day_usd = (price_per_1000_capsules_usd / 1000) * 56 * 12 * 180`
      - `annual_drug_cost_56_day_usd = (price_per_1000_capsules_usd / 1000) * 112 * 6 * 180`
      - `annual_packaging_cost_28_day_usd = card_cost_usd * 180 * 12`
      - `annual_packaging_cost_56_day_usd = card_cost_usd * 180 * 6`  (use the card_cost for the 56-day blister_card_count)
      - `annual_reimbursement_28_day_usd = reimbursement_per_cycle_180_patients * 12`
      - `annual_reimbursement_56_day_usd = reimbursement_per_cycle_180_patients * 6`
      - margins and difference per the formulas in the prompt
   c. **IMPORTANT for packaging cost**: The `card_cost.csv` likely has rows for different `blister_card_count` values (e.g., 28 and 56). Each medication's `blister_card_count` from `ingredient_cost.csv` (or however it's specified) determines which card cost row to use. For the 28-day model, use the card cost for `blister_card_count=28`; for the 56-day model, use the card cost for `blister_card_count=56`. Read the CSVs carefully to understand the join logic. If `blister_card_count` is per-medication in `ingredient_cost.csv`, that's the 28-day card; the 56-day card count would be double. If `card_cost.csv` has a single row per card count, look up accordingly.
   d. Sort medications alphabetically by `medication` name.
   e. Round all currency values to 2 decimal places.
   f. The `assumptions` block must use **exactly** these keys and values:
      ```
      "assumptions": {
        "patients_per_medication": 180,
        "fills_per_year_28_day": 12,
        "fills_per_year_56_day": 6,
        "capsules_per_fill_28_day": 56,
        "capsules_per_fill_56_day": 112,
        "switch_threshold_usd": 9000
      }
      ```
   g. Each medication object must have **exactly** these keys (in any order): `medication`, `price_per_1000_capsules_usd`, `blister_card_count`, `card_cost_usd`, `reimbursement_per_cycle_180_patients_usd`, `annual_drug_cost_28_day_usd`, `annual_drug_cost_56_day_usd`, `annual_packaging_cost_28_day_usd`, `annual_packaging_cost_56_day_usd`, `annual_reimbursement_28_day_usd`, `annual_reimbursement_56_day_usd`, `annual_margin_28_day_usd`, `annual_margin_56_day_usd`, `annual_margin_difference_56_minus_28_usd`.
   h. The `totals` block must have exactly: `total_annual_margin_28_day_usd`, `total_annual_margin_56_day_usd`, `total_annual_margin_difference_56_minus_28_usd`, `absolute_total_margin_difference_usd`.
   i. Decision rule: if `abs(total_difference) < 9000` → `"convert_to_56_day"`, else `"keep_28_day"`.
   j. Include a `justification` string that mentions the absolute difference and the threshold.

4. **CRITICAL**: After reading the CSVs and before computing, print their contents and column names to stdout so you can verify the join logic. Pay special attention to:
   - How `blister_card_count` relates medications to card costs
   - Whether the 28-day model uses one card count and the 56-day model uses another
   - The prompt says packaging cost is matched by `blister_card_count` — so for 28-day fills (56 capsules), find the matching blister card count in `card_cost.csv`; for 56-day fills (112 capsules), find that matching count

5. **Create `/root/syncpack_summary.md`** with 4–8 non-empty lines including:
   - Total 28-day margin (USD)
   - Total 56-day margin (USD)
   - Absolute difference (USD)
   - The exact decision slug (`convert_to_56_day` or `keep_28_day`)

6. **Validate**:
```bash
python -m json.tool /root/syncpack_analysis.json > /dev/null && echo 'JSON valid'
cat /root/syncpack_analysis.json
cat /root/syncpack_summary.md
```

7. **Run the tests**:
```bash
cd / && python -m pytest /tests/test_outputs.py -v 2>&1 | head -80
```
If any test fails, read the error carefully, fix the issue, and re-run. Pay particular attention to key names, values in assumptions, and numeric precision.

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
Task metadata: author_email=gpt54@example.com, author_name=GPT-5.4, category=financial-analysis, difficulty=medium, tags=[med-sync, packaging, csv, json, decision-analysis].
Verifier config: timeout_sec=900.0.