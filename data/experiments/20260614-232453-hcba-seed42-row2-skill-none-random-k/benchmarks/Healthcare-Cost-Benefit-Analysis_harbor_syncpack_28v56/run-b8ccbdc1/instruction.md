# Task Instruction

Execute the following steps in order:

1. **Inspect the input files** to understand their structure and contents:
 ```
 cat /root/ingredient_cost.csv
 cat /root/card_cost.csv
 cat /root/reimbursement.csv
 ```

2. **Create `/root/solve.py`** with the following logic:

 ```python
 import csv
 import json

 # Read ingredient_cost.csv
 with open('/root/ingredient_cost.csv') as f:
 reader = csv.DictReader(f)
 ingredients = {row['medication']: float(row['price_per_1000_capsules_usd']) for row in reader}

 # Read card_cost.csv
 with open('/root/card_cost.csv') as f:
 reader = csv.DictReader(f)
 cards = {int(row['blister_card_count']): float(row['card_cost_usd']) for row in reader}

 # Read reimbursement.csv
 with open('/root/reimbursement.csv') as f:
 reader = csv.DictReader(f)
 reimbursements = {row['medication']: float(row['reimbursement_per_cycle_180_patients_usd']) for row in reader}

 # Constants
 patients = 180
 fills_28 = 12
 fills_56 = 6
 caps_28 = 56
 caps_56 = 112
 threshold = 9000

 medications = []
 for med in sorted(ingredients.keys()):
 price_per_1000 = ingredients[med]
 reimb_per_cycle = reimbursements[med]

 # We need to figure out the blister_card_count for this medication.
 # The card_cost.csv maps blister_card_count -> card_cost_usd.
 # For 28-day model: 56 capsules per fill, so blister_card_count = 56
 # For 56-day model: 112 capsules per fill, so blister_card_count = 112
 # BUT the problem says packaging cost uses card_cost_usd matched by blister_card_count.
 # Let's check what blister_card_counts exist in card_cost.csv.
 # The task says "per patient per fill, matched by blister_card_count".
 # For 28-day: blister_card_count = capsules_per_fill_28_day = 56
 # For 56-day: blister_card_count = capsules_per_fill_56_day = 112
 # Wait - re-read: it says card_cost per patient per fill matched by blister_card_count.
 # This likely means the card has a certain count matching the fill size.

 # Actually, let me re-read. The schema has ONE blister_card_count per medication.
 # So the medication has a single blister_card_count, and the card cost is looked up from that.
 # For 28-day: each fill uses (capsules_per_fill / blister_card_count) cards? No...
 # The schema shows one card_cost_usd per medication, suggesting one card type.
 # Let me check: the task says 28-day = 56 caps/fill, 56-day = 112 caps/fill.
 # If blister_card_count in card_cost.csv has entries like 28 and 56 (matching cycle days),
 # then 28-day model uses card with blister_card_count=28, 56-day uses blister_card_count=56.
 # OR the entries might be 56 and 112 (matching capsule counts).
 # We need to inspect the actual CSV to determine this.

 # For now, this is a placeholder. After inspecting the CSV, adjust accordingly.
 pass
 ```

 **STOP here and inspect the CSV files first.** After seeing their contents, write the complete solve.py. The key ambiguity is how `blister_card_count` maps to the two models. Inspect `card_cost.csv` carefully - it likely has entries for both 28 and 56 (the cycle lengths), meaning:
 - 28-day model uses the card with `blister_card_count=28`
 - 56-day model uses the card with `blister_card_count=56`

 OR it may have entries matching capsule counts (56 and 112). Use whatever values actually appear in the file.

3. **After inspecting CSVs, write the complete `/root/solve.py`:**

 The core calculations per medication are:
 ```
 annual_drug_cost = (price_per_1000 / 1000) * capsules_per_fill * fills_per_year * patients
 annual_packaging_cost = card_cost_usd * fills_per_year * patients
 annual_reimbursement = reimbursement_per_cycle * fills_per_year
 annual_margin = annual_reimbursement - annual_drug_cost - annual_packaging_cost
 difference = margin_56 - margin_28
 ```

 For the JSON output:
 - Round all currency values to 2 decimal places using `round(value, 2)`
 - Sort medications alphabetically by name
 - Include all fields exactly as specified in the schema
 - The `blister_card_count` and `card_cost_usd` in each medication entry should reflect the card used for that medication (likely one card type per medication based on the schema having a single value)

 **IMPORTANT about blister_card_count in the output schema:** The schema shows ONE `blister_card_count` and ONE `card_cost_usd` per medication. This likely means each medication is associated with a specific card count. Check if `ingredient_cost.csv` or `reimbursement.csv` includes a `blister_card_count` column linking medications to card sizes. If not, the two card_cost entries (28 and 56) are used for the two models respectively, and you should store the 28-day card info in the output (or whichever makes sense given the data).

 For the decision:
 ```python
 if abs(total_difference) < 9000:
 decision = 'convert_to_56_day'
 else:
 decision = 'keep_28_day'
 ```

4. **For `/root/syncpack_summary.md`:**
 - Use **comma-formatted currency** values. This is critical (previous failure was due to missing commas).
 - Use `f'${value:,.2f}'` for formatting (e.g., `$-42,908.83` not `$-42908.83`).
 - Include 4-8 non-empty lines containing:
   - Total 28-day margin
   - Total 56-day margin  
   - Absolute difference
   - The exact decision slug (`convert_to_56_day` or `keep_28_day`)

   Example format:
   ```markdown
   # Syncpack Cycle Analysis Summary

   Total 28-day annual margin: $XX,XXX.XX
   Total 56-day annual margin: $XX,XXX.XX
   Absolute margin difference: $XX,XXX.XX
   Recommendation: convert_to_56_day
   ```

5. **Run the script:**
 ```
 cd /root && python solve.py
 ```

6. **Validate outputs:**
 ```
 cat /root/syncpack_analysis.json | python -m json.tool
 cat /root/syncpack_summary.md
 ```
   - Verify JSON is valid and parseable
   - Verify summary has comma-formatted numbers
   - Verify medications are sorted alphabetically
   - Verify all currency values are rounded to 2 decimals

7. **Run the test suite if available:**
 ```
 cd /root && python -m pytest test_output.py -v 2>&1 || true
 ```
   If tests fail, read the error messages carefully, fix the issue in solve.py, and re-run.

**Key reminders:**
- The MOST CRITICAL fix from previous feedback: use `f'{value:,.2f}'` (with comma) for all currency values in the markdown summary.
- Drug cost formula: `(price_per_1000 / 1000) * capsules_per_fill * fills * patients`
- Packaging cost formula: `card_cost * fills * patients`  
- Reimbursement formula: `reimb_per_cycle * fills` (already for 180 patients)
- Inspect the actual CSV files before writing calculations to resolve any ambiguity about how blister_card_count maps to the two models.

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