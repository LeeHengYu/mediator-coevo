# Task Instruction

Execute the following steps to produce /root/answers.json:

## Step 1: Understand the data layout

Explore the directory structure under /root/2025-q2 and /root/2025-q3 to understand what files are available. These are likely SEC 13F filings in TSV, CSV, XML, or similar format. There will be cover page data (with manager names and accession numbers) and holdings/infotable data (with CUSIPs, values, share types, etc.).

```bash
find /root/2025-q2 -type f | head -60
find /root/2025-q3 -type f | head -60
```

Examine a few representative files to understand their format:
```bash
head -30 /root/2025-q2/*.tsv 2>/dev/null || head -30 /root/2025-q2/*.csv 2>/dev/null
ls -la /root/2025-q2/ /root/2025-q3/
```

## Step 2: Identify the cover page / manager name data

Look for files that contain manager names (cover pages). These might be named something like COVERPAGE.tsv, or be in subdirectories. Search for 'bridgewater' (case-insensitive) across both quarter directories:

```bash
grep -ril 'bridgewater' /root/2025-q2/
grep -ril 'bridgewater' /root/2025-q3/
```

Once you find the cover page files, inspect them to understand the schema (columns like FILING_MANAGER_NAME, ACCESSION_NUMBER, etc.).

## Step 3: Match the closest filing manager name

For each quarter, find the filing manager name that best matches 'bridgewater associates' (case-insensitive fuzzy/substring match). Use the closest match. Note the exact manager name and accession number for each quarter.

## Step 4: Find the holdings/infotable data

Using the accession number from each quarter, locate the corresponding holdings data (infotable). This will contain columns like CUSIP, VALUE (in thousands typically for 13F), SSHPRNAMT (shares), SSHPRNAMTTYPE, INVESTMENT_DISCRETION, PUT_CALL, etc.

Inspect the infotable files:
```bash
head -20 <infotable_file>
```

## Step 5: Filter to stock-like holdings only

13F infotables have a column for share type (SSHPRNAMTTYPE) which can be 'SH' (shares) or 'PRN' (principal). They may also have PUT_CALL column. Stock-like holdings are those where:
- SSHPRNAMTTYPE is 'SH' (shares), AND
- PUT_CALL is empty/none (not a put or call option)

Filter both quarters' data accordingly.

## Step 6: Aggregate by CUSIP within each quarter

For each quarter, group by CUSIP and sum the VALUE column. This gives total value per CUSIP per quarter.

## Step 7: Compute value changes

Merge Q2 and Q3 data by CUSIP. Compute value_change = Q3_value - Q2_value for each CUSIP.

## Step 8: Determine the three result arrays

1. **top4_increased_cusips**: The 4 CUSIPs with the largest positive value change, ordered from largest increase to smallest increase.

2. **top3_decreased_cusips**: The 3 CUSIPs with the most negative value change, ordered from largest decrease (most negative first) to smaller decrease (less negative).

3. **new_positions_top2**: CUSIPs present in Q3 but absent in Q2, ordered by descending positive value change, take the top 2.

## Step 9: Write /root/answers.json

Write a Python script that does all of the above. Here is a template approach (adapt based on actual file formats discovered in Steps 1-2):

```python
import json
import os
import csv
from collections import defaultdict

# Adapt these paths and parsing based on what you discover in Steps 1-4
# This is a template - you MUST adjust based on actual file structure

def load_tsv(filepath):
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f, delimiter='\t')
        return list(reader)

def find_bridgewater_accession(quarter_dir):
    # Search cover page files for bridgewater associates
    # Return accession number of closest match
    pass

def load_holdings(quarter_dir, accession):
    # Load infotable for given accession
    pass

def filter_stock_holdings(holdings):
    # Keep only stock-like: SH type, no PUT_CALL
    pass

def aggregate_by_cusip(holdings):
    # Sum VALUE by CUSIP
    pass

# ... implement based on discovered structure
```

Write the final JSON with exact schema and quarter labels as specified.

## Critical requirements:
- The quarter labels must be exactly "2025-q2" and "2025-q3"
- The fund_query fields must be exactly "bridgewater associates"
- CUSIPs should be strings
- Output must be valid JSON written to /root/answers.json
- Do NOT guess CUSIPs. Derive them from the actual data.
- For top3_decreased_cusips, order from MOST negative (largest absolute decrease) first to least negative.
- Verify your output by reading back the file and checking it parses as valid JSON with the correct keys.

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
Task metadata: author_email=noreply@anthropic.com, author_name=Claude, category=finance, difficulty=hard, tags=[data processing, financial analysis, 13f].
Verifier config: timeout_sec=900.0.