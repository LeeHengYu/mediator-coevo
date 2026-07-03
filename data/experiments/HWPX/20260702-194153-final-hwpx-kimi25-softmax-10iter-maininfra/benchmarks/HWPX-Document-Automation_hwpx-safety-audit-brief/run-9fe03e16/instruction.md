# Task Instruction

Complete the warehouse safety audit brief by filling the HWPX template with data from the provided JSON files.

## Step-by-step Plan

### 1. Explore the workspace
```bash
find /root -maxdepth 2 -type f | head -60
ls -la /root/
```
Identify the template file (`safety_audit_template.hwpx`), `audit_overview.json`, and `corrective_actions.json`.

### 2. Read the JSON data files
```bash
cat /root/audit_overview.json
cat /root/corrective_actions.json
```
Note all field names, values, the risk tier, and the inspection date (in YYYY-MM-DD format).

### 3. Inspect the HWPX template structure
HWPX is a ZIP archive. Unzip it to a temp directory and examine the XML content:
```bash
mkdir -p /tmp/hwpx_work
cp /root/safety_audit_template.hwpx /tmp/hwpx_work/template.hwpx
cd /tmp/hwpx_work
unzip template.hwpx -d template_extracted
find template_extracted -type f
cat template_extracted/Contents/section0.xml
```
Also check if there's a section1.xml or other content files. Identify:
- All `{{...}}` placeholders and where they appear
- The summary/overview section fields
- The audit table value cells
- The three corrective-action lines
- Every occurrence of the risk tier placeholder
- Every occurrence of the inspection date placeholder
- XML namespaces used

### 4. Write a Python script to perform all replacements
Create `/tmp/hwpx_work/fill_template.py` that does the following:

```python
import json, zipfile, os, re, copy
import xml.etree.ElementTree as ET

# Load JSON data
with open('/root/audit_overview.json') as f:
    overview = json.load(f)
with open('/root/corrective_actions.json') as f:
    actions = json.load(f)

# Define namespace map from the XML file (parse it first to discover namespaces)
# Register all namespaces to preserve them on write

# Key processing logic:
# a) For each section XML (section0.xml, possibly section1.xml):
#    - Parse the XML
#    - For each paragraph (<hp:p>), collect all text from <hp:t> elements
#    - Join the text to reconstruct the full paragraph text
#    - Perform placeholder replacements using the JSON data
#    - Handle the risk tier: replace ALL occurrences, and append severity note
#      using mapping: High -> 즉시조치, Medium -> 계획보완, Low -> 모니터링
#      Format: "<risk_tier> (<severity_korean>)"
#    - Reformat inspection date from YYYY-MM-DD to YYYY.MM.DD everywhere
#    - Fill corrective action lines in order from corrective_actions.json
#    - Ensure NO {{...}} placeholders remain
#    - For any modified paragraph, remove <hp:lineSegArray> elements (layout cache)
#    - Redistribute the replaced text back into the <hp:t> elements
#      (simplest: put all text in the first <hp:t>, clear the rest)

# b) Repackage the HWPX:
#    - Copy all files from the extracted archive
#    - Replace the modified section XML(s)
#    - Write to /root/safety_audit_brief_final.hwpx
```

**Critical details for the script:**

- **Placeholder joining**: Placeholders like `{{audit_date}}` may be split across multiple `<hp:t>` tags (e.g., `{{audit`, `_date}}`). You MUST join all `<hp:t>` text within each `<hp:r>` run or even across runs within a paragraph to detect and replace them.
- **Strategy**: For each `<hp:p>` paragraph, collect ALL text from ALL `<hp:t>` descendants, join them, do replacements, then put the full replaced text into the first `<hp:t>` and set all other `<hp:t>` elements to empty string.
- **Layout cache removal**: For every `<hp:p>` where text was modified, find and remove all `<hp:lineSegArray>` child elements. This prevents overlapping character display.
- **Namespace preservation**: Before parsing, read the raw XML to extract namespace declarations. Register them all with `ET.register_namespace()` so they're preserved in output.
- **Date reformatting**: Use regex `r'(\d{4})-(\d{2})-(\d{2})'` → `r'\1.\2.\3'` but ONLY for the specific inspection date value, applied after placeholder substitution.
- **Risk tier + severity note**: After replacing the risk tier placeholder with the actual value (e.g., "High"), also append the Korean severity note. So the final text should be like `High (즉시조치)`. Apply this everywhere the risk tier appears.
- **Corrective actions**: Fill in the same order as they appear in the JSON array.
- **Validation**: After all replacements, scan the entire XML text for `{{` and `}}` to confirm no placeholders remain.

### 5. Run the script
```bash
cd /tmp/hwpx_work
python3 fill_template.py
```

### 6. Validate the output
```bash
# Check it's a valid ZIP/HWPX
unzip -t /root/safety_audit_brief_final.hwpx

# Extract and check the section XML
mkdir -p /tmp/hwpx_verify
cd /tmp/hwpx_verify
unzip /root/safety_audit_brief_final.hwpx -d verify
cat verify/Contents/section0.xml

# Verify no placeholders remain
grep -r '{{' verify/Contents/ && echo 'FAIL: placeholders remain' || echo 'PASS: no placeholders'

# Verify date format is YYYY.MM.DD (not YYYY-MM-DD)
grep -oP '\d{4}-\d{2}-\d{2}' verify/Contents/section0.xml && echo 'FAIL: old date format found' || echo 'PASS: dates reformatted'

# Verify severity note is present
grep -o '즉시조치\|계획보완\|모니터링' verify/Contents/section0.xml

# Verify no lineSegArray in modified paragraphs (ideally none remaining)
grep -c 'lineSegArray' verify/Contents/section0.xml
```

### 7. Run verifier tests if available
```bash
find /root -name 'test_*.py' -o -name '*test*.py' | head -10
# If tests exist:
cd /root && python3 -m pytest tests/ -v 2>&1 | tail -40
```

## Important Reminders
- Do NOT skip the exploration steps. The exact placeholder names and XML structure must be read from the actual files.
- The corrective actions must be filled in the EXACT order from the JSON array.
- Every single `{{...}}` must be replaced — check thoroughly.
- The risk tier severity note format should match exactly what the verifier expects. Look at the template to see if there's a pattern for how notes should be appended.
- Keep all existing section titles and row labels unchanged.
- The output must be a proper ZIP-based HWPX package (not just an XML file).

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
Task metadata: author_email=catpaw@example.com, author_name=CatPaw Task Engineer, category=document-editing, difficulty=medium, tags=[hwpx, xml-editing, document-processing, latent-method-reuse].
Verifier config: timeout_sec=600.0.