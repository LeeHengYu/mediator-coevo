# Task Instruction

Complete the clinic intake summary by filling a .hwpx template with patient data.

## Context
- `.hwpx` files are ZIP archives containing XML files (Korean Hangul word processor format)
- The template `clinic_intake_template.hwpx` contains `{{...}}` placeholders
- Patient data is in `patient_intake.json`
- Output: `/root/clinic_intake_ready.hwpx`

## Step-by-step Plan

### Step 1: Inspect the input files
1. Run `cat patient_intake.json` to see all patient data fields and values.
2. Run `file clinic_intake_template.hwpx` to confirm it's a ZIP.
3. Copy the template to a working directory: `cp clinic_intake_template.hwpx /tmp/work.hwpx`
4. `mkdir -p /tmp/hwpx_work && cd /tmp/hwpx_work && unzip -o /tmp/work.hwpx`
5. Run `find /tmp/hwpx_work -type f` to list all files in the package.
6. For each XML file found (especially under `Contents/`), `cat` it and look for `{{` placeholder patterns. Record every unique placeholder string and every file that contains one.

### Step 2: Understand the template structure
1. Identify the main content XML (likely `Contents/section0.xml` or similar).
2. List ALL `{{...}}` placeholders found across all files. Note which ones repeat.
3. Check for layout-cache elements: look for XML elements like `<hp:linesegarray>`, `<hp:lineSegArray>`, `<hp:lineseg>`, `<hc:linesegarray>`, or similar caching tags near text runs. These are the "stale layout-cache elements" that must be removed from modified paragraphs.

### Step 3: Build the replacement script
Write a Python script `/tmp/fill_template.py` that does the following:

```python
import json
import zipfile
import os
import re
import shutil
from datetime import datetime, date

# 1. Load patient data
with open('patient_intake.json', 'r', encoding='utf-8') as f:
    patient = json.load(f)

# 2. Print all keys and values for debugging
print("Patient data:")
for k, v in patient.items():
    print(f"  {k}: {v}")

# 3. Prepare derived values:

# Age calculation: Korean full-year age (만 나이) as of visit date
# Parse birth date and visit date from the JSON (inspect actual field names first)
# Korean full-year age = age based on whether birthday has passed in the visit year
# Formula: visit_year - birth_year - (1 if birthday hasn't occurred yet this year else 0)
# Format the age note as "(<N>세)"

# Phone normalization: convert to 000-0000-0000 format
# Strip all non-digit characters, then format as XXX-XXXX-XXXX

# 4. Build replacement dictionary mapping each {{placeholder}} to its value
# Include the age note appended after the birth date value

# 5. Extract the hwpx
work_dir = '/tmp/hwpx_extracted'
if os.path.exists(work_dir):
    shutil.rmtree(work_dir)
os.makedirs(work_dir)

with zipfile.ZipFile('clinic_intake_template.hwpx', 'r') as zin:
    zin.extractall(work_dir)

# 6. Process each file in the archive
# For each file, read content. If it contains '{{', perform replacements.
# For XML files with modified paragraphs, remove layout cache elements.

# 7. Remove layout cache: 
# In hwpx XML, layout caches are typically <hp:linesegarray>...</hp:linesegarray>
# or similar elements within <hp:p> paragraph elements.
# For ANY paragraph that was modified (contained a placeholder), remove these elements.
# Use XML parsing (lxml or xml.etree) to do this properly.

# 8. Verify no {{...}} remains in any file

# 9. Repack as ZIP with same structure
output_path = '/root/clinic_intake_ready.hwpx'
# Use zipfile to recreate, preserving directory structure and compression
```

**IMPORTANT**: Do NOT write this script blindly. First complete Steps 1 and 2 to see:
- The exact field names in the JSON
- The exact placeholder names in the XML
- The exact XML namespace prefixes and element names for layout caches
- The date format used in the JSON
- The phone number format in the JSON

Then adapt the script accordingly.

### Step 4: Handle layout cache removal carefully
When parsing the XML to remove layout caches from modified paragraphs:
1. Use `xml.etree.ElementTree` with proper namespace handling.
2. Identify paragraph elements (likely `<hp:p>` or similar).
3. For each paragraph, check if its text content was modified (contained a placeholder).
4. If modified, find and remove child elements that represent layout caches (e.g., `linesegarray`, `lineSegArray`, `lineseg` — inspect the actual element names from the template).
5. Be careful with namespace prefixes — use the actual namespaces from the XML files.
6. **Alternative approach if XML parsing is complex**: Use regex-based removal of layout cache elements ONLY within paragraphs that contained placeholders. But prefer proper XML parsing.

### Step 5: Age calculation details
- Korean full-year age (만 나이): Calculate based on the visit date. If the person's birthday has not yet occurred in the visit year, subtract 1 from (visit_year - birth_year).
- Format: `(<N>세)` — e.g., `(45세)`
- This should appear right after the birth date text, likely replacing a placeholder or appended to the birth date placeholder value.

### Step 6: Phone number normalization
- Extract only digits from the callback phone number.
- Format as `000-0000-0000` (3-4-4 grouping, standard Korean mobile format).
- If the number has 11 digits (e.g., 01012345678), format as `010-1234-5678`.
- If it has 10 digits, format as `02-1234-5678` or `031-123-4567` — use 3-4-4 if 11 digits, otherwise 2-4-4 or 3-3-4 as appropriate. **Check the actual number length first.**

### Step 7: Validate the output
1. Run: `python3 -c "import zipfile; z=zipfile.ZipFile('/root/clinic_intake_ready.hwpx'); z.testzip(); print('ZIP OK')"` to verify valid ZIP.
2. Extract the output and grep for `{{` across all files: `mkdir -p /tmp/verify && cd /tmp/verify && unzip -o /root/clinic_intake_ready.hwpx && grep -r '{{' . || echo 'No placeholders remain'`
3. Verify the age note appears: `grep -r '세)' /tmp/verify/`
4. Verify the phone format: `grep -r '[0-9]\{3\}-[0-9]\{4\}-[0-9]\{4\}' /tmp/verify/`
5. Verify Korean labels are preserved: `grep -r` for a few known Korean labels from the original.
6. Verify handwritten-signature note is preserved.
7. Check that layout cache elements were removed from modified paragraphs.

### Critical Reminders
- The `.hwpx` ZIP must preserve the exact directory structure and all non-modified files.
- When repacking, use `zipfile.ZIP_DEFLATED` compression and walk the extracted directory to add all files with their relative paths.
- Do NOT add the root extraction directory itself as a path prefix.
- Ensure UTF-8 encoding is preserved in all XML files.
- If the template has a `mimetype` file, it may need to be stored uncompressed (first entry, no compression) — check the original ZIP structure.
- Read each file BEFORE and AFTER editing to confirm changes landed correctly.

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