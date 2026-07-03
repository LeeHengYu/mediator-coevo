# Task Instruction

You need to finish the clinic intake summary by filling in a template HWPX document with patient data from a JSON file.

## Steps

### 1. Inspect the workspace
```bash
ls /root/
find /root/ -name '*.hwpx' -o -name '*.json' | head -20
```

### 2. Read the patient data
```bash
cat /root/patient_intake.json
```
Note all field values. Pay special attention to:
- Patient name (may appear multiple times in the template)
- Birth date and visit date (needed for Korean full-year age calculation)
- Phone number (needs normalization to `000-0000-0000` format)

### 3. Examine the HWPX template structure
HWPX files are ZIP archives containing XML files.
```bash
mkdir -p /tmp/hwpx_work
cp /root/clinic_intake_template.hwpx /tmp/hwpx_work/template.zip
cd /tmp/hwpx_work
unzip template.zip -d template_extracted
find template_extracted -type f | sort
```

### 4. Find all placeholders
Search for `{{` across all XML files to identify every placeholder:
```bash
grep -rn '{{' template_extracted/
```
Also check if placeholders might be split across XML tags by examining the raw XML around any `{{` occurrences:
```bash
for f in $(find template_extracted -name '*.xml'); do cat "$f"; echo '===END OF FILE==='; done
```

### 5. Compute the Korean full-year age
Korean full-year age (만 나이) = the age as of the visit date, computed as:
- years = visit_year - birth_year
- if the visit date hasn't reached the birthday yet in that year, subtract 1

Format the age note as `(<N>세)` to be appended after the birth date value.

### 6. Normalize the phone number
Strip all non-digit characters from the callback phone number, then format as `XXX-XXXX-XXXX` (3-4-4 grouping for Korean mobile numbers).

### 7. Write a Python script to perform all replacements
Create a Python script that:
1. Extracts the HWPX ZIP to a temp directory
2. Reads patient_intake.json
3. For each XML file (especially section*.xml files in the Contents/ directory):
   a. Reads the full XML text
   b. First, handles potentially split placeholders: join adjacent `<hp:t>` text content within the same paragraph run if they form part of a placeholder pattern. A simpler approach that worked before: just do string replacement on the raw XML text, since `{{placeholder}}` markers are likely intact within single `<hp:t>` elements.
   c. Replaces ALL `{{placeholder}}` occurrences with the corresponding values from the JSON
   d. For the birth date placeholder, appends the age note ` (<N>세)` after the birth date value
   e. For the phone placeholder, uses the normalized phone number
   f. Ensures the patient name replacement covers ALL occurrences (including any confirmation line)
   g. **Strips layout cache elements**: Remove all `<hp:lineSegArray>...</hp:lineSegArray>` elements (including their content) from any paragraph whose text was modified. This prevents overlapping character rendering. Use regex: `re.sub(r'<hp:lineSegArray>.*?</hp:lineSegArray>', '', xml_text, flags=re.DOTALL)` — it's safe to strip ALL lineSegArray elements from modified files since the word processor will regenerate them.
4. Writes modified XML files back
5. Re-packages everything into a valid ZIP with `.hwpx` extension at `/root/clinic_intake_ready.hwpx`, preserving the original directory structure and using ZIP_DEFLATED compression

### 8. Verify no placeholders remain
```bash
mkdir -p /tmp/verify
cp /root/clinic_intake_ready.hwpx /tmp/verify/output.zip
cd /tmp/verify
unzip output.zip -d output_extracted
grep -rn '{{' output_extracted/
```
This grep must return NO results.

### 9. Verify the output is a valid ZIP/HWPX
```bash
python3 -c "import zipfile; z=zipfile.ZipFile('/root/clinic_intake_ready.hwpx'); print('Valid ZIP, files:', z.namelist()); z.close()"
```

### 10. Verify key content is present
Check that the age note, normalized phone, patient name, and Korean labels are all present:
```bash
grep -r '세)' output_extracted/
grep -r '0000' output_extracted/  # phone pattern
```

### 11. Run the verifier if available
```bash
cd /root && find . -name 'test_output*' -o -name 'test_outputs*' | head -5
# If found:
python3 -m pytest test_output.py -v 2>&1 || python3 -m pytest tests/test_outputs.py -v 2>&1
```

## Critical Details
- The age note format is exactly `(<N>세)` with a space before the opening parenthesis when appended after the birth date.
- Phone normalization: digits only, then format as `XXX-XXXX-XXXX`.
- ALL `{{...}}` placeholders must be replaced — check for repeated patient name placeholders.
- Strip `<hp:lineSegArray>` elements from modified XML to prevent stale layout cache.
- Preserve all existing Korean labels and the handwritten-signature note — do NOT remove or alter text that isn't a placeholder.
- The output must be saved to exactly `/root/clinic_intake_ready.hwpx`.

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