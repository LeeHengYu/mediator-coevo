# Task Instruction

Complete the clinic intake summary by filling a HWPX template with patient data. Follow these steps precisely:

## Step 1: Inspect the workspace
```
ls -la /root/
cat /root/patient_intake.json
```

## Step 2: Understand the HWPX structure
HWPX files are ZIP archives containing XML files. Unzip the template to inspect its structure:
```
mkdir -p /tmp/hwpx_work
cp /root/clinic_intake_template.hwpx /tmp/hwpx_work/template.zip
cd /tmp/hwpx_work
unzip template.zip -d template_contents
find template_contents -type f
```

## Step 3: Inspect all XML content files
Read every XML file in the extracted package, especially files under `Contents/` (likely `section0.xml` or similar). Identify:
- All `{{...}}` placeholders and their exact text
- The structure of paragraphs and text runs
- Any layout-cache elements (look for tags like `<hp:linesegarray>`, `<hp:lineSegArray>`, `<hp:lineseg>`, `<hp:LineSeg>`, or similar caching elements within paragraph tags)
- Korean labels and the handwritten-signature note

```
for f in $(find template_contents -name '*.xml'); do echo "=== $f ==="; cat "$f"; echo; done
```

Also check for any `.rels` files or other non-XML content.

## Step 4: Parse patient_intake.json and compute derived values
Write a Python script `/tmp/hwpx_work/fill_template.py` that:

### 4a: Reads patient_intake.json
Load all fields from the JSON file.

### 4b: Computes Korean full-year age
Korean full-year age (만 나이) as of the visit date: `age = visit_year - birth_year`, adjusted down by 1 if the visit date is before the birthday in that year. Format as `(<N>세)`. This note must be appended right after the birth date value in the document.

### 4c: Normalizes phone number
Strip all non-digit characters from the callback phone number, then format as `000-0000-0000` (3-4-4 grouping). If the number has 11 digits (common Korean mobile format like 01012345678), split as `010-1234-5678`. If 10 digits, split as `02-1234-5678` or `031-123-4567` depending on area code length — but default to 3-4-4 for 11-digit numbers.

### 4d: Performs placeholder replacement
- Extract the template HWPX (ZIP)
- For each XML file in the package, find all `{{...}}` patterns
- IMPORTANT: Placeholders might be split across multiple XML text runs/spans within a paragraph. You MUST handle this case. Read the full text content of each paragraph, check if concatenated text contains `{{...}}` patterns, and if a placeholder spans multiple runs, merge those runs into one before replacing.
- Replace each placeholder with the corresponding value from patient_intake.json
- For the birth date placeholder, append ` (<N>세)` with the computed age
- For the phone placeholder, use the normalized format
- Ensure ALL occurrences are replaced, including repeated ones (e.g., patient name may appear multiple times)

### 4e: Removes stale layout-cache elements
For any paragraph whose text content was modified, remove layout-cache child elements. Look for elements with local names like `linesegarray`, `LineSeg`, `lineSegArray`, `lineseg`, or any element that appears to be a layout cache. Remove them entirely from modified paragraphs so the document renders cleanly.

### 4f: Repackages as HWPX
- Repackage all files back into a ZIP with `.hwpx` extension
- IMPORTANT: Use the same compression method and directory structure as the original
- The mimetype file (if present) should be stored uncompressed as the first entry (like ODF/EPUB convention)
- Save to `/root/clinic_intake_ready.hwpx`

### 4g: Validates the output
- Unzip the output and verify no `{{` or `}}` remains in any file
- Verify the age note is present
- Verify the phone number is in correct format
- Verify Korean labels and signature note are preserved
- Print all text content for visual verification

## Step 5: Run the script
```
cd /tmp/hwpx_work
python3 fill_template.py
```

## Step 6: Final validation
```
mkdir -p /tmp/hwpx_work/output_check
cp /root/clinic_intake_ready.hwpx /tmp/hwpx_work/output_check/output.zip
cd /tmp/hwpx_work/output_check
unzip output.zip -d output_contents
grep -r '{{' output_contents/ || echo 'No placeholders remaining - GOOD'
grep -r '}}' output_contents/ || echo 'No closing braces remaining - GOOD'
for f in $(find output_contents -name '*.xml'); do echo "=== $f ==="; cat "$f"; echo; done
```

Verify that:
1. No `{{...}}` text remains anywhere
2. The age note `(<N>세)` appears after the birth date
3. The phone number is in `000-0000-0000` format
4. Korean labels are intact
5. The handwritten-signature note is preserved
6. The file is a valid ZIP archive

## Critical Notes
- Do NOT hardcode placeholder names before inspecting the template. Read the actual XML first, then build the replacement map.
- Handle the case where XML tags split a placeholder across multiple text elements within a single paragraph.
- Be careful with XML namespaces in HWPX files — use namespace-aware parsing.
- When removing layout cache elements, be conservative: only remove from paragraphs you actually modified.
- The output path MUST be exactly `/root/clinic_intake_ready.hwpx`.

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