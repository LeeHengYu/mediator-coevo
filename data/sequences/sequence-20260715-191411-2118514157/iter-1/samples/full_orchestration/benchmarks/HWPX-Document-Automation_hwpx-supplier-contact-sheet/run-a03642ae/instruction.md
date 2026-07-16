# Task Instruction

Complete the following task step-by-step.

## Goal
Update the HWPX supplier contact sheet `supplier_contact_template.hwpx` using the values in `supplier_contact.json`, then save the finished file to `/root/supplier_contact_ready.hwpx`.

## Steps

### 1. Understand the HWPX format
A `.hwpx` file is a ZIP-based package (like OOXML). It contains XML files inside. The main document content is typically in a file like `Contents/section0.xml` (or similar path). Explore the structure first.

### 2. Inspect the template
```bash
cd /root
ls -la
cat supplier_contact.json
```
Then explore the HWPX structure:
```bash
mkdir -p /tmp/hwpx_work
cp supplier_contact_template.hwpx /tmp/hwpx_work/template.zip
cd /tmp/hwpx_work
unzip template.zip -d template_extracted
find template_extracted -type f | sort
```

### 3. Find all placeholders
Search for `{{` across all extracted files to locate every placeholder:
```bash
grep -r '{{' template_extracted/
```
Note every file and every placeholder pattern found (e.g., `{{company_name}}`, `{{contact_person}}`, etc.).

### 4. Read the JSON data
Parse `supplier_contact.json` and note every key-value pair. Each `{{key}}` placeholder in the XML must be replaced with the corresponding JSON value.

### 5. Write a Python script to perform the replacement
Create a Python script `/tmp/hwpx_work/process.py` that:

a. Reads `supplier_contact.json` from `/root/`.
b. Opens `supplier_contact_template.hwpx` as a ZIP.
c. Iterates over every entry in the ZIP.
d. For XML/text files, reads the content as UTF-8 text, performs placeholder replacement for every key in the JSON (replacing `{{key}}` with the value), and also handles any case where the placeholder might be split across XML tags (e.g., `{{` in one text run and `}}` in another). **Important**: Check if placeholders appear cleanly within single text nodes first. If they do, simple string replacement is sufficient.
e. **Critical – remove stale layout-cache elements**: For any XML file where text was modified, remove elements that cache character layout positions. These are typically `<hp:linesegarray>` or `<lineseg>` or `<hp:lineSegArray>` elements (and their children). Use an XML parser or regex to strip these elements from modified paragraphs (or from the entire file if simpler). This ensures the document opens cleanly without overlapping characters.
f. Writes all entries (modified and unmodified) into a new ZIP file at `/root/supplier_contact_ready.hwpx`, preserving the original compression method for each entry.

### 6. Run the script
```bash
python3 /tmp/hwpx_work/process.py
```

### 7. Validate the output
Verify the result:
```bash
# Check it's a valid ZIP
unzip -t /root/supplier_contact_ready.hwpx

# Check no placeholders remain
mkdir -p /tmp/hwpx_work/output_check
cd /tmp/hwpx_work/output_check
unzip /root/supplier_contact_ready.hwpx -d output_extracted
grep -r '{{' output_extracted/ || echo 'NO PLACEHOLDERS FOUND - GOOD'

# Verify Korean labels are preserved (spot check a few)
grep -r '회사' output_extracted/ && echo 'Korean labels present'

# Verify the static note line is unchanged
grep -r '본 연락처' output_extracted/ || grep -r '참고' output_extracted/ || echo 'Check static note manually'

# Show the replaced content for visual verification
for f in $(find output_extracted -name '*.xml'); do echo "=== $f ==="; cat "$f"; echo; done
```

### 8. Fix any issues
- If any `{{...}}` placeholders remain, check whether they were split across XML elements. In that case, update the script to concatenate adjacent text runs before replacement, or use regex across the raw XML string.
- If Korean labels are missing, the replacement was too aggressive — ensure only `{{...}}` patterns are replaced.
- If layout-cache elements were not removed, verify the element names by inspecting the XML namespace and tag names, then adjust the removal logic.

## Key constraints to remember
- Every `{{...}}` placeholder must be replaced — zero remaining.
- Korean field labels (like 회사명, 담당자, etc.) must be preserved.
- Static note lines must be unchanged.
- Stale layout-cache elements (`linesegarray`, `lineSegArray`, `lineseg`, or similar) in modified paragraphs must be removed.
- Output must be a valid `.hwpx` ZIP package at `/root/supplier_contact_ready.hwpx`.

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