# Task Instruction

You must fill in the HWPX training feedback template and save the result. Follow these steps precisely:

1. **Inspect the workspace:**
   ```bash
   ls /root/
   ```
   Locate `training_feedback_template.hwpx` and `training_feedback.json`.

2. **Read the JSON data:**
   ```bash
   cat /root/training_feedback.json
   ```
   Note every key-value pair. You will need all of them.

3. **Unzip the HWPX template to a temp directory and inspect the XML:**
   ```bash
   mkdir -p /tmp/hwpx_work
   cp /root/training_feedback_template.hwpx /tmp/hwpx_work/template.zip
   cd /tmp/hwpx_work && unzip -o template.zip -d template_contents
   ```
   Then inspect the section XML files:
   ```bash
   find /tmp/hwpx_work/template_contents -name '*.xml' | sort
   cat /tmp/hwpx_work/template_contents/Contents/section0.xml
   ```
   If there are additional section files (section1.xml, etc.), inspect those too. Identify every `{{...}}` placeholder across ALL XML files.

4. **Write a Python script** (`/tmp/hwpx_work/fill_template.py`) that does the following:

   a. Load `training_feedback.json`.
   
   b. Unzip `training_feedback_template.hwpx` into a temporary directory.
   
   c. For each XML file under `Contents/` (section0.xml, section1.xml, etc.), parse with `lxml.etree` preserving namespaces.
   
   d. Walk every text node (`elem.text` and `elem.tail` for all elements). For each `{{KEY}}` placeholder found:
      - Replace with the corresponding JSON value, applying these transformations:
        - **참석자수**: Extract digits only from the JSON value (e.g., "32명" → "32").
        - **만족도**: Format as `X.X점 (5.0점 만점)` where X.X is the numeric score from JSON (e.g., if JSON has "4.5" or "4.5/5.0", output "4.5점 (5.0점 만점)").
        - **종합의견** (or whatever key maps to the overall opinion): Append ` 후속 심화반 검토 요망.` after the JSON value (with a space before 후속).
        - All other keys: substitute the JSON value directly.
   
   e. **Critical: Remove stale layout caches.** For every `<hp:p>` paragraph element that had any text modification, find and remove all child `<hp:lineSegArray>` elements (and their children). Use the actual namespace URI found in the document (look for the `hp` namespace prefix or the URI like `http://www.hancom.co.kr/hwpml/2011/paragraph`).
   
   f. Serialize each modified XML back, writing it to the same relative path in the temp directory.
   
   g. Re-zip the entire temp directory into `/root/training_feedback_ready.hwpx`, preserving the original directory structure exactly. Use `zipfile.ZipFile` and walk the directory, making sure paths inside the zip match the original structure (no extra top-level folder).

5. **Run the script:**
   ```bash
   cd /tmp/hwpx_work && python3 fill_template.py
   ```

6. **Validate the output thoroughly:**
   ```bash
   # Check it's a valid zip
   python3 -c "import zipfile; z=zipfile.ZipFile('/root/training_feedback_ready.hwpx'); print(z.namelist()); z.close()"
   
   # Check no {{...}} placeholders remain
   mkdir -p /tmp/hwpx_verify && cd /tmp/hwpx_verify && unzip -o /root/training_feedback_ready.hwpx
   grep -r '{{' /tmp/hwpx_verify/ && echo 'FAIL: placeholders remain' || echo 'PASS: no placeholders'
   
   # Verify specific transformations
   grep -o '[0-9]*점 ([0-9.]*점 만점)' /tmp/hwpx_verify/Contents/section*.xml || echo 'Check 만족도 format'
   grep '후속 심화반 검토 요망' /tmp/hwpx_verify/Contents/section*.xml && echo 'PASS: appended text found' || echo 'FAIL: appended text missing'
   
   # Verify no lineSegArray in modified paragraphs (spot check)
   grep -c 'lineSegArray' /tmp/hwpx_verify/Contents/section*.xml
   ```

7. **Run the test suite if present:**
   ```bash
   cd /root && python3 -m pytest test_output.py -v 2>&1 | head -80
   ```
   If tests fail, read the failure messages carefully, fix the script, re-run, and re-validate.

**Key pitfalls to avoid (from cross-task experience):**
- Make sure the namespace handling is dynamic — extract the actual namespace URIs from the parsed XML rather than hardcoding them.
- Ensure the zip structure matches exactly (no extra wrapper directory).
- The 참석자수 transformation must produce digits only (use `re.sub(r'[^0-9]', '', value)`).
- The 만족도 transformation must extract the numeric score and format precisely as `X.X점 (5.0점 만점)`.
- The 종합의견 append must add exactly ` 후속 심화반 검토 요망.` (space + text + period).
- Do NOT skip any section XML files — check ALL of them for placeholders.
- Korean labels and the static note line must remain unchanged.

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