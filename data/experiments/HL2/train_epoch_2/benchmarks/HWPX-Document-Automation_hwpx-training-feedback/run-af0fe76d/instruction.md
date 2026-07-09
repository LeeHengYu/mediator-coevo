# Task Instruction

## Task: Fill in training feedback HWPX template and save the result

### Goal
Replace all `{{...}}` placeholders in `training_feedback_template.hwpx` with values from `training_feedback.json`, apply the required transformations, and save the result to `/root/training_feedback_ready.hwpx`.

### Step-by-step plan

#### 1. Understand the HWPX format
A `.hwpx` file is a ZIP-based package (like OOXML). It contains XML files inside. First, explore the structure:
```bash
cd /root
cp training_feedback_template.hwpx template_backup.hwpx
mkdir -p hwpx_work
cd hwpx_work
unzip -o ../training_feedback_template.hwpx
find . -type f | head -60
```

#### 2. Read the JSON data
```bash
cat /root/training_feedback.json
```
Note all key-value pairs. You will need to map each `{{key}}` placeholder to its JSON value.

#### 3. Identify all placeholders
Search every XML file in the extracted package for `{{` patterns:
```bash
grep -rn '{{' . --include='*.xml'
```
Also check `.rels` files and any other text files:
```bash
grep -rn '{{' .
```
Record every placeholder found and which file it appears in.

#### 4. Apply replacements with transformations
Write a Python script (`/root/fill_template.py`) that:

a. Reads `training_feedback.json`.

b. Extracts the HWPX (ZIP) to a temp directory.

c. For every file in the extracted archive, if it's a text/XML file, perform these replacements:
   - For each JSON key, replace `{{key}}` with the JSON value, EXCEPT for the keys below that need transformation.
   - **`참석자수`**: Convert to digits only. E.g., if JSON has `"참석자수": "25명"`, write `25`. Use regex to extract digits: `re.sub(r'[^0-9]', '', value)`.
   - **`만족도`**: Rewrite as `X.X점 (5.0점 만점)` style. E.g., if JSON has `"만족도": "4.5"` or `"만족도": 4.5`, output `4.5점 (5.0점 만점)`. Extract the numeric score and format it.
   - **Overall opinion / `종합의견` or similar key**: After substituting the JSON comment value, append ` 후속 심화반 검토 요망.` (with a space before it) at the end of that text. Be careful: identify which placeholder corresponds to the overall opinion by inspecting the template. It might be `{{종합의견}}` or `{{overall_opinion}}` or similar.

d. **CRITICAL — Remove stale layout-cache elements**: After text replacement, for any `<hp:p>` (paragraph) element whose text content was modified, remove all `<hp:linesegarray>` or `<hc:lineseg>` child elements (these are layout cache). Use an XML parser (lxml or ElementTree) for XML files to do this cleanly:
   - Parse each XML file that had replacements.
   - For each paragraph element that contained a `{{...}}` placeholder (i.e., was modified), find and remove any layout-cache sub-elements. Common tag patterns: elements with local name `linesegarray`, `lineseg`, `lineSegArray`, `lineSeg`. Check the actual XML namespace and tag names in the template.
   - Serialize back to XML preserving the original encoding declaration and namespaces.

e. **Important**: Some placeholders may be split across multiple XML runs/spans (e.g., `<hp:t>{{</hp:t><hp:t>key}}</hp:t>`). Before doing simple string replacement, check if this is the case. If so, you need to consolidate the text within a paragraph before replacing. Strategy:
   - For each paragraph, collect all text content from child `<hp:t>` elements.
   - Join them, check if the joined text contains `{{...}}`.
   - If yes, perform the replacement on the joined text, put the result in the first `<hp:t>`, and clear the remaining `<hp:t>` elements (or remove their parent runs).

f. Repackage the modified files back into a ZIP with `.hwpx` extension at `/root/training_feedback_ready.hwpx`. Preserve the original ZIP structure (directory layout, file paths). Use `zipfile.ZipFile` in Python. Make sure to preserve the `mimetype` file (if present) as the first entry, uncompressed (like ODF convention), if the original template does this.

#### 5. Validate the output
```bash
# Check it's a valid ZIP
unzip -t /root/training_feedback_ready.hwpx

# Check no placeholders remain
mkdir -p /root/hwpx_verify
cd /root/hwpx_verify
unzip -o /root/training_feedback_ready.hwpx
grep -rn '{{' .
# This MUST return nothing.

# Verify the transformed values appear
grep -rn '후속 심화반 검토 요망' .
grep -rn '점 (5.0점 만점)' .
grep -rn '참석자수' .  # Check the digit-only value appears near this label
```

#### 6. Important constraints
- Do NOT change Korean labels or static note lines — only replace `{{...}}` placeholders.
- Ensure NO `{{...}}` text remains anywhere in the output package.
- The file must be a valid `.hwpx` (ZIP) package.
- Remove layout-cache elements (`linesegarray`/`lineseg` or similar) from any paragraph whose text was modified, so the document opens cleanly.

### Error handling
- If placeholders are split across XML runs, handle the consolidation as described in step 4e.
- If a JSON key doesn't match any placeholder, log a warning but continue.
- If a placeholder doesn't match any JSON key, log a warning — this likely means you need to check for alternate key names or nested JSON structure.
- After the script runs, always verify with the grep checks in step 5.

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