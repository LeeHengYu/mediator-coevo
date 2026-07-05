# July 4 note

## Part 1. Application scenario

We consider a stream of related tasks. The tasks do not have to be identical, but they should share enough structure that experience from one task may help another. Some tasks may already have been executed, so the system has run artifacts such as traces, verifier outputs, rewards, failure reports, mediator summaries, and successful solution patterns.

When a new task arrives, whether or not the system has seen the exact task before, the system first places it into a task graph. The graph represents estimated transfer relationships between the new task and previous tasks: which tasks are related, in which direction experience may transfer, and how strong that transfer is expected to be.

The system then uses a diffusion policy to decide which prior run artifacts should be passed to the task-execution agent. The task-execution agent should not receive all past experience. It should receive a selected, budgeted, and rendered context containing artifacts that are likely to help the current task.

The task-execution agent then solves the task using the current task input plus the diffused context. After execution, the system may receive feedback from an automatic verifier, a reward function, a human evaluator, or an optional LLM judge. The LLM judge is not required for every task. It is mainly useful when the task outcome cannot be directly verified or when semantic quality needs to be evaluated.

After feedback is collected, the new run produces new artifacts. These artifacts are added to the experience bank. Future tasks can then benefit from the expanded artifact bank. In deployment, the artifact bank can keep growing, but the graph-construction and diffusion-policy harness should normally remain fixed unless it is updated later through an offline retraining and validation process.

The application loop is:

```text
new task arrives
-> place the task in the task graph
-> select and diffuse useful prior artifacts
-> task-execution agent solves the task
-> verifier / reward function / human evaluator / optional LLM judge gives feedback
-> new artifacts enter the experience bank
-> future tasks use the expanded experience base
```

The key distinction is that the incoming task is externally requested. Diffusion decides what prior experience should be sent to that task. It should not normally decide whether the task exists or whether the user needs the task done.

## Part 2. Agentic system design

At a high level, the system has two online agents: an orchestrator and a task-execution agent. We also separate out an offline heuristic-learning agent.

The main distinction is simple: offline learns the harness, and online runs the harness. Offline, the heuristic-learning agent studies training runs and updates the harness for the orchestrator. Online, the orchestrator uses the frozen harness to make task-specific routing decisions for incoming tasks. The artifact bank may grow online, but the harness itself should not update after every task.

### 2.1 Task-execution agent

The task-execution agent is responsible for solving the current task. It receives the task input together with the context selected by the orchestrator, then uses that information to produce the task solution. It does not decide which past artifacts should be used, nor does it update the task graph or diffusion policy.

The internal design of the task-execution agent can be left to the specific task set or experimental environment. For some tasks, it may be a planner plus a tool-using executor. For others, it may be a code agent, a workflow agent, or a domain-specific solver. The framework only requires that it can receive the task input and selected context, execute the task, and return outputs and traces.

In the current repository, this role is implemented by the Planner + Executor pipeline. The Planner rewrites or plans the task using the selected context, and the Executor runs the resulting instruction in the task environment. This already matches the intended boundary: selected context is prepared outside the Planner, while the Planner focuses on using that context to plan the task. A thin wrapper around the current Planner + Executor pipeline may be useful to expose this role as a single task-execution-agent interface, without changing the underlying behavior. This implementation can change for other task sets, but the conceptual role stays the same.

The boundary is simple: the orchestrator decides what experience should be sent, while the task-execution agent decides how to use that experience to solve the task.

### 2.2 Orchestrator

The orchestrator is responsible for experience-level decisions. It does not solve the task directly. Instead, it decides how past experience should be organized and passed to the task-execution agent.

At runtime, the orchestrator contains two LLM-assisted roles. The first is a graph heuristic role, which places the current task into a task graph and estimates its relationship to previous tasks. The second is a diffusion policy role, which decides which prior run artifacts should be sent to the current task.

The runtime orchestrator uses the learned harness for these two roles. It does not learn the LLM weights, and it does not need to update itself online. The harness can stay frozen during deployment, while runtime LLM calls still make task-specific decisions under that harness.

The clean boundary is: the orchestrator decides what experience should be sent, while the task-execution agent uses that experience to solve the task.

### 2.3 Offline heuristic-learning agent

The offline heuristic-learning agent is separate from the runtime orchestrator. Its job is to learn and modify the harness used by the orchestrator's two LLM-assisted roles: the graph heuristic role and the diffusion policy role.

It uses training runs and feedback to revise how these roles should operate. After the revised harness is validated, it can be frozen and deployed. During deployment, the runtime orchestrator uses this frozen harness instead of updating itself after every task.

This separation keeps learning and deployment clean. The system can still make task-specific runtime decisions with LLM calls, but the rules governing those calls are learned offline and then fixed for deployment.

## Part 3. Experiment design

### 3.1 Evaluation unit and metrics

The evaluation unit should be a task sequence, not an isolated task. This matches the application scenario: tasks arrive over time, each completed task produces new artifacts, and later tasks may benefit from earlier experience.

For a test sequence, the harness is frozen. The artifact bank may grow as tasks are executed, but the graph and diffusion harness should not be updated online. At task `t`, the system may only use artifacts from tasks before `t`.

The primary question is whether the frozen learned harness improves performance over the whole sequence as experience accumulates.

Primary metrics:

- cumulative success over the sequence;
- cumulative or mean reward over the sequence;
- total dollar cost per successful task;
- total tokens per successful task.

Secondary metrics can show how the benefit appears over time:

- warm-up efficiency: how quickly the system starts succeeding after a few seed tasks;
- cumulative success or reward curve over task index;
- negative transfer rate: how often diffused artifacts hurt performance compared with a no-diffusion baseline.

All methods should be evaluated on the same task sequence order. This keeps the comparison focused on the graph and diffusion harness rather than on differences in task sampling.

### 3.2 Train, validation, and test split

A natural split is 60/20/20 over the task set:

- 60% training tasks for learning the harness;
- 20% validation tasks for selecting or tuning the harness;
- 20% test tasks for final evaluation.

The basic training unit is a task sequence, not a single task. For example, each learning epoch can sample a random sequence of 10 tasks from the training split. Running one sequence gives one learning signal for the offline heuristic-learning agent.

At the start of each learning epoch, the runtime artifact bank should be cleared. This makes each sequence a fresh deployment episode. The offline heuristic-learning agent's learning history is not cleared; it accumulates evidence across epochs and uses the sequence-level feedback to revise the harness.

Within each sequence, only a small subset of tasks should be used as seed or warm-up tasks with no prior experience. After these seed tasks run, their artifacts enter the artifact bank. The later tasks in the same sequence can then receive diffused artifacts selected by the current harness.

The training loop is:

```text
sample a random training task sequence
-> clear the runtime artifact bank
-> run a small seed subset with no prior experience
-> run later tasks with artifact diffusion enabled
-> collect sequence-level feedback
-> offline heuristic-learning agent updates the harness
```

Validation and test use the same sequence format, but the harness is frozen. The artifact bank is still cleared at the start of each sequence and grows within the sequence, but the harness is not updated. All compared methods should use the same sampled task sequences and task order.

### 3.3 Practical implementation path for the offline HL agent

For the first implementation, the offline heuristic-learning agent does not need to be hardcoded as a new runtime component. We can use Codex locally as the offline HL agent: it reads the training sequence results, proposes changes to the graph and diffusion harness, and updates the relevant configuration, prompts, or policy code.

This is reasonable for early development because the offline HL agent is not part of the deployment-time task loop. It only runs between learning epochs or after batches of training sequences. If this local Codex-driven workflow produces useful harness updates, we can later package it into a more reproducible form.

The staged path is:

```text
local Codex offline HL loop
-> validate that harness updates improve sequence-level metrics
-> stabilize the workflow
-> package it into a reproducible workflow
```
