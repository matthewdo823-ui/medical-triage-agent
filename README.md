# Medical Symptom Triage Agent

Medical Symptom Triage Agent is a multi-agent healthcare triage system built for the LA Hacks Fetch.ai Agentverse challenge. A user describes symptoms in natural language through ASI:One, the system classifies urgency, generates likely explanations, recommends an appropriate care pathway, and returns a structured, guarded response designed to be understandable under stress.

The real-world problem is triage delay. Many people do not know whether a symptom belongs in self-care, urgent care, the emergency room, or a 911 call, and that uncertainty creates dangerous delays for emergencies while also overloading healthcare systems with lower-acuity visits. Online searches are too slow for urgent issues, and modern llms respond with little empathy. This project focuses on helping users make better first-step decisions faster.

Agentverse is the right platform for this because the system benefits from discoverability, agent-to-agent interoperability, mailbox delivery, and optional monetization. A triage workflow is a natural fit for specialized cooperative agents, and Agentverse gives those agents a standardized way to be found, invoked, and eventually monetized through platform-native protocols.

## Overview

- User interface: ASI:One via Fetch.ai Chat Protocol
- Orchestrator model: Anthropic Claude Haiku `claude-haiku-4-5-20251001`
- Specialist model: Google Gemma 4 `gemma-4-27b-it`
- Framework: `uagents`
- Safety layer: fast-path regex emergency detection plus deterministic guardrails
- Optional monetization: Payment Protocol gate for premium/full report delivery

## Architecture

```text
User (ASI:One)
      |
      v
Chat Protocol
      |
      v
Orchestrator Agent (Claude Haiku)
      |
      +--> Symptom Classifier Agent (Gemma 4)
      |
      +--> Knowledge Retrieval Agent (Gemma 4)
      |
      +--> Care Router Agent (Gemma 4)
      |
      v
Guardrails Pipeline
      |
      v
Response Synthesis (Claude Haiku)
      |
      v
Payment Protocol (optional gate)
      |
      v
User (ASI:One)
```

## Model Architecture Rationale

This project uses a dual-model design because the two hardest parts of medical triage are different kinds of work.

Claude Haiku handles the user-facing synthesis layer. That final step needs calm tone, readable structure, careful phrasing, and strong formatting discipline. Claude is used only after the specialist pipeline has already produced structured triage inputs, so it focuses on turning machine-readable signals into a clear explanation for the user.

Gemma 4 (`gemma-4-27b-it`) handles the three specialist tasks: symptom classification, knowledge retrieval, and care routing. Those tasks benefit from a fast, capable open model with native JSON mode and strong structured inference behavior. The pipeline leans on Gemma for repeatable schema-shaped outputs rather than long free-form prose.

Using two providers also demonstrates architectural flexibility and production-minded design. In real systems, the best model for structured routing is not always the best model for user communication, and this design reflects that separation clearly.

## Safety Design Decisions

### 1. Fast-path emergency detection

The orchestrator runs deterministic regex-based emergency detection before any LLM call. This is used for life-threatening patterns such as stroke, cardiac emergency, anaphylaxis, overdose, pediatric critical events, and obstetric emergencies.

Rationale: LLM inference can take seconds. A user describing a stroke or cardiac event should receive immediate emergency instructions without waiting for model inference.

### 2. Conservative severity defaults on error

If the classifier fails, the system defaults to severity `3` instead of a low-acuity response. If the router fails, it defaults to urgent care evaluation.

Rationale: It is safer to over-triage than under-triage. Sending a non-emergency patient to urgent care is inconvenient; sending an emergency home is dangerous.

### 3. Output guardrails pipeline

All final `TriageResponse` objects are passed through deterministic guardrails after model output and before the user sees anything.

Rationale: LLMs can hallucinate, drift from schema intent, soften an emergency, or omit required warnings. Safety-critical output needs code-level validation, not prompt-only control.

### 4. Hedged diagnostic language enforcement

Guardrails rewrite overly definitive phrasing such as "you have X" into hedged language such as "your symptoms may be consistent with X."

Rationale: Definitive diagnosis without examination, testing, or clinician review is unsafe and may create legal and clinical risk.

### 5. Warning signs are mandatory for every pathway

Even self-care and monitoring pathways must include escalation symptoms that tell the user when to seek emergency care.

Rationale: Mild presentations can worsen. Users should never leave the workflow without knowing what changes would make the situation urgent.

### 6. Session logging without raw PII

The local SQLite session log stores a hash of the input plus outcome metadata such as severity, pathway, processing time, safety flags, and whether fast-path emergency handling was used. It does not store raw symptom text.

Rationale: This follows HIPAA-adjacent best practices. Operational analytics matter, but raw symptom narratives are more sensitive than the system needs for debugging and performance review.

### 7. Mental health crisis routing to 988

When suicidal ideation or self-harm flags are present, the system injects 988 Suicide & Crisis Lifeline guidance and escalates the response.

Rationale: 988 is the appropriate crisis resource for many mental health emergencies. Routing every such situation directly to 911 can create harmful unintended consequences.

## Fetch.ai Integration Points

### Chat Protocol

The orchestrator implements Fetch.ai chat compatibility using `uagents` plus `uagents_core.contrib.protocols.chat`. It accepts `ChatMessage`, sends `ChatAcknowledgement` immediately to avoid ASI:One timeouts, and returns a `ChatMessage` response with `TextContent` plus `EndSessionContent`.

### Agentverse Registration

The orchestrator is configured as a mailbox-enabled local agent with published agent details and README metadata so it can be indexed and discovered. It also includes best-effort Agentverse registration logic using `AGENTVERSE_API_KEY` and pushes discoverability keywords.

- Agent name: `medical-triage-agent`
- Description: optimized for symptom triage discovery in ASI:One
- Tags/keywords: `medical`, `health`, `triage`, `symptoms`, `healthcare`, `emergency`
- Agent address: generated from the configured seed phrase at runtime

### Payment Protocol

Payment Protocol support is optional and disabled by default. When enabled, it can gate delivery of the full markdown triage report while still preserving the rest of the agent workflow.

Why gate the full report: this demonstrates a realistic monetization path for premium triage output, richer explanations, or downstream referral/report products while keeping the core architecture Fetch-native.

## Local Setup

### 1. Create a virtual environment and install dependencies

```bash
cd medical-triage-agent
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure environment variables

Copy `.env.example` to `.env` and fill in the required values:

```bash
cp .env.example .env
```

Required keys:

- `ANTHROPIC_API_KEY`: used by Claude Haiku for final response synthesis
- `GOOGLE_API_KEY`: used by Gemma 4 for the specialist sub-agents
- `ORCHESTRATOR_SEED`
- `CLASSIFIER_SEED`
- `KNOWLEDGE_SEED`
- `ROUTER_SEED`

Recommended for Agentverse discoverability:

- `AGENTVERSE_API_KEY`
- `AGENTVERSE_BASE_URL=https://agentverse.ai`

Optional:

- `ENABLE_PAYMENT_PROTOCOL=false`
- `PAYMENT_AMOUNT_ATESTFET=10`
- `PUBLIC_AGENT_ENDPOINT=http://127.0.0.1:8000/submit`

### 3. Run tests

```bash
# Run all tests
pytest tests/ -v

# Run only safety tests first
pytest tests/test_safety.py -v
```

### 4. Run the orchestrator locally

```bash
python agents/orchestrator.py
```

### 5. Run specialist agents locally

Open separate terminals and run:

```bash
python agents/symptom_classifier.py
python agents/knowledge_retrieval.py
python agents/care_router.py
```

### 6. Test with ASI:One / Agentverse

Once the orchestrator is running with mailbox support and valid Agentverse credentials:

1. Start the orchestrator and let it publish its manifest.
2. Confirm the agent is reachable through Agentverse Inspector or mailbox flow.
3. Verify the registration metadata and keywords are visible on Agentverse.
4. Send a symptom description through an ASI:One-compatible chat session.
5. Confirm that the agent immediately acknowledges the message, then returns a triage response.

## Project Structure

```text
medical-triage-agent/
├── agents/
│   ├── orchestrator.py
│   ├── symptom_classifier.py
│   ├── knowledge_retrieval.py
│   └── care_router.py
├── models/
├── prompts/
├── safety/
├── tests/
├── utils/
├── .env.example
├── requirements.txt
└── README.md
```

## Known Limitations

- This project is not FDA-cleared, clinically validated, or HIPAA-compliant in its current form.
- It does not have access to physical exam findings, vital signs, lab work, imaging, or clinician judgment, which are often the most important parts of real triage.
- It depends on internet connectivity and external model APIs, so there is no offline operating mode.
- Optional web retrieval is intentionally time-bounded and limited; the current design prioritizes responsiveness over exhaustive literature review.
- The orchestrator currently contains direct helper-call pathways for specialist logic in addition to the standalone sub-agent implementations; full network-only delegation can be expanded further.

## Future Work

- EHR integration for richer clinical context
- Voice input for accessibility and hands-free symptom reporting
- Multilingual support
- Better pediatric and obstetric specialization
- More robust payment-gated premium workflows
- Human-in-the-loop escalation to telehealth or nurse triage services
- Outcome tracking and feedback loops for calibration improvement

## Submission Notes

This project is designed to demonstrate more than prompt chaining. It shows a production-minded multi-agent architecture with:

- Specialized agents with clear responsibilities
- Dual-model orchestration based on task fit
- Deterministic emergency short-circuiting
- Post-LLM safety validation
- Agentverse-native discoverability
- Optional monetization through Fetch.ai protocols

That combination is exactly why this problem belongs on Agentverse.
# medical-triage-agent
