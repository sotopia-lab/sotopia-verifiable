# Training Pipeline & Ablation Study Diagrams

## Training Pipeline Architecture

```
                    SOTOPIA-VERIFIABLE TRAINING PIPELINE

┌─────────────────────┐                    ┌─────────────────────┐
│   TRAINEE MODEL     │                    │   PARTNER MODEL     │
│  (Qwen2.5-7B+LoRA) │◄──────────────────►│     (GPT-4o)        │
│   392M trainable    │    Strategic       │      Fixed          │
│    parameters       │   Interaction      │   No Learning       │
└─────────────────────┘                    └─────────────────────┘
           │                                          │
           │              ┌─────────────────────┐     │
           └──────────────►│  SOCIAL SCENARIO    │◄────┘
                          │  Verifiable Rules   │
                          │  Binary Win/Loss    │
                          └─────────────────────┘
                                     │
                                     ▼
                          ┌─────────────────────┐
                          │ GAME INTERACTION    │
                          │ ~6 turns average    │
                          │ Structured moves    │
                          └─────────────────────┘
                                     │
                ┌────────────────────┼────────────────────┐
                ▼                    ▼                    ▼
   ┌─────────────────────┐ ┌─────────────────────┐ ┌─────────────────────┐
   │ TRAINING DATA COLL  │ │  GRPO TRAINING      │ │  WANDB LOGGING      │
   │ Prompt-Response     │ │  16 generations     │ │  Experiment Track   │
   │ Binary Rewards      │ │  Policy Updates     │ │  Real-time Metrics  │
   └─────────────────────┘ └─────────────────────┘ └─────────────────────┘
                                     │
                                     │ Iterative
                                     │ Improvement
                                     ▼
                          ┌─────────────────────┐
                          │   UPDATED MODEL     │
                          │  Better Social AI   │
                          └─────────────────────┘

SCENARIO TAXONOMY INPUT:                 SUCCESS METRICS:
┌─────────────────────┐                 ┌─────────────────────┐
│ • Social Inter-     │                 │ • Win Rate vs       │
│   dependence Theory │                 │   Partner Models    │
│ • Relational Models │                 │ • Training          │
│   (Fiske 1990)      │                 │   Stability         │
│ • Resource Exchange │                 │ • Social            │
│   Theory            │                 │   Appropriateness   │
│ • 10 diverse        │                 │ • Behavioral        │
│   scenarios         │                 │   Diversity         │
└─────────────────────┘                 └─────────────────────┘
```

## Potential Ablation Studies

### 1. Reward Structure Ablation
```
Binary Rewards (Current)    Continuous Rewards      Hybrid Rewards
     ±1.0/0.0              Graduated 0.0-1.0       Game + Process
        │                        │                      │
        ▼                        ▼                      ▼
   ┌─────────┐              ┌─────────┐            ┌─────────┐
   │ Clean   │              │ Noisy   │            │ Complex │
   │ Signal  │              │ but     │            │ but     │
   │ Simple  │              │ Rich    │            │ Rich    │
   │ Stable  │              │ Info    │            │ Info    │
   └─────────┘              └─────────┘            └─────────┘

Expected: High Stability   Expected: High Info    Expected: Balanced
```

### 2. Partner Model Strength Ablation
```
GPT-4o (Current)        GPT-3.5-Turbo         Co-evolving Agent
Strong Strategic        Weaker Strategic       Adaptive Strategic
    Partner                Partner                 Partner
      │                     │                       │
      ▼                     ▼                       ▼
┌─────────────┐       ┌─────────────┐         ┌─────────────┐
│ High        │       │ Easier      │         │ Dynamic     │
│ Challenge   │       │ Training    │         │ Complexity  │
│ Robust      │       │ Faster      │         │ Curriculum  │
│ Learning    │       │ Convergence │         │ Learning    │
└─────────────┘       └─────────────┘         └─────────────┘

Research Question: Optimal partner strength for social skill development
```

### 3. Scenario Complexity Progression
```
Simple Binary Choices    Multi-step Negotiations    Open-ended Social
   "Choose A or B"         "Negotiate price X"        "Build relationship"
        │                        │                          │
        ▼                        ▼                          ▼
   ┌─────────┐              ┌─────────┐                ┌─────────┐
   │ Easy to │              │ Strategic│                │ Realistic│
   │ Verify  │              │ Depth    │                │ Complex  │
   │ Fast    │              │ Moderate │                │ Hard to  │
   │ Train   │              │ Verify   │                │ Verify   │
   └─────────┘              └─────────┘                └─────────┘

Curriculum: Simple → Strategic → Complex social interactions
```

### 4. Training Algorithm Comparison
```
     GRPO              PPO              DQN
(Group Reward)    (Proximal Policy)  (Deep Q-Network)
     │                   │                  │
     ▼                   ▼                  ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ Multi-agent │    │ Single-agent│    │ Value-based │
│ Reward      │    │ Policy      │    │ Discrete    │
│ Social      │    │ Gradient    │    │ Actions     │
│ Focused     │    │ General     │    │ Limited     │
└─────────────┘    └─────────────┘    └─────────────┘

Hypothesis: GRPO best for social multi-agent learning
```

## Training Data Flow Analysis

### Current Data Characteristics (200 examples)
```
Scenario Distribution:
┌─────────────────────────────────────────────────────────────────┐
│ Competitive (40%) ████████████████                              │
│ Cooperative (35%) ██████████████                                │
│ Mixed (25%)       ██████████                                    │
└─────────────────────────────────────────────────────────────────┘

Reward Distribution:
┌─────────────────────────────────────────────────────────────────┐
│ Win (+1.0):  47%  ███████████████████████                       │
│ Loss (-1.0): 35%  ██████████████████                            │
│ Draw (0.0):  18%  █████████                                     │
└─────────────────────────────────────────────────────────────────┘

Game Length Distribution:
┌─────────────────────────────────────────────────────────────────┐
│ 2-4 turns:   25% ████████████                                   │
│ 5-8 turns:   60% ████████████████████████████████               │
│ 9+ turns:    15% ███████                                        │
└─────────────────────────────────────────────────────────────────┘
```

### Scaling Analysis (Projected)
```
Current (200) → Target (500) → Future (2000)
     │               │              │
     ▼               ▼              ▼
┌─────────┐    ┌─────────┐    ┌─────────┐
│ Proof   │    │ Stable  │    │ Robust  │
│ of      │    │ Train   │    │ Social  │
│ Concept │    │ Dynamic │    │ AI      │
└─────────┘    └─────────┘    └─────────┘

Expected improvements:
• Better strategy diversity
• More stable convergence
• Stronger generalization
```

## Research Impact Projection

### Scientific Contributions Timeline
```
Month 1 (Current)      Month 2-3           Month 4-6
Infrastructure    →    Enhanced Data   →   Evaluation &
   Complete            Collection          Publication
      │                   │                   │
      ▼                   ▼                   ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ • Working   │    │ • 500+      │    │ • Human     │
│   Pipeline  │    │   Examples  │    │   Evaluation│
│ • Binary    │    │ • Better    │    │ • Benchmark │
│   Rewards   │    │   Scenarios │    │   Results   │
│ • LoRA      │    │ • Ablation  │    │ • Paper     │
│   Training  │    │   Studies   │    │   Draft     │
└─────────────┘    └─────────────┘    └─────────────┘

Target Venues: NeurIPS, ICLR, AAAI (Social AI Track)
```
