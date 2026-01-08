# Brick Architecture: Tests ARE the Interface (v2.0 Proposed)

**Status:** Proposed Implementation (WAPR-ARCH-BRICK-V2)
**Paradigm:** Inversion of Control via Generative Testing
**Citations:** Popper, Lakatos, Beizer, Leveson, Liskov, Hoare, Parnas, Wadler

## 1. Core Concept
Instead of writing UI code and then writing tests to verify it, we define **Bricks**. A Brick is a self-contained unit of specification that *generates* both the implementation (HTML/JS/CSS) and the verification logic (Tests).

## 2. Proposed Implementation Structure

The architecture requires significant updates to `probar` (Testing/Architect) and `presentar` (UI Runtime).

### 2.1 Probar (The Architect)
*Responsibility: Definition, Validation, Generation*

```text
probar/
├── src/
│   ├── brick/              # CORE DEFINITIONS
│   │   ├── mod.rs          # Brick trait definition
│   │   ├── element.rs      # ElementSpec (DOM existence)
│   │   ├── aria.rs         # AriaSpec (Accessibility)
│   │   ├── transition.rs   # TransitionSpec (State Machine Edges)
│   │   ├── interaction.rs  # InteractionSpec (Events)
│   │   ├── visual.rs       # VisualSpec (CSS/Layout)
│   │   ├── worker.rs       # WorkerSpec (WASM Interop)
│   │   ├── trace.rs        # TraceSpec (Dapper/OpenTelemetry)
│   │   └── model.rs        # ModelSpec (Data Schema)
│   ├── architect/          # CODE GENERATION
│   │   ├── mod.rs          # Coordinator
│   │   ├── html_gen.rs     # ElementSpec → HTML
│   │   ├── css_gen.rs      # VisualSpec → CSS
│   │   ├── js_gen.rs       # InteractionSpec → JS (Glue)
│   │   ├── worker_gen.rs   # WorkerSpec → worker.js
│   │   ├── prs_gen.rs      # Bricks → .prs manifest
│   │   └── state_machine.rs # TransitionSpec → Rust FSM
│   ├── yuan_gate.rs        # Zero-swallow error policy enforcement
│   ├── macros.rs           # #[brick], #[yuan_gate] exports
│   └── lib.rs
└── probador-macros/        # PROC MACROS
    └── src/lib.rs          # AST Parsing for #[brick]
```

### 2.2 Presentar (The Runtime)
*Responsibility: Rendering, Runtime Assertion, Jidoka*

```text
presentar/
├── src/
│   ├── widget/             # WIDGET IMPLEMENTATIONS
│   │   ├── mod.rs          # Trait Bound: Widget : Brick
│   │   ├── button.rs       # impl ButtonBrick
│   │   ├── text_display.rs # impl TextBrick
│   │   ├── progress_bar.rs # impl ProgressBrick
│   │   ├── audio_recorder.rs # impl AudioBrick
│   │   └── chart.rs        # impl ChartBrick
│   ├── runtime/            # RUNTIME SAFETY
│   │   ├── assertion_validator.rs # Pre-render checks
│   │   ├── jidoka.rs       # Halt execution on assertion fail
│   │   └── render_gate.rs  # Block render if brick missing
│   ├── prs/                # SCHEMA
│   │   ├── schema_v2.rs    # .prs v2.0 (with assertions)
│   │   ├── parser.rs
│   │   └── validator.rs    # Runtime assertion executor
│   └── lib.rs
```

## 3. The Popperian Falsification Checklist (100 Points)

Implementation is verified via 6 categories of falsifiable hypotheses.

### Category A: Compile Time (`tests/falsification/compile_time.rs`)
*Hypothesis: "Invalid bricks cannot compile."*
- [ ] A1. Brick without `generates` attribute fails compilation.
- [ ] A2. Transition referencing undefined State fails compilation.
- [ ] A3. Interaction without defined Handler fails compilation.
- [ ] A4. `ElementSpec` with invalid ARIA role fails compilation.

### Category B: Runtime Safety (`tests/falsification/runtime.rs`)
*Hypothesis: "Invalid states cannot be represented at runtime."*
- [ ] B1. `render_gate` prevents rendering undefined components.
- [ ] B2. `jidoka` halts execution immediately upon assertion failure.
- [ ] B3. `yuan_gate` catches swallowed exceptions in generated JS.

### Category C: Code Generation (`tests/falsification/codegen.rs`)
*Hypothesis: "Generated artifacts strictly adhere to specs."*
- [ ] C1. Generated HTML IDs match `ElementSpec` exactly.
- [ ] C2. Generated CSS strictly implements `VisualSpec`.
- [ ] C3. Generated Worker JS implements the defined FSM.

### Category D: Integration (`tests/falsification/presentar_integration.rs`)
*Hypothesis: "Presentar correctly interprets Probar bricks."*
- [ ] D1. `presentar` widgets accept `probar` brick definitions.
- [ ] D2. Runtime assertions in `prs` schema are executed.

### Category E: Tracing (`tests/falsification/tracing.rs`)
*Hypothesis: "All events are causally linked."*
- [ ] E1. Distributed trace context propagates across Worker boundary.
- [ ] E2. Broken causal chains trigger falsification.

## 4. Why Bugs Become Impossible

| Bug Class        | Traditional Cause                            | Brick Solution                               |
|------------------|----------------------------------------------|----------------------------------------------|
| **Missing element** | Typo in HTML ID or missing tag               | No `#[brick(generates)]` = doesn't exist     |
| **Wrong ARIA**      | Developer forgot attribute                   | Defined in brick, enforced at generation     |
| **State mismatch**  | UI/Worker synchronization drift              | State machine derived from transition bricks |
| **Missing handler** | Event listener not attached                  | Interaction brick generates handler          |
| **Race condition**  | Async events out of order                    | Transition ordering enforced by FSM          |
| **CSS invisibility**| Style conflict hides element                 | Visual brick asserts visibility/layout       |

## 5. Build Flow

1.  `probador collect-bricks tests/ui_spec.rs` (Parse)
2.  `probador generate --output www-demo/` (Generate Artifacts)
3.  `probador test --headless` (Validate)