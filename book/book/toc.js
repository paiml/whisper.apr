// Populate the sidebar
//
// This is a script, and not included directly in the page, to control the total size of the book.
// The TOC contains an entry for each page, so if each page includes a copy of the TOC,
// the total size of the page becomes O(n**2).
class MDBookSidebarScrollbox extends HTMLElement {
    constructor() {
        super();
    }
    connectedCallback() {
        this.innerHTML = '<ol class="chapter"><li class="chapter-item expanded affix "><a href="introduction.html">Introduction</a></li><li class="chapter-item expanded affix "><li class="part-title">Getting Started</li><li class="chapter-item expanded "><a href="getting-started/installation.html"><strong aria-hidden="true">1.</strong> Installation</a></li><li class="chapter-item expanded "><a href="getting-started/quick-start.html"><strong aria-hidden="true">2.</strong> Quick Start</a></li><li class="chapter-item expanded "><a href="getting-started/browser-integration.html"><strong aria-hidden="true">3.</strong> Browser Integration</a></li><li class="chapter-item expanded "><a href="getting-started/core-concepts.html"><strong aria-hidden="true">4.</strong> Core Concepts</a></li><li class="chapter-item expanded affix "><li class="part-title">Architecture</li><li class="chapter-item expanded "><a href="architecture/overview.html"><strong aria-hidden="true">5.</strong> Overview</a></li><li class="chapter-item expanded "><a href="architecture/wasm-first.html"><strong aria-hidden="true">6.</strong> WASM-First Design</a></li><li class="chapter-item expanded "><a href="architecture/audio-pipeline.html"><strong aria-hidden="true">7.</strong> Audio Pipeline</a></li><li class="chapter-item expanded "><a href="architecture/transformer.html"><strong aria-hidden="true">8.</strong> Transformer Architecture</a></li><li><ol class="section"><li class="chapter-item expanded "><a href="architecture/encoder.html"><strong aria-hidden="true">8.1.</strong> Encoder</a></li><li class="chapter-item expanded "><a href="architecture/decoder.html"><strong aria-hidden="true">8.2.</strong> Decoder</a></li><li class="chapter-item expanded "><a href="architecture/attention.html"><strong aria-hidden="true">8.3.</strong> Multi-Head Attention</a></li></ol></li><li class="chapter-item expanded "><a href="architecture/apr-format.html"><strong aria-hidden="true">9.</strong> .apr Model Format</a></li><li class="chapter-item expanded "><a href="architecture/quantization.html"><strong aria-hidden="true">10.</strong> Quantization</a></li><li class="chapter-item expanded "><a href="architecture/trueno-integration.html"><strong aria-hidden="true">11.</strong> Trueno Integration</a></li><li class="chapter-item expanded affix "><li class="part-title">API Reference</li><li class="chapter-item expanded "><a href="api-reference/whisper-apr.html"><strong aria-hidden="true">12.</strong> WhisperApr</a></li><li class="chapter-item expanded "><a href="api-reference/transcribe-options.html"><strong aria-hidden="true">13.</strong> TranscribeOptions</a></li><li class="chapter-item expanded "><a href="api-reference/transcription-result.html"><strong aria-hidden="true">14.</strong> TranscriptionResult</a></li><li class="chapter-item expanded "><a href="api-reference/decoding-strategy.html"><strong aria-hidden="true">15.</strong> DecodingStrategy</a></li><li class="chapter-item expanded "><a href="api-reference/audio-processing.html"><strong aria-hidden="true">16.</strong> Audio Processing</a></li><li class="chapter-item expanded "><a href="api-reference/error-handling.html"><strong aria-hidden="true">17.</strong> Error Handling</a></li><li class="chapter-item expanded "><a href="api-reference/javascript-bindings.html"><strong aria-hidden="true">18.</strong> JavaScript Bindings</a></li><li class="chapter-item expanded affix "><li class="part-title">Performance</li><li class="chapter-item expanded "><a href="performance/benchmarks.html"><strong aria-hidden="true">19.</strong> Benchmarks Overview</a></li><li class="chapter-item expanded "><a href="performance/rtf-analysis.html"><strong aria-hidden="true">20.</strong> Real-Time Factor Analysis</a></li><li class="chapter-item expanded "><a href="performance/wasm-simd.html"><strong aria-hidden="true">21.</strong> WASM SIMD Performance</a></li><li class="chapter-item expanded "><a href="performance/memory.html"><strong aria-hidden="true">22.</strong> Memory Optimization</a></li><li class="chapter-item expanded "><a href="performance/quantization-impact.html"><strong aria-hidden="true">23.</strong> Model Quantization Impact</a></li><li class="chapter-item expanded "><a href="performance/browser-comparison.html"><strong aria-hidden="true">24.</strong> Browser Comparison</a></li><li class="chapter-item expanded "><a href="performance/renacer-profiling.html"><strong aria-hidden="true">25.</strong> Profiling with Renacer</a></li><li class="chapter-item expanded affix "><li class="part-title">Examples</li><li class="chapter-item expanded "><a href="examples/basic-transcription.html"><strong aria-hidden="true">26.</strong> Basic Transcription</a></li><li class="chapter-item expanded "><a href="examples/real-time-microphone.html"><strong aria-hidden="true">27.</strong> Real-Time Microphone</a></li><li class="chapter-item expanded "><a href="examples/file-upload.html"><strong aria-hidden="true">28.</strong> File Upload</a></li><li class="chapter-item expanded "><a href="examples/language-detection.html"><strong aria-hidden="true">29.</strong> Language Detection</a></li><li class="chapter-item expanded "><a href="examples/translation.html"><strong aria-hidden="true">30.</strong> Translation</a></li><li class="chapter-item expanded "><a href="examples/web-workers.html"><strong aria-hidden="true">31.</strong> Web Worker Integration</a></li><li class="chapter-item expanded "><a href="examples/react-integration.html"><strong aria-hidden="true">32.</strong> React Integration</a></li><li class="chapter-item expanded "><a href="examples/pwa.html"><strong aria-hidden="true">33.</strong> Progressive Web App</a></li><li class="chapter-item expanded affix "><li class="part-title">Development Guide</li><li class="chapter-item expanded "><a href="development/contributing.html"><strong aria-hidden="true">34.</strong> Contributing</a></li><li class="chapter-item expanded "><a href="development/extreme-tdd.html"><strong aria-hidden="true">35.</strong> Extreme TDD</a></li><li class="chapter-item expanded "><a href="development/testing.html"><strong aria-hidden="true">36.</strong> Testing</a></li><li><ol class="section"><li class="chapter-item expanded "><a href="development/unit-tests.html"><strong aria-hidden="true">36.1.</strong> Unit Tests</a></li><li class="chapter-item expanded "><a href="development/property-based-tests.html"><strong aria-hidden="true">36.2.</strong> Property-Based Tests</a></li><li class="chapter-item expanded "><a href="development/wasm-tests.html"><strong aria-hidden="true">36.3.</strong> WASM Tests</a></li><li class="chapter-item expanded "><a href="development/wer-validation.html"><strong aria-hidden="true">36.4.</strong> WER Validation</a></li></ol></li><li class="chapter-item expanded "><a href="development/benchmarking.html"><strong aria-hidden="true">37.</strong> Benchmarking</a></li><li class="chapter-item expanded "><a href="development/quality-gates.html"><strong aria-hidden="true">38.</strong> Quality Gates</a></li><li class="chapter-item expanded "><a href="development/pmat-integration.html"><strong aria-hidden="true">39.</strong> PMAT Integration</a></li><li class="chapter-item expanded affix "><li class="part-title">Advanced Topics</li><li class="chapter-item expanded "><a href="advanced/model-conversion.html"><strong aria-hidden="true">40.</strong> Custom Model Conversion</a></li><li class="chapter-item expanded "><a href="advanced/streaming.html"><strong aria-hidden="true">41.</strong> Streaming Inference</a></li><li class="chapter-item expanded "><a href="advanced/vad.html"><strong aria-hidden="true">42.</strong> Voice Activity Detection</a></li><li class="chapter-item expanded "><a href="advanced/webgpu.html"><strong aria-hidden="true">43.</strong> WebGPU Backend</a></li><li class="chapter-item expanded "><a href="advanced/server-side.html"><strong aria-hidden="true">44.</strong> Server-Side Deployment</a></li><li class="chapter-item expanded "><a href="advanced/edge-deployment.html"><strong aria-hidden="true">45.</strong> Edge Deployment</a></li><li class="chapter-item expanded "><a href="advanced/mobile-optimization.html"><strong aria-hidden="true">46.</strong> Optimizing for Mobile</a></li><li class="chapter-item expanded affix "><li class="part-title">Appendix</li><li class="chapter-item expanded "><a href="appendix/glossary.html"><strong aria-hidden="true">47.</strong> Glossary</a></li><li class="chapter-item expanded "><a href="appendix/references.html"><strong aria-hidden="true">48.</strong> References</a></li><li class="chapter-item expanded "><a href="appendix/faq.html"><strong aria-hidden="true">49.</strong> FAQ</a></li><li class="chapter-item expanded "><a href="appendix/changelog.html"><strong aria-hidden="true">50.</strong> Changelog</a></li><li class="chapter-item expanded "><a href="appendix/model-comparison.html"><strong aria-hidden="true">51.</strong> Whisper Model Comparison</a></li><li class="chapter-item expanded "><a href="appendix/browser-compatibility.html"><strong aria-hidden="true">52.</strong> Browser Compatibility</a></li></ol>';
        // Set the current, active page, and reveal it if it's hidden
        let current_page = document.location.href.toString();
        if (current_page.endsWith("/")) {
            current_page += "index.html";
        }
        var links = Array.prototype.slice.call(this.querySelectorAll("a"));
        var l = links.length;
        for (var i = 0; i < l; ++i) {
            var link = links[i];
            var href = link.getAttribute("href");
            if (href && !href.startsWith("#") && !/^(?:[a-z+]+:)?\/\//.test(href)) {
                link.href = path_to_root + href;
            }
            // The "index" page is supposed to alias the first chapter in the book.
            if (link.href === current_page || (i === 0 && path_to_root === "" && current_page.endsWith("/index.html"))) {
                link.classList.add("active");
                var parent = link.parentElement;
                if (parent && parent.classList.contains("chapter-item")) {
                    parent.classList.add("expanded");
                }
                while (parent) {
                    if (parent.tagName === "LI" && parent.previousElementSibling) {
                        if (parent.previousElementSibling.classList.contains("chapter-item")) {
                            parent.previousElementSibling.classList.add("expanded");
                        }
                    }
                    parent = parent.parentElement;
                }
            }
        }
        // Track and set sidebar scroll position
        this.addEventListener('click', function(e) {
            if (e.target.tagName === 'A') {
                sessionStorage.setItem('sidebar-scroll', this.scrollTop);
            }
        }, { passive: true });
        var sidebarScrollTop = sessionStorage.getItem('sidebar-scroll');
        sessionStorage.removeItem('sidebar-scroll');
        if (sidebarScrollTop) {
            // preserve sidebar scroll position when navigating via links within sidebar
            this.scrollTop = sidebarScrollTop;
        } else {
            // scroll sidebar to current active section when navigating via "next/previous chapter" buttons
            var activeSection = document.querySelector('#sidebar .active');
            if (activeSection) {
                activeSection.scrollIntoView({ block: 'center' });
            }
        }
        // Toggle buttons
        var sidebarAnchorToggles = document.querySelectorAll('#sidebar a.toggle');
        function toggleSection(ev) {
            ev.currentTarget.parentElement.classList.toggle('expanded');
        }
        Array.from(sidebarAnchorToggles).forEach(function (el) {
            el.addEventListener('click', toggleSection);
        });
    }
}
window.customElements.define("mdbook-sidebar-scrollbox", MDBookSidebarScrollbox);
