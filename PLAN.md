# Webapp Performance Optimization Plan

## Objectives
- Improve initial render and interactive readiness for existing pages (especially `Home.tsx`) without altering layout or visual structure.
- Cut the largest JS bundle size by ≥30 % and deliver LCP ≤2.5 s on mid-tier devices.
- Preserve accessibility and existing UX flows while reducing CPU, memory, and network cost.

## KPIs & Monitoring
- Lighthouse Performance ≥90 (mobile + desktop), CLS <0.1, TBT <200 ms.
- Real-user metrics via `web-vitals` (LCP, FID/INP, CLS) streamed to analytics.
- Bundle analyzer report committed per release; WebPageTest synthetic run for `/`, `/auth/*`, `/patient/dashboard`.

## Constraints
- No layout, content, or navigation changes; performance-only modifications.
- Prefer native browser capabilities and existing stack (React + Vite/Next? adjust per repo) before adding libraries.
- Every optimization must include a measurable before/after metric.

## Workstreams

### 1. Bundle Health & Code Splitting
- Run `pnpm build --report` (or `npm run analyze`) to inventory chunk sizes; flag anything >200 KB gz.
- Split `Home.tsx` into route-level async chunks (hero, storyline, showcase) using React.lazy + Suspense; keep fold-critical logic synchronous.
- Eliminate unused polyfills, shared utils, and third-party packages; prefer platform APIs.
- Promote component-level memoization (`React.memo`, `useMemo`) for static props; ensure hooks dependencies are memo-safe to avoid re-renders.
- Enable modern build output (ESM, differential serving) so evergreen browsers skip legacy transforms.

### 2. Asset & Font Optimization
- Convert hero/section imagery to AVIF/WebP derivatives with responsive `<picture>` tags; keep legacy fallback.
- Introduce blur-up or solid-color placeholders for large assets; lazy-load anything below first viewport.
- Inline critical CSS and preload the primary font subset; defer remaining font weights, set `font-display: swap`.
- Audit SVGs used in `Home.tsx` and shared components; collapse inline duplicates and minify paths.

### 3. Network & Caching Strategy
- Ensure CDN caching headers for static assets (immutable hash + 1y TTL); enable brotli + HTTP/2.
- Implement service worker (or extend existing PWA layer) for offline cache of shell + fonts and `stale-while-revalidate` for API responses.
- Add `<link rel="preconnect">` for API + analytics origins; selectively `<link rel="prefetch">` route bundles when hovering nav items.
- Review API payloads for `Home.tsx` data (testimonials, stats) and introduce compression or trimming of unused fields.

### 4. Runtime Efficiency
- Replace synchronous data massaging in render with memoized selectors; move heavy transforms to build-time JSON.
- Virtualize long lists (e.g., testimonials, blog feeds) or limit DOM nodes via pagination.
- Debounce scroll/resize listeners; prefer `IntersectionObserver` for in-view checks.
- Audit context providers and prop drilling; split contexts or use selectors so updates don’t cascade through entire trees.

### 5. Data Fetching & State
- Adopt SWR/React Query (if not already) with caching, background refresh, and de-duped requests; otherwise implement lightweight cache hook.
- Coalesce parallel fetches for hero stats/trust badges via single batched endpoint.
- Ensure SSR/SSG pages serialize only what is needed; hydrate deferred data via `useEffect` to keep HTML lean.

### 6. Testing & Guardrails
- Add Lighthouse CI + Bundlewatch to PR pipeline; fail builds when thresholds regress.
- Track Web Vitals in production (Azul, Sentry, or custom endpoint) with alerts for percentile regressions.
- Document performance budgets per page (JS <170 KB gz, images <300 KB above the fold).

## Phased Rollout
1. **Audit & Instrumentation** – Capture current metrics, wire Lighthouse CI, land bundle analyzer tooling.
2. **Quick Wins** – Fonts, image compression, unused dependency removal, caching headers.
3. **Structural Optimizations** – Code splitting, lazy-loading, runtime memoization, API payload trimming.
4. **Hardening & Monitoring** – Add tests, regression alerts, and schedule monthly performance reviews.

## Success Criteria
- ≤2.5 s LCP for `Home.tsx` and ≤3.0 s for secondary routes on 4G.
- JS bundle reduction ≥30 %, image weight reduction ≥40 % above the fold.
- No increase in layout shifts or accessibility regressions (axe-core + manual spot checks).

