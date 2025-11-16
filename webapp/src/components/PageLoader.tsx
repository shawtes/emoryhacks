export default function PageLoader() {
  return (
    <div className="page-loader" role="status" aria-live="polite">
      <span className="page-loader__spinner" />
      <span className="visually-hidden">Loading page…</span>
    </div>
  )
}


