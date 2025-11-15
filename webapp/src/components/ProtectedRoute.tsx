import { Navigate } from 'react-router-dom'

interface ProtectedRouteProps {
  children: React.ReactNode
}

// TODO: Replace with actual authentication check
// For now, this is a placeholder that always allows access
// In production, check if user is authenticated
export default function ProtectedRoute({ children }: ProtectedRouteProps) {
  // const isAuthenticated = checkAuth() // TODO: Implement auth check
  const isAuthenticated = true // Temporary: always allow for UI development

  if (!isAuthenticated) {
    return <Navigate to="/login" replace />
  }

  return <>{children}</>
}


