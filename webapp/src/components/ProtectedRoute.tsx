import { Navigate } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'
import PageLoader from './PageLoader'
import type { Persona } from '../lib/firebase'

interface ProtectedRouteProps {
  children: React.ReactNode
  requiredRole?: Persona
}

export default function ProtectedRoute({ children, requiredRole }: ProtectedRouteProps) {
  const { user, persona, isAuthLoading } = useAuth()

  if (isAuthLoading) {
    return <PageLoader />
  }

  if (!user) {
    const redirectPath = requiredRole === 'patient' ? '/patient/login' : '/doctor/login'
    return <Navigate to={redirectPath} replace />
  }

  if (requiredRole && persona && persona !== requiredRole) {
    const fallback = persona === 'patient' ? '/patient/dashboard' : '/doctor/dashboard'
    return <Navigate to={fallback} replace />
  }

  return <>{children}</>
}



