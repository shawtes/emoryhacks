import { lazy } from 'react'

const routeLoaders = {
  Home: () => import('../pages/Home'),
  DoctorLogin: () => import('../pages/Login'),
  DoctorSignup: () => import('../pages/Signup'),
  PatientLogin: () => import('../pages/PatientLogin'),
  PatientSignup: () => import('../pages/PatientSignup'),
  PatientAssessment: () => import('../pages/PatientAssessment'),
  DoctorDashboard: () => import('../pages/DoctorDashboard'),
  PatientDashboard: () => import('../pages/PatientDashboard'),
  TechStack: () => import('../pages/TechStack'),
}

export type PrefetchableRoute = keyof typeof routeLoaders

export const HomePage = lazy(routeLoaders.Home)
export const DoctorLoginPage = lazy(routeLoaders.DoctorLogin)
export const DoctorSignupPage = lazy(routeLoaders.DoctorSignup)
export const PatientLoginPage = lazy(routeLoaders.PatientLogin)
export const PatientSignupPage = lazy(routeLoaders.PatientSignup)
export const PatientAssessmentPage = lazy(routeLoaders.PatientAssessment)
export const DoctorDashboardPage = lazy(routeLoaders.DoctorDashboard)
export const PatientDashboardPage = lazy(routeLoaders.PatientDashboard)
export const TechStackPage = lazy(routeLoaders.TechStack)

export function prefetchRoute(route: PrefetchableRoute) {
  void routeLoaders[route]?.()
}

const handlerCache = new Map<PrefetchableRoute, () => void>()
const eventCache = new Map<
  PrefetchableRoute,
  {
    onMouseEnter: () => void
    onFocus: () => void
    onTouchStart: () => void
  }
>()

export function getPrefetchHandler(route: PrefetchableRoute) {
  if (!handlerCache.has(route)) {
    handlerCache.set(route, () => {
      prefetchRoute(route)
    })
  }

  return handlerCache.get(route)!
}

export function getPrefetchEvents(route: PrefetchableRoute) {
  if (!eventCache.has(route)) {
    const handler = getPrefetchHandler(route)
    eventCache.set(route, {
      onMouseEnter: handler,
      onFocus: handler,
      onTouchStart: handler,
    })
  }

  return eventCache.get(route)!
}


