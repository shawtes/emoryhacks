import {
  createContext,
  useCallback,
  useContext,
  useMemo,
  useState,
  type ReactNode,
} from 'react'
import type { FirebaseApp } from 'firebase/app'
import type { Analytics } from 'firebase/analytics'
import { getFirebaseClients, type Persona } from '../lib/firebase'

interface FirebaseState {
  persona?: Persona
  app?: FirebaseApp
  analytics?: Analytics | null
}

interface FirebaseContextValue extends FirebaseState {
  isLoading: boolean
  error: Error | null
  selectPersona: (persona: Persona) => Promise<void>
}

const FirebaseContext = createContext<FirebaseContextValue | undefined>(
  undefined,
)

export function FirebaseProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<FirebaseState>({})
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<Error | null>(null)

  const selectPersona = useCallback(
    async (nextPersona: Persona) => {
      if (state.persona === nextPersona && state.app) {
        return
      }

      setIsLoading(true)
      setError(null)

      try {
        const clients = await getFirebaseClients(nextPersona)
        setState({
          persona: nextPersona,
          app: clients.app,
          analytics: clients.analytics ?? null,
        })
      } catch (err) {
        const normalizedError =
          err instanceof Error
            ? err
            : new Error('Failed to initialize Firebase for persona')
        setError(normalizedError)
        throw normalizedError
      } finally {
        setIsLoading(false)
      }
    },
    [state.app, state.persona],
  )

  const value = useMemo<FirebaseContextValue>(
    () => ({
      persona: state.persona,
      app: state.app,
      analytics: state.analytics,
      isLoading,
      error,
      selectPersona,
    }),
    [state, isLoading, error, selectPersona],
  )

  return (
    <FirebaseContext.Provider value={value}>
      {children}
    </FirebaseContext.Provider>
  )
}

export function useFirebase() {
  const context = useContext(FirebaseContext)
  if (!context) {
    throw new Error('useFirebase must be used within a FirebaseProvider')
  }
  return context
}


