import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from 'react'
import {
  getAuth,
  onAuthStateChanged,
  signInWithEmailAndPassword,
  signOut as firebaseSignOut,
  type User,
} from 'firebase/auth'
import { getIdTokenResult } from 'firebase/auth'
import { useFirebase } from './FirebaseContext'
import type { Persona } from '../lib/firebase'
import { getFirebaseApp } from '../lib/firebase'

const PERSONA_STORAGE_KEY = 'voicevital-active-persona'

interface AuthContextValue {
  user: User | null
  claims: Record<string, unknown> | null
  persona: Persona | null
  isAuthLoading: boolean
  signIn: (persona: Persona, email: string, password: string) => Promise<void>
  signOut: () => Promise<void>
  error: string | null
}

const AuthContext = createContext<AuthContextValue | undefined>(undefined)

export function AuthProvider({ children }: { children: ReactNode }) {
  const { persona: firebasePersona, selectPersona } = useFirebase()
  const [user, setUser] = useState<User | null>(null)
  const [claims, setClaims] = useState<Record<string, unknown> | null>(null)
  const [activePersona, setActivePersona] = useState<Persona | null>(null)
  const [isAuthLoading, setIsAuthLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const persistPersona = useCallback((personaValue: Persona | null) => {
    if (typeof window === 'undefined') {
      return
    }
    if (personaValue) {
      window.localStorage.setItem(PERSONA_STORAGE_KEY, personaValue)
    } else {
      window.localStorage.removeItem(PERSONA_STORAGE_KEY)
    }
  }, [])

  useEffect(() => {
    if (!firebasePersona) {
      if (typeof window === 'undefined') {
        return
      }
      const storedPersona = window.localStorage.getItem(PERSONA_STORAGE_KEY) as Persona | null
      if (storedPersona === 'doctor' || storedPersona === 'patient') {
        void selectPersona(storedPersona)
      }
    }
  }, [firebasePersona, selectPersona])

  useEffect(() => {
    if (!firebasePersona) {
      setIsAuthLoading(false)
      return
    }
    setIsAuthLoading(true)
    const app = getFirebaseApp(firebasePersona)
    const auth = getAuth(app)
    const unsubscribe = onAuthStateChanged(auth, async (currentUser) => {
      if (!currentUser) {
        setUser(null)
        setClaims(null)
        setActivePersona(null)
        setIsAuthLoading(false)
        return
      }
      const tokenResult = await getIdTokenResult(currentUser, true)
      const roleFromClaims = tokenResult.claims.role as Persona | undefined
      const resolvedPersona = roleFromClaims ?? firebasePersona
      setUser(currentUser)
      setClaims(tokenResult.claims)
      setActivePersona(resolvedPersona)
      persistPersona(resolvedPersona)
      setIsAuthLoading(false)
    })
    return () => unsubscribe()
  }, [firebasePersona, persistPersona])

  const handleSignIn = useCallback(
    async (persona: Persona, email: string, password: string) => {
      setIsAuthLoading(true)
      setError(null)
      try {
        await selectPersona(persona)
        const app = getFirebaseApp(persona)
        const auth = getAuth(app)
        const credential = await signInWithEmailAndPassword(auth, email, password)
      const tokenResult = await getIdTokenResult(credential.user, true)
      const roleFromClaims = (tokenResult.claims.role as Persona | undefined) ?? persona
      if (roleFromClaims && roleFromClaims !== persona) {
        const mismatchMessage =
          roleFromClaims === 'patient'
            ? 'This account is for patients. Please use the patient portal to sign in.'
            : 'This account is for doctors. Please use the doctor portal to sign in.'
        await firebaseSignOut(auth)
        setUser(null)
        setClaims(null)
        setActivePersona(null)
        persistPersona(null)
        throw new Error(mismatchMessage)
      }
      setUser(credential.user)
      setClaims(tokenResult.claims)
      setActivePersona(roleFromClaims)
      persistPersona(roleFromClaims)
      } catch (err) {
        const message =
          err instanceof Error ? err.message : 'Failed to sign in. Please try again.'
        setError(message)
        throw err
      } finally {
        setIsAuthLoading(false)
      }
    },
    [selectPersona],
  )

  const handleSignOut = useCallback(async () => {
    const personaToSignOut = activePersona ?? firebasePersona
    if (!personaToSignOut) {
      return
    }
    setIsAuthLoading(true)
    setError(null)
    try {
      const auth = getAuth(getFirebaseApp(personaToSignOut))
      await firebaseSignOut(auth)
      setUser(null)
      setClaims(null)
      setActivePersona(null)
      persistPersona(null)
    } catch (err) {
      const message =
        err instanceof Error ? err.message : 'Failed to sign out. Please try again.'
      setError(message)
      throw err
    } finally {
      setIsAuthLoading(false)
    }
  }, [activePersona, firebasePersona, persistPersona])

  const value = useMemo<AuthContextValue>(
    () => ({
      user,
      claims,
      persona: activePersona,
      isAuthLoading,
      signIn: handleSignIn,
      signOut: handleSignOut,
      error,
    }),
    [user, claims, activePersona, isAuthLoading, handleSignIn, handleSignOut, error],
  )

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}

export function useAuth() {
  const context = useContext(AuthContext)
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider')
  }
  return context
}

