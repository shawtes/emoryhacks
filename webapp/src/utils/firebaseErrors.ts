import { FirebaseError } from 'firebase/app'

const errorMap: Record<string, string> = {
  'permission-denied': 'You do not have permission for that action. Please sign in as the correct persona.',
  'not-found': 'We could not find a matching record. Double-check the ID and try again.',
  'failed-precondition': 'This action cannot be completed yet. Please verify the status and try again.',
  'already-exists': 'An account with this email already exists.',
  'unauthenticated': 'Please sign in before trying again.',
  'unavailable': 'Service temporarily unavailable. Please retry shortly.',
}

export const getFirebaseErrorMessage = (error: unknown, fallback = 'Something went wrong. Please try again.') => {
  if (error instanceof FirebaseError) {
    return errorMap[error.code] ?? error.message ?? fallback
  }
  if (error instanceof Error) {
    return error.message
  }
  if (typeof error === 'string') {
    return error
  }
  return fallback
}

