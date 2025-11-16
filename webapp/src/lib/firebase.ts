// webapp/src/lib/firebase.ts
import type { FirebaseApp, FirebaseOptions } from 'firebase/app';
import { initializeApp } from 'firebase/app';
import type { Analytics } from 'firebase/analytics';
import { getAnalytics, isSupported } from 'firebase/analytics';

export type Persona = 'doctor' | 'patient';

const sharedConfig = {
  projectId: import.meta.env.VITE_FIREBASE_PROJECT_ID,
  storageBucket: import.meta.env.VITE_FIREBASE_STORAGE_BUCKET,
  messagingSenderId: import.meta.env.VITE_FIREBASE_MESSAGING_SENDER_ID,
};

const firebaseConfigs: Record<Persona, FirebaseOptions> = {
  doctor: {
    apiKey: import.meta.env.VITE_FIREBASE_API_KEY_DOCTOR,
    authDomain: import.meta.env.VITE_FIREBASE_AUTH_DOMAIN_DOCTOR,
    appId: import.meta.env.VITE_FIREBASE_APP_ID_DOCTOR,
    measurementId: import.meta.env.VITE_FIREBASE_MEASUREMENT_ID_DOCTOR,
    ...sharedConfig,
  },
  patient: {
    apiKey: import.meta.env.VITE_FIREBASE_API_KEY_PATIENT,
    authDomain: import.meta.env.VITE_FIREBASE_AUTH_DOMAIN_PATIENT,
    appId: import.meta.env.VITE_FIREBASE_APP_ID_PATIENT,
    measurementId: import.meta.env.VITE_FIREBASE_MEASUREMENT_ID_PATIENT,
    ...sharedConfig,
  },
};

const apps: Partial<Record<Persona, FirebaseApp>> = {};
const analyticsCache: Partial<Record<Persona, Promise<Analytics | null>>> = {};

export const getFirebaseApp = (persona: Persona) => {
  if (!apps[persona]) {
    apps[persona] = initializeApp(
      firebaseConfigs[persona],
      `voicevitals-${persona}`,
    );
  }
  return apps[persona]!;
};

export const getFirebaseAnalytics = (persona: Persona) => {
  if (!analyticsCache[persona]) {
    analyticsCache[persona] = isSupported()
      .then((supported) =>
        supported ? getAnalytics(getFirebaseApp(persona)) : null,
      )
      .catch(() => null);
  }
  return analyticsCache[persona]!;
};

export const getFirebaseClients = async (persona: Persona) => ({
  app: getFirebaseApp(persona),
  analytics: await getFirebaseAnalytics(persona),
});