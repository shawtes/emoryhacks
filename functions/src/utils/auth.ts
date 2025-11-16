// functions/src/utils/auth.ts
import * as functions from "firebase-functions";
import type {DecodedIdToken} from "firebase-admin/auth";
import {ROLES} from "../config";

export const assertAuthenticated = (token?: DecodedIdToken): DecodedIdToken => {
  if (!token) {
    throw new functions.https.HttpsError(
      "unauthenticated",
      "Authentication required",
    );
  }
  return token;
};

export const assertDoctor = (token?: DecodedIdToken) => {
  const ensuredToken = assertAuthenticated(token);
  if (ensuredToken.role !== ROLES.doctor) {
    throw new functions.https.HttpsError(
      "permission-denied",
      "Doctor role required",
    );
  }
  return ensuredToken;
};

export const assertPatient = (token?: DecodedIdToken) => {
  const ensuredToken = assertAuthenticated(token);
  if (ensuredToken.role !== ROLES.patient) {
    throw new functions.https.HttpsError(
      "permission-denied",
      "Patient role required",
    );
  }
  return ensuredToken;
};
