// functions/src/activatePatient.ts
import * as functions from "firebase-functions";
import {auth, db, admin} from "./firebaseAdmin";
import {COLLECTIONS, ROLES} from "./config";

interface ActivatePatientPayload {
  patientId?: string;
  email?: string;
  password?: string;
}

const assertString = (value: unknown, field: string) => {
  if (typeof value !== "string" || value.trim() === "") {
    throw new functions.https.HttpsError(
      "invalid-argument",
      `${field} is required and must be a non-empty string`,
    );
  }
  return value.trim();
};

export const activatePatient = functions.https.onCall<ActivatePatientPayload>(
  async (request) => {
    const data = request.data;
    const patientId = assertString(data?.patientId, "patientId");
    const email = assertString(data?.email, "email");
    const password = assertString(data?.password, "password");

    const patientRef = db.collection(COLLECTIONS.patients).doc(patientId);
    const patientSnap = await patientRef.get();

    if (!patientSnap.exists) {
      throw new functions.https.HttpsError(
        "not-found",
        "Patient ID not found. Please contact your doctor.",
      );
    }

    const patient = patientSnap.data() ?? {};
    if (patient.status !== "pendingActivation") {
      throw new functions.https.HttpsError(
        "failed-precondition",
        "Patient is already active or not eligible for activation.",
      );
    }

    try {
      const userRecord = await auth.createUser({
        email,
        password,
        displayName: patient.name ?? undefined,
      });

      await auth.setCustomUserClaims(userRecord.uid, {
        role: ROLES.patient,
        clinicId: patient.clinicId,
        patientId,
        doctorId: patient.doctorId ?? null,
      });

      await patientRef.update({
        authUid: userRecord.uid,
        email,
        status: "active",
        activatedAt: admin.firestore.FieldValue.serverTimestamp(),
      });

      return {patientId, authUid: userRecord.uid};
    } catch (error) {
      if (error instanceof functions.https.HttpsError) {
        throw error;
      }
      throw new functions.https.HttpsError(
        "internal",
        "Failed to activate patient account",
        (error as Error).message,
      );
    }
  },
);
// functions/src/activatePatient.ts

