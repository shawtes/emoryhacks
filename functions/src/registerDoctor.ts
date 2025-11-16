// functions/src/registerDoctor.ts
import {randomUUID} from "node:crypto";
import * as functions from "firebase-functions";
import {auth, db, admin} from "./firebaseAdmin";
import {COLLECTIONS, ROLES} from "./config";

interface RegisterDoctorPayload {
  clinicId?: string;
  email?: string;
  password?: string;
  profile?: {
    fullName?: string;
    specialties?: string[];
    phone?: string;
  };
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

export const registerDoctor = functions.https.onCall<RegisterDoctorPayload>(
  async (request) => {
    const data = request.data;
    const clinicId = assertString(data?.clinicId, "clinicId");
    const email = assertString(data?.email, "email");
    const password = assertString(data?.password, "password");

    const clinicSnap = await db
      .collection(COLLECTIONS.clinics)
      .doc(clinicId)
      .get();
    if (!clinicSnap.exists) {
      throw new functions.https.HttpsError("not-found", "Clinic ID not found");
    }
    const clinic = clinicSnap.data() ?? {};
    if (clinic.isActive === false) {
      throw new functions.https.HttpsError(
        "failed-precondition",
        "Clinic is not active. Please contact an administrator.",
      );
    }

    try {
      const userRecord = await auth.createUser({
        email,
        password,
        displayName: data?.profile?.fullName,
      });

      await auth.setCustomUserClaims(userRecord.uid, {
        role: ROLES.doctor,
        clinicId,
      });

      const doctorId = userRecord.uid ?? randomUUID();
      await db.collection(COLLECTIONS.doctors).doc(doctorId).set({
        authUid: userRecord.uid,
        clinicId,
        status: "active",
        profile: {
          fullName: data?.profile?.fullName ?? null,
          specialties: data?.profile?.specialties ?? [],
          phone: data?.profile?.phone ?? null,
        },
        createdAt: admin.firestore.FieldValue.serverTimestamp(),
        createdByClinicId: clinicId,
      });

      return {doctorId};
    } catch (error) {
      if (error instanceof functions.https.HttpsError) {
        throw error;
      }
      throw new functions.https.HttpsError(
        "internal",
        "Failed to register doctor account",
        (error as Error).message,
      );
    }
  },
);

