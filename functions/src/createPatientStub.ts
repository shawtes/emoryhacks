// functions/src/createPatientStub.ts
import {randomUUID} from "node:crypto";
import * as functions from "firebase-functions";
import {admin, db} from "./firebaseAdmin";
import {COLLECTIONS} from "./config";
import {assertDoctor} from "./utils/auth";

interface CreatePatientStubPayload {
  patientId?: string;
  patient?: Record<string, unknown>;
}

export const createPatientStub =
  functions.https.onCall<CreatePatientStubPayload>(async (request) => {
    const doctorClaims = assertDoctor(request.auth?.token);
    const doctorUid = request.auth?.uid;
    if (!doctorUid) {
      throw new functions.https.HttpsError(
        "unauthenticated",
        "Authentication info missing",
      );
    }

    const data = request.data;
    const patientId =
      typeof data?.patientId === "string" && data.patientId.trim() ?
        data.patientId.trim() :
        randomUUID();

    const patientPayload = {
      clinicId: doctorClaims.clinicId,
      doctorId: doctorUid,
      status: "pendingActivation",
      createdByDoctorUid: doctorUid,
      createdAt: admin.firestore.FieldValue.serverTimestamp(),
      ...(typeof data?.patient === "object" && data.patient !== null ?
        data.patient :
        {}),
    };

    await db
      .collection(COLLECTIONS.patients)
      .doc(patientId)
      .set(patientPayload, {merge: true});

    return {patientId};
  });

