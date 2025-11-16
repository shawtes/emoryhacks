import * as functions from "firebase-functions";
import {admin, db} from "../firebaseAdmin";
import {COLLECTIONS} from "../config";
import {assertPatient} from "../utils/auth";

interface ValidateAudioMetadataPayload {
  patientId?: string;
  recordingId?: string;
  metadata?: {
    doctorId?: string | null;
    clinicId?: string | null;
    recordedAt?: string;
    durationSeconds?: number;
    notes?: string;
    analysisStatus?: string;
    storagePath?: string | null;
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

export const validateAudioMetadata =
  functions.https.onCall<ValidateAudioMetadataPayload>(async (request) => {
    const claims = assertPatient(request.auth?.token);
    const data = request.data;
    const patientId = assertString(
      data?.patientId ?? claims.patientId,
      "patientId",
    );
    const recordingId = assertString(data?.recordingId, "recordingId");

    if (claims.patientId !== patientId) {
      throw new functions.https.HttpsError(
        "permission-denied",
        "Patients may only validate their own recordings.",
      );
    }

    const patientSnap = await db
      .collection(COLLECTIONS.patients)
      .doc(patientId)
      .get();
    if (!patientSnap.exists) {
      throw new functions.https.HttpsError(
        "not-found",
        "Patient document not found.",
      );
    }
    const patient = patientSnap.data() ?? {};

    const metadata = {
      clinicId: patient.clinicId,
      doctorId: data?.metadata?.doctorId ?? patient.doctorId ?? null,
      storagePath: data?.metadata?.storagePath ?? null,
      recordedAt: data?.metadata?.recordedAt ?
        admin.firestore.Timestamp.fromDate(new Date(data.metadata.recordedAt)) :
        admin.firestore.FieldValue.serverTimestamp(),
      durationSeconds: data?.metadata?.durationSeconds ?? null,
      notes: data?.metadata?.notes ?? null,
      analysisStatus: data?.metadata?.analysisStatus ?? "pendingReview",
      validatedAt: admin.firestore.FieldValue.serverTimestamp(),
    };

    await db
      .collection(COLLECTIONS.patientAudio)
      .doc(patientId)
      .collection("recordings")
      .doc(recordingId)
      .set(metadata, {merge: true});

    return {recordingId, patientId};
  });
// functions/src/audio/validateAudioMetadata.ts

