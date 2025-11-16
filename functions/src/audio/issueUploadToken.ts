// functions/src/audio/issueUploadToken.ts
import {randomUUID} from "node:crypto";
import * as functions from "firebase-functions";
import {AUDIO_BUCKET_PREFIX} from "../config";
import {assertPatient} from "../utils/auth";

interface IssueUploadTokenPayload {
  recordingId?: string;
  doctorId?: string;
  mimeType?: string;
  expiresInSeconds?: number;
}

export const issueUploadToken = functions.https.onCall<IssueUploadTokenPayload>(
  async (request) => {
    const patientClaims = assertPatient(request.auth?.token);

    if (!patientClaims.patientId) {
      throw new functions.https.HttpsError(
        "failed-precondition",
        "Patient ID missing from authentication claims",
      );
    }

    const data = request.data;
    const recordingId =
      typeof data?.recordingId === "string" && data.recordingId.trim() ?
        data.recordingId.trim() :
        randomUUID();

    const mimeType =
      typeof data?.mimeType === "string" && data.mimeType.trim() ?
        data.mimeType.trim() :
        "audio/webm";

    const storagePath = [
      AUDIO_BUCKET_PREFIX,
      patientClaims.patientId,
      `${recordingId}.webm`,
    ].join("/");
    const expiresInSeconds =
      typeof data?.expiresInSeconds === "number" && data.expiresInSeconds > 0 ?
        Math.min(data.expiresInSeconds, 60 * 60) :
        15 * 60;
    return {
      recordingId,
      storagePath,
      uploadUrl: null,
      expiresInSeconds,
      metadata: {
        clinicId: patientClaims.clinicId,
        doctorId:
          typeof data?.doctorId === "string" && data.doctorId.trim() ?
            data.doctorId.trim() :
            patientClaims.doctorId ?? null,
        patientId: patientClaims.patientId,
        mimeType,
      },
    };
  },
);
// functions/src/audio/issueUploadToken.ts

