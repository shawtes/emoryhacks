// functions/src/firebaseAdmin.ts
import * as admin from "firebase-admin";

if (admin.apps.length === 0) {
  admin.initializeApp();
}

export const auth = admin.auth();
export const db = admin.firestore();
export const storage = admin.storage();

export {admin};


