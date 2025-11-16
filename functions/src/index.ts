// functions/src/index.ts
import {registerDoctor} from "./registerDoctor";
import {createPatientStub} from "./createPatientStub";
import {activatePatient} from "./activatePatient";
import {issueUploadToken} from "./audio/issueUploadToken";
import {validateAudioMetadata} from "./audio/validateAudioMetadata";

export {
  registerDoctor,
  createPatientStub,
  activatePatient,
  issueUploadToken,
  validateAudioMetadata,
};
