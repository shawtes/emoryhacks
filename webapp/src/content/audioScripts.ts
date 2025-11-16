export interface ScriptSegment {
  id: string
  durationMs: number
  text: string
}

export interface AssessmentScript {
  locale: string
  totalDurationMs: number
  segments: ScriptSegment[]
}

const baseSegments: ScriptSegment[] = [
  {
    id: 'intro',
    durationMs: 9000,
    text: 'Today I feel focused and ready to collaborate. My morning started with a brisk walk, followed by a balanced breakfast of oatmeal, berries, and tea.',
  },
  {
    id: 'breath',
    durationMs: 9000,
    text: 'As I speak, I notice how evenly I’m breathing and how each sentence flows from the last. I’m thinking about the people I appreciate and the work we can do together to stay healthy.',
  },
  {
    id: 'environment',
    durationMs: 9000,
    text: 'The room around me is calm, with soft light and a gentle hum from the vents. I’m grateful for the chance to check in and describe how I’m doing.',
  },
  {
    id: 'reflection',
    durationMs: 9000,
    text: 'I’m paying attention to the pace of my words and the tone of my voice. I want everything I share to sound honest, steady, and hopeful.',
  },
  {
    id: 'closing',
    durationMs: 9000,
    text: 'I’ll finish by saying that I’m hopeful, curious, and paying close attention. Thank you for listening and helping me track my health.',
  },
]

const scripts: Record<string, AssessmentScript> = {
  'en-US': {
    locale: 'en-US',
    totalDurationMs: baseSegments.reduce((acc, segment) => acc + segment.durationMs, 0),
    segments: baseSegments,
  },
}

export function getAssessmentScript(locale: string = 'en-US'): AssessmentScript {
  return scripts[locale] ?? scripts['en-US']
}

export function registerScript(locale: string, script: AssessmentScript) {
  scripts[locale] = script
}


