import { useState, useRef, useEffect } from 'react'

export interface AudioReadyPayload {
  blob: Blob
  file: File
  url: string
  durationSeconds: number
}

interface AudioRecorderProps {
  onRecordingComplete: (audioFile: File) => void
  disabled?: boolean
  autoSubmit?: boolean
  showAnalyzeButton?: boolean
  showPreview?: boolean
  onAudioReady?: (payload: AudioReadyPayload) => void
  onRecordingStateChange?: (state: 'idle' | 'recording') => void
  onReset?: () => void
  onError?: (error: Error | string) => void
}

export default function AudioRecorder({
  onRecordingComplete,
  disabled,
  autoSubmit = true,
  showAnalyzeButton = true,
  showPreview = true,
  onAudioReady,
  onRecordingStateChange,
  onReset,
  onError,
}: AudioRecorderProps) {
  const [isRecording, setIsRecording] = useState(false)
  const [recordingTime, setRecordingTime] = useState(0)
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null)
  const [audioUrl, setAudioUrl] = useState<string | null>(null)

  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const chunksRef = useRef<Blob[]>([])
  const timerRef = useRef<number | null>(null)
  const recordingTimeRef = useRef(0)

  useEffect(() => {
    return () => {
      if (timerRef.current) {
        clearInterval(timerRef.current)
      }
      if (audioUrl) {
        URL.revokeObjectURL(audioUrl)
      }
    }
  }, [audioUrl])

  const cleanupStream = (stream: MediaStream) => {
    stream.getTracks().forEach(track => track.stop())
  }

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const mediaRecorder = new MediaRecorder(stream)
      mediaRecorderRef.current = mediaRecorder
      chunksRef.current = []
      recordingTimeRef.current = 0
      setRecordingTime(0)

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunksRef.current.push(event.data)
        }
      }

      mediaRecorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: 'audio/webm' })
        const fileName = `recording-${Date.now()}.webm`
        const file = new File([blob], fileName, { type: 'audio/webm' })
        const url = URL.createObjectURL(blob)

        setAudioBlob(blob)
        setAudioUrl((prev) => {
          if (prev) URL.revokeObjectURL(prev)
          return url
        })

        const payload: AudioReadyPayload = {
          blob,
          file,
          url,
          durationSeconds: recordingTimeRef.current,
        }

        onAudioReady?.(payload)

        if (autoSubmit) {
          onRecordingComplete(file)
        }

        cleanupStream(stream)
        onRecordingStateChange?.('idle')
      }

      mediaRecorder.start()
      setIsRecording(true)
      onRecordingStateChange?.('recording')

      timerRef.current = window.setInterval(() => {
        setRecordingTime(prev => {
          const next = prev + 1
          recordingTimeRef.current = next
          return next
        })
      }, 1000)
    } catch (err) {
      console.error('Error accessing microphone:', err)
      onError?.(err instanceof Error ? err : new Error('Failed to access microphone'))
    }
  }

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop()
      setIsRecording(false)
      if (timerRef.current) {
        clearInterval(timerRef.current)
        timerRef.current = null
      }
    }
  }

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60)
    const secs = seconds % 60
    return `${mins}:${secs.toString().padStart(2, '0')}`
  }

  const handleSubmit = () => {
    if (audioBlob) {
      const audioFile = new File([audioBlob], `recording-${Date.now()}.webm`, { type: 'audio/webm' })
      onRecordingComplete(audioFile)
    }
  }

  const handleReset = () => {
    setAudioBlob(null)
    if (audioUrl) {
      URL.revokeObjectURL(audioUrl)
      setAudioUrl(null)
    }
    setRecordingTime(0)
    recordingTimeRef.current = 0
    onReset?.()
  }

  return (
    <div className="audio-recorder">
      {!isRecording && !audioBlob && (
        <button
          className="record-button start"
          onClick={startRecording}
          disabled={disabled}
        >
          🎤 Start Recording
        </button>
      )}

      {isRecording && (
        <div className="recording-controls">
          <button
            className="record-button stop"
            onClick={stopRecording}
          >
            ⏹ Stop Recording
          </button>
          <div className="recording-indicator">
            <span className="pulse"></span>
            <span>Recording: {formatTime(recordingTime)}</span>
          </div>
        </div>
      )}

      {showPreview && audioBlob && !isRecording && (
        <div className="recording-preview">
          {audioUrl && (
            <audio controls src={audioUrl} className="audio-player" />
          )}
          <div className="preview-actions">
            {showAnalyzeButton && (
              <button
                className="submit-button btn-primary"
                onClick={handleSubmit}
                disabled={disabled}
              >
                Analyze Recording
              </button>
            )}
            <button
              className="reset-button btn-secondary"
              onClick={handleReset}
            >
              Record Again
            </button>
          </div>
        </div>
      )}
    </div>
  )
}

