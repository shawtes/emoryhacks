import { useState, useRef, useEffect } from 'react'

interface AudioRecorderProps {
  onRecordingComplete: (audioFile: File) => void
  disabled?: boolean
}

export default function AudioRecorder({ onRecordingComplete, disabled }: AudioRecorderProps) {
  const [isRecording, setIsRecording] = useState(false)
  const [recordingTime, setRecordingTime] = useState(0)
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null)
  const [audioUrl, setAudioUrl] = useState<string | null>(null)
  
  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const chunksRef = useRef<Blob[]>([])
  const timerRef = useRef<number | null>(null)

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

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const mediaRecorder = new MediaRecorder(stream)
      mediaRecorderRef.current = mediaRecorder
      chunksRef.current = []

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunksRef.current.push(event.data)
        }
      }

      mediaRecorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: 'audio/webm' })
        setAudioBlob(blob)
        const url = URL.createObjectURL(blob)
        setAudioUrl(url)
        
        // Convert to WAV format for API
        convertToWav(blob)
        
        // Stop all tracks
        stream.getTracks().forEach(track => track.stop())
      }

      mediaRecorder.start()
      setIsRecording(true)
      setRecordingTime(0)

      // Start timer
      timerRef.current = window.setInterval(() => {
        setRecordingTime(prev => prev + 1)
      }, 1000)
    } catch (err) {
      console.error('Error accessing microphone:', err)
      alert('Failed to access microphone. Please check permissions.')
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

  const convertToWav = async (blob: Blob) => {
    // For simplicity, we'll send the webm file directly
    // The backend should handle conversion, or we can use a library like lamejs
    // For hackathon purposes, we'll create a File object from the blob
    const audioFile = new File([blob], 'recording.webm', { type: 'audio/webm' })
    onRecordingComplete(audioFile)
  }

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60)
    const secs = seconds % 60
    return `${mins}:${secs.toString().padStart(2, '0')}`
  }

  const handleSubmit = () => {
    if (audioBlob) {
      const audioFile = new File([audioBlob], 'recording.webm', { type: 'audio/webm' })
      onRecordingComplete(audioFile)
    }
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

      {audioBlob && !isRecording && (
        <div className="recording-preview">
          {audioUrl && (
            <audio controls src={audioUrl} className="audio-player" />
          )}
          <div className="preview-actions">
            <button
              className="submit-button btn-primary"
              onClick={handleSubmit}
              disabled={disabled}
            >
              Analyze Recording
            </button>
            <button
              className="reset-button btn-secondary"
              onClick={() => {
                setAudioBlob(null)
                setAudioUrl(null)
                setRecordingTime(0)
                if (audioUrl) {
                  URL.revokeObjectURL(audioUrl)
                }
              }}
            >
              Record Again
            </button>
          </div>
        </div>
      )}
    </div>
  )
}

