import { useState, useRef } from 'react'

interface FileUploaderProps {
  onFileSelect: (file: File) => void
  disabled?: boolean
}

export default function FileUploader({ onFileSelect, disabled }: FileUploaderProps) {
  const [dragActive, setDragActive] = useState(false)
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true)
    } else if (e.type === 'dragleave') {
      setDragActive(false)
    }
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(false)

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0])
    }
  }

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    e.preventDefault()
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0])
    }
  }

  const handleFile = (file: File) => {
    // Validate file type
    const isAudio = file.type.startsWith('audio/')
    const isAllowedExt = /\.(mp3|wav|m4a)$/i.test(file.name)
    if (!(isAudio && isAllowedExt)) {
      alert('Please select an MP3 (preferred), WAV, or M4A file.')
      return
    }

    setSelectedFile(file)
    onFileSelect(file)
  }

  const handleButtonClick = () => {
    fileInputRef.current?.click()
  }

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes'
    const k = 1024
    const sizes = ['Bytes', 'KB', 'MB', 'GB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i]
  }

  return (
    <div className="file-uploader">
      <div
        className={`upload-area ${dragActive ? 'drag-active' : ''} ${disabled ? 'disabled' : ''}`}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept="audio/mpeg,audio/mp3,audio/wav,audio/x-wav,audio/m4a"
          onChange={handleChange}
          disabled={disabled}
          style={{ display: 'none' }}
        />
        
        {!selectedFile ? (
          <>
            <div className="upload-icon">📁</div>
            <p className="upload-text">
              Drag and drop an audio file here, or
            </p>
            <button
              className="browse-button btn-primary"
              onClick={handleButtonClick}
              disabled={disabled}
            >
              Browse Files
            </button>
            <p className="upload-hint">
              Supports WAV, MP3, M4A, WebM, OGG
            </p>
          </>
        ) : (
          <div className="file-info">
            <div className="file-icon">🎵</div>
            <div className="file-details">
              <p className="file-name">{selectedFile.name}</p>
              <p className="file-size">{formatFileSize(selectedFile.size)}</p>
            </div>
            <button
              className="remove-button"
              onClick={() => {
                setSelectedFile(null)
                if (fileInputRef.current) {
                  fileInputRef.current.value = ''
                }
              }}
              disabled={disabled}
            >
              ✕
            </button>
          </div>
        )}
      </div>
    </div>
  )
}

