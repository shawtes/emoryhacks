export interface DevAudioMetadata {
  patientId: string
  segmentsCompleted: number
  durationSeconds: number
  locale: string
}

export interface DevAudioEntry extends DevAudioMetadata {
  id: string
  filename: string
  createdAt: number
  blob: Blob
}

const DB_NAME = 'VoiceVitalDevAudio'
const STORE_NAME = 'devaudio'
const DB_VERSION = 1

function openDatabase(): Promise<IDBDatabase> {
  if (typeof window === 'undefined' || !window.indexedDB) {
    return Promise.reject(new Error('IndexedDB not available'))
  }

  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION)

    request.onupgradeneeded = () => {
      const db = request.result
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        db.createObjectStore(STORE_NAME, { keyPath: 'id' })
      }
    }

    request.onsuccess = () => resolve(request.result)
    request.onerror = () => reject(request.error ?? new Error('Failed to open IndexedDB'))
  })
}

async function withStore<T>(mode: IDBTransactionMode, handler: (store: IDBObjectStore) => T | Promise<T>): Promise<T> {
  const db = await openDatabase()
  return new Promise<T>((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, mode)
    const store = tx.objectStore(STORE_NAME)
    let result: T | undefined

    Promise.resolve(handler(store))
      .then((value) => {
          result = value
      })
      .catch((error) => {
        tx.abort()
        reject(error)
      })

    tx.oncomplete = () => {
      db.close()
      if (typeof result === 'undefined') {
        reject(new Error('Transaction completed without a result'))
        return
      }
      resolve(result)
    }
    tx.onerror = () => reject(tx.error ?? new Error('Transaction failed'))
  })
}

export async function saveDevAudio(blob: Blob, metadata: DevAudioMetadata): Promise<DevAudioEntry> {
  const id = `${metadata.patientId}_${Date.now()}`
  const entry: DevAudioEntry = {
    ...metadata,
    id,
    filename: `${id}.webm`,
    createdAt: Date.now(),
    blob,
  }

  await withStore('readwrite', (store) => store.put(entry))
  return entry
}

export async function listDevAudio(): Promise<DevAudioEntry[]> {
  return withStore('readonly', (store) => {
    return new Promise<DevAudioEntry[]>((resolve, reject) => {
      const request = store.getAll()
      request.onsuccess = () => resolve(request.result as DevAudioEntry[])
      request.onerror = () => reject(request.error ?? new Error('Failed to read dev audio entries'))
    })
  })
}


