# React Integration

Integrate whisper.apr into a React application with hooks and components.

## Installation

```bash
npm install whisper-apr
# or
yarn add whisper-apr
```

## useWhisper Hook

```typescript
// hooks/useWhisper.ts
import { useState, useEffect, useCallback, useRef } from 'react';

interface WhisperOptions {
    modelUrl?: string;
    language?: string;
    onProgress?: (progress: number) => void;
}

interface TranscriptionResult {
    text: string;
    segments: Array<{
        text: string;
        start: number;
        end: number;
    }>;
}

export function useWhisper(options: WhisperOptions = {}) {
    const [isLoading, setIsLoading] = useState(true);
    const [isTranscribing, setIsTranscribing] = useState(false);
    const [error, setError] = useState<Error | null>(null);
    const workerRef = useRef<Worker | null>(null);

    useEffect(() => {
        const worker = new Worker(
            new URL('../workers/whisper.worker.ts', import.meta.url),
            { type: 'module' }
        );

        worker.onmessage = (e) => {
            const { type, data } = e.data;
            switch (type) {
                case 'ready':
                    setIsLoading(false);
                    break;
                case 'progress':
                    options.onProgress?.(data.progress);
                    break;
                case 'error':
                    setError(new Error(data.message));
                    setIsTranscribing(false);
                    break;
            }
        };

        workerRef.current = worker;
        worker.postMessage({
            type: 'init',
            modelUrl: options.modelUrl || '/models/whisper-tiny.apr'
        });

        return () => worker.terminate();
    }, [options.modelUrl]);

    const transcribe = useCallback(async (audio: Float32Array): Promise<TranscriptionResult> => {
        if (!workerRef.current) throw new Error('Worker not initialized');

        setIsTranscribing(true);
        setError(null);

        return new Promise((resolve, reject) => {
            const handler = (e: MessageEvent) => {
                if (e.data.type === 'result') {
                    setIsTranscribing(false);
                    workerRef.current?.removeEventListener('message', handler);
                    resolve(e.data.result);
                } else if (e.data.type === 'error') {
                    reject(new Error(e.data.message));
                }
            };

            workerRef.current!.addEventListener('message', handler);
            workerRef.current!.postMessage({
                type: 'transcribe',
                audio,
                language: options.language
            });
        });
    }, [options.language]);

    return { transcribe, isLoading, isTranscribing, error };
}
```

## Transcription Component

```tsx
// components/Transcriber.tsx
import React, { useState, useRef } from 'react';
import { useWhisper } from '../hooks/useWhisper';

export function Transcriber() {
    const [result, setResult] = useState<string>('');
    const [progress, setProgress] = useState(0);
    const fileInputRef = useRef<HTMLInputElement>(null);

    const { transcribe, isLoading, isTranscribing, error } = useWhisper({
        onProgress: setProgress
    });

    const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (!file) return;

        try {
            const audioData = await decodeAudioFile(file);
            const result = await transcribe(audioData);
            setResult(result.text);
        } catch (err) {
            console.error('Transcription failed:', err);
        }
    };

    if (isLoading) {
        return <div className="loading">Loading model...</div>;
    }

    return (
        <div className="transcriber">
            <input
                ref={fileInputRef}
                type="file"
                accept="audio/*"
                onChange={handleFileSelect}
                disabled={isTranscribing}
            />

            {isTranscribing && (
                <div className="progress">
                    <div
                        className="progress-bar"
                        style={{ width: `${progress}%` }}
                    />
                </div>
            )}

            {error && (
                <div className="error">{error.message}</div>
            )}

            {result && (
                <div className="result">
                    <h3>Transcription:</h3>
                    <p>{result}</p>
                </div>
            )}
        </div>
    );
}

async function decodeAudioFile(file: File): Promise<Float32Array> {
    const arrayBuffer = await file.arrayBuffer();
    const audioContext = new AudioContext({ sampleRate: 16000 });
    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
    return audioBuffer.getChannelData(0);
}
```

## Recording Component

```tsx
// components/VoiceRecorder.tsx
import React, { useState, useRef } from 'react';
import { useWhisper } from '../hooks/useWhisper';

export function VoiceRecorder() {
    const [isRecording, setIsRecording] = useState(false);
    const [result, setResult] = useState('');
    const mediaRecorderRef = useRef<MediaRecorder | null>(null);
    const chunksRef = useRef<Blob[]>([]);

    const { transcribe, isTranscribing } = useWhisper();

    const startRecording = async () => {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        const mediaRecorder = new MediaRecorder(stream);

        mediaRecorder.ondataavailable = (e) => {
            chunksRef.current.push(e.data);
        };

        mediaRecorder.onstop = async () => {
            const blob = new Blob(chunksRef.current, { type: 'audio/webm' });
            chunksRef.current = [];

            const audioData = await blobToFloat32Array(blob);
            const result = await transcribe(audioData);
            setResult(result.text);
        };

        mediaRecorderRef.current = mediaRecorder;
        mediaRecorder.start();
        setIsRecording(true);
    };

    const stopRecording = () => {
        mediaRecorderRef.current?.stop();
        setIsRecording(false);
    };

    return (
        <div className="voice-recorder">
            <button
                onClick={isRecording ? stopRecording : startRecording}
                disabled={isTranscribing}
            >
                {isRecording ? 'Stop Recording' : 'Start Recording'}
            </button>

            {isTranscribing && <span>Transcribing...</span>}
            {result && <p>{result}</p>}
        </div>
    );
}
```

## Context Provider

```tsx
// context/WhisperContext.tsx
import React, { createContext, useContext, ReactNode } from 'react';
import { useWhisper } from '../hooks/useWhisper';

interface WhisperContextType {
    transcribe: (audio: Float32Array) => Promise<any>;
    isLoading: boolean;
    isTranscribing: boolean;
    error: Error | null;
}

const WhisperContext = createContext<WhisperContextType | null>(null);

export function WhisperProvider({ children }: { children: ReactNode }) {
    const whisper = useWhisper({
        modelUrl: '/models/whisper-base.apr'
    });

    return (
        <WhisperContext.Provider value={whisper}>
            {children}
        </WhisperContext.Provider>
    );
}

export function useWhisperContext() {
    const context = useContext(WhisperContext);
    if (!context) {
        throw new Error('useWhisperContext must be used within WhisperProvider');
    }
    return context;
}
```

## TypeScript Types

```typescript
// types/whisper.ts
export interface TranscribeOptions {
    language?: string;
    task?: 'transcribe' | 'translate';
    timestamps?: boolean;
    wordTimestamps?: boolean;
}

export interface Segment {
    text: string;
    start: number;
    end: number;
    words?: Word[];
}

export interface Word {
    word: string;
    start: number;
    end: number;
    probability: number;
}

export interface TranscriptionResult {
    text: string;
    language: string;
    segments: Segment[];
    duration: number;
}
```

## See Also

- [Web Worker Integration](./web-workers.md)
- [File Upload](./file-upload.md)
- [JavaScript Bindings](../api-reference/javascript-bindings.md)
