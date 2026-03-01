import React, { useEffect, useMemo, useRef, useState, useCallback } from 'react';
import { useOutletContext } from 'react-router-dom';
import './VoiceInsight.css';

const VoiceInsight = () => {
  const { user } = useOutletContext() || {};
  const [isRecording, setIsRecording] = useState(false);
  const [isRecognizing, setIsRecognizing] = useState(false);
  const [audioBlob, setAudioBlob] = useState(null);
  const [audioUrl, setAudioUrl] = useState(null);
  const [liveTranscript, setLiveTranscript] = useState('');
  const [editableTranscript, setEditableTranscript] = useState('');
  const [elapsedSec, setElapsedSec] = useState(0);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [recognitionWarning, setRecognitionWarning] = useState('');
  const mediaRecorderRef = useRef(null);
  const chunksRef = useRef([]);
  const recordingStartRef = useRef(null);
  const recognitionRef = useRef(null);
  const shouldRestartRecognitionRef = useRef(false);
  const finalTranscriptRef = useRef('');
  const tickIntervalRef = useRef(null);
  const isRecordingRef = useRef(false);

  const MIN_RECORDING_SECONDS = 5;
  const MAX_RECORDING_SECONDS = 60;
  const MIN_WORDS = 10;

  const recognitionSupported = useMemo(() => {
    if (typeof window === 'undefined') return false;
    return Boolean(window.SpeechRecognition || window.webkitSpeechRecognition);
  }, []);

  const transcriptWordCount = useMemo(() => {
    const text = (editableTranscript || liveTranscript || '').trim();
    if (!text) return 0;
    return text.split(/\s+/).filter(Boolean).length;
  }, [editableTranscript, liveTranscript]);

  const _stopTick = useCallback(() => {
    if (tickIntervalRef.current) {
      clearInterval(tickIntervalRef.current);
      tickIntervalRef.current = null;
    }
  }, []);

  const _stopRecognition = useCallback(() => {
    shouldRestartRecognitionRef.current = false;
    try {
      if (recognitionRef.current) {
        recognitionRef.current.onresult = null;
        recognitionRef.current.onerror = null;
        recognitionRef.current.onend = null;
        recognitionRef.current.stop();
      }
    } catch {
      // non-fatal
    } finally {
      recognitionRef.current = null;
      setIsRecognizing(false);
    }
  }, []);

  const startRecording = useCallback(async () => {
    setError('');
    setResult(null);
    setAudioBlob(null);
    setLiveTranscript('');
    setEditableTranscript('');
    setElapsedSec(0);
    setRecognitionWarning('');
    finalTranscriptRef.current = '';
    if (audioUrl) {
      URL.revokeObjectURL(audioUrl);
      setAudioUrl(null);
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      chunksRef.current = [];
      recordingStartRef.current = Date.now();
      isRecordingRef.current = true;

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };
      mediaRecorder.onstop = () => {
        stream.getTracks().forEach((t) => t.stop());
        const duration = recordingStartRef.current ? (Date.now() - recordingStartRef.current) / 1000 : 0;
        if (chunksRef.current.length > 0 && duration >= MIN_RECORDING_SECONDS) {
          const blob = new Blob(chunksRef.current, { type: 'audio/webm' });
          setAudioBlob(blob);
          setAudioUrl(URL.createObjectURL(blob));
          setError('');
        } else if (duration > 0 && duration < MIN_RECORDING_SECONDS) {
          setError(`Recording too short (${duration.toFixed(1)}s). Please record at least ${MIN_RECORDING_SECONDS} seconds.`);
        }
      };
      mediaRecorder.start();
      setIsRecording(true);

      // Tick timer (max 60s)
      _stopTick();
      tickIntervalRef.current = setInterval(() => {
        const start = recordingStartRef.current;
        if (!start) return;
        const sec = Math.floor((Date.now() - start) / 1000);
        setElapsedSec(sec);
        if (sec >= MAX_RECORDING_SECONDS) {
          // Auto-stop at 60 seconds
          try {
            mediaRecorderRef.current?.stop();
          } catch {
            // non-fatal
          } finally {
            mediaRecorderRef.current = null;
            setIsRecording(false);
            isRecordingRef.current = false;
            _stopRecognition();
            _stopTick();
            setEditableTranscript((finalTranscriptRef.current || '').trim());
          }
        }
      }, 250);

      // Live transcription via Web Speech API (if supported)
      if (recognitionSupported) {
        try {
          const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
          const recognition = new SR();
          recognition.continuous = true;
          recognition.interimResults = true;
          recognition.lang = 'en-US';

          shouldRestartRecognitionRef.current = true;
          recognitionRef.current = recognition;
          setIsRecognizing(true);

          recognition.onresult = (event) => {
            let interim = '';
            let finalText = finalTranscriptRef.current || '';

            for (let i = event.resultIndex; i < event.results.length; ++i) {
              const res = event.results[i];
              const piece = (res?.[0]?.transcript || '').trim();
              if (!piece) continue;
              if (res.isFinal) {
                finalText = `${finalText} ${piece}`.trim();
              } else {
                interim = `${interim} ${piece}`.trim();
              }
            }

            finalTranscriptRef.current = finalText;
            const combined = `${finalText} ${interim}`.trim();
            setLiveTranscript(combined);
          };

          recognition.onerror = (e) => {
            const msg = e?.error ? `Speech recognition error: ${e.error}` : 'Speech recognition error.';
            setRecognitionWarning(msg);
          };

          recognition.onend = () => {
            setIsRecognizing(false);
            // Some browsers stop recognition periodically; try restarting while recording
            if (shouldRestartRecognitionRef.current && isRecordingRef.current) {
              setTimeout(() => {
                try {
                  recognition.start();
                  setIsRecognizing(true);
                } catch {
                  // non-fatal
                }
              }, 250);
            }
          };

          recognition.start();
        } catch {
          setRecognitionWarning(
            'Live transcription is not available right now. Audio will be processed after recording.'
          );
        }
      } else {
        setRecognitionWarning(
          'Live transcription not supported in this browser. Audio will be processed after recording.'
        );
      }
    } catch (err) {
      setError('Microphone access denied. Please allow microphone permission.');
    }
  }, [MAX_RECORDING_SECONDS, MIN_RECORDING_SECONDS, _stopRecognition, _stopTick, audioUrl, liveTranscript, recognitionSupported]);

  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current && isRecording) {
      try {
        mediaRecorderRef.current.stop();
      } catch {
        // non-fatal
      } finally {
        mediaRecorderRef.current = null;
        setIsRecording(false);
        isRecordingRef.current = false;
        _stopRecognition();
        _stopTick();
        setEditableTranscript((finalTranscriptRef.current || liveTranscript || '').trim());
      }
    }
  }, [_stopRecognition, _stopTick, isRecording, liveTranscript]);

  const handleReRecord = useCallback(() => {
    _stopRecognition();
    _stopTick();
    setAudioBlob(null);
    if (audioUrl) {
      URL.revokeObjectURL(audioUrl);
      setAudioUrl(null);
    }
    setResult(null);
    setError('');
    setRecognitionWarning('');
    setLiveTranscript('');
    setEditableTranscript('');
    setElapsedSec(0);
    finalTranscriptRef.current = '';
  }, [audioUrl]);

  const submitTranscript = async () => {
    if (!user?.email) {
      setError('Please sign in to submit your voice insight.');
      return;
    }
    const transcriptToSend = (editableTranscript || liveTranscript || '').trim();
    if (!transcriptToSend) {
      setError('No speech detected. Please check microphone permissions.');
      return;
    }
    if (transcriptToSend.split(/\s+/).filter(Boolean).length < MIN_WORDS) {
      setError(`Please capture at least ${MIN_WORDS} words before submitting.`);
      return;
    }
    setLoading(true);
    setError('');
    try {
      const apiBase = import.meta.env.VITE_API_URL || 'http://127.0.0.1:5000';
      const res = await fetch(`${apiBase}/api/voice-analysis`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_email: user.email,
          transcript: transcriptToSend,
        }),
      });
      const data = await res.json();
      if (!res.ok || data.error) {
        throw new Error(data.error || 'Voice analysis failed.');
      }
      setResult(data);
    } catch (err) {
      setError(err.message || 'Something went wrong. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const submitAudioFallback = async () => {
    if (!user?.email) {
      setError('Please sign in to submit your voice insight.');
      return;
    }
    if (!audioBlob) {
      setError('Please record your response first.');
      return;
    }
    setLoading(true);
    setError('');
    try {
      const apiBase = import.meta.env.VITE_API_URL || 'http://127.0.0.1:5000';
      const formData = new FormData();
      formData.append('audio', audioBlob, 'recording.webm');
      formData.append('user_email', user.email);
      // Send any transcript preview too (helps debugging / fallback)
      const preview = (editableTranscript || liveTranscript || '').trim();
      if (preview) formData.append('transcript', preview);

      const res = await fetch(`${apiBase}/api/voice-analysis`, {
        method: 'POST',
        body: formData,
      });
      const data = await res.json();
      if (!res.ok || data.error) {
        // If backend returns transcript_preview, surface it so user can edit and re-submit
        if (data?.transcript_preview && !editableTranscript) {
          setEditableTranscript(String(data.transcript_preview));
        }
        throw new Error(data.error || 'Voice analysis failed.');
      }
      setResult(data);
    } catch (err) {
      setError(err.message || 'Something went wrong. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    return () => {
      _stopRecognition();
      _stopTick();
      if (audioUrl) URL.revokeObjectURL(audioUrl);
    };
  }, [_stopRecognition, _stopTick, audioUrl]);

  const mm = String(Math.floor(Math.min(elapsedSec, MAX_RECORDING_SECONDS) / 60)).padStart(2, '0');
  const ss = String(Math.min(elapsedSec, MAX_RECORDING_SECONDS) % 60).padStart(2, '0');

  return (
    <div className="voice-insight-container">
      <div className="voice-insight-card">
        <h2 className="voice-insight-title">
          <i className="fas fa-microphone-alt" /> Record Your Career Aspiration
        </h2>
        <p className="voice-insight-prompt">
          In up to {MAX_RECORDING_SECONDS} seconds, describe your dream career and why you want to pursue it.
        </p>

        {!result ? (
          <>
            {recognitionWarning && (
              <p className="voice-warning">
                <i className="fas fa-info-circle" /> {recognitionWarning}
              </p>
            )}

            <div className="voice-controls">
              {!isRecording ? (
                <button
                  type="button"
                  className="voice-btn voice-btn-record"
                  onClick={startRecording}
                  disabled={loading}
                >
                  <i className="fas fa-microphone" /> Start Recording
                </button>
              ) : (
                <button
                  type="button"
                  className="voice-btn voice-btn-stop"
                  onClick={stopRecording}
                >
                  <i className="fas fa-stop" /> Stop Recording
                </button>
              )}
            </div>

            {isRecording && (
              <div className="voice-recording-status">
                <span className="voice-dot" aria-hidden="true" />
                <span>
                  Recording… {mm}:{ss} / 01:00
                  {isRecognizing && <span className="voice-recognizing"> · Live</span>}
                </span>
              </div>
            )}

            {(isRecording || liveTranscript) && (
              <div className="voice-transcript-live">
                <div className="voice-transcript-live-header">
                  <span className="voice-transcript-live-title">Live Transcript</span>
                  <span className="voice-transcript-live-meta">
                    {transcriptWordCount} words
                  </span>
                </div>
                <div className={`voice-transcript-box ${isRecording ? 'recording' : ''}`}>
                  {liveTranscript ? liveTranscript : 'Listening…'}
                </div>
              </div>
            )}

            {audioUrl && !isRecording && (
              <div className="voice-preview">
                <p className="voice-preview-label">Preview</p>
                <audio src={audioUrl} controls className="voice-audio" />

                <div className="voice-edit-transcript">
                  <p className="voice-preview-label">Edit Transcript</p>
                  <textarea
                    className="voice-transcript-edit"
                    value={editableTranscript}
                    onChange={(e) => setEditableTranscript(e.target.value)}
                    placeholder="Your transcript will appear here. You can edit it before submitting."
                    rows={5}
                  />
                  <div className="voice-transcript-actions">
                    <span className="voice-transcript-hint">
                      {transcriptWordCount < MIN_WORDS
                        ? `Capture at least ${MIN_WORDS} words to enable submit.`
                        : 'Looks good—submit when ready.'}
                    </span>
                    <span className="voice-transcript-count">{transcriptWordCount} words</span>
                  </div>

                  {!editableTranscript.trim() && (
                    <p className="voice-empty-speech">
                      No speech detected. Please check microphone permissions.
                    </p>
                  )}
                </div>

                <div className="voice-preview-actions">
                  <button
                    type="button"
                    className="voice-btn voice-btn-secondary"
                    onClick={handleReRecord}
                  >
                    <i className="fas fa-redo" /> Re-record
                  </button>
                  <button
                    type="button"
                    className="voice-btn voice-btn-submit"
                    onClick={submitTranscript}
                    disabled={loading || transcriptWordCount < MIN_WORDS}
                  >
                    {loading ? (
                      <>
                        <i className="fas fa-spinner fa-spin" /> Analyzing…
                      </>
                    ) : (
                      <>
                        <i className="fas fa-paper-plane" /> Submit Transcript
                      </>
                    )}
                  </button>
                  <button
                    type="button"
                    className="voice-btn voice-btn-secondary"
                    onClick={submitAudioFallback}
                    disabled={loading}
                    title="Uses backend speech-to-text. Useful if live transcription fails."
                  >
                    <i className="fas fa-cloud-upload-alt" /> Submit Audio (fallback)
                  </button>
                </div>
              </div>
            )}

            {error && <p className="voice-error">{error}</p>}
          </>
        ) : (
          <div className="voice-result">
            <h3>Voice Insight Result</h3>
            {result.transcribed_text && (
              <div className="voice-transcript">
                <strong>Transcript:</strong>
                <p>{result.transcribed_text}</p>
              </div>
            )}
            <div className="voice-metrics">
              {typeof result.motivation_score === 'number' && (
                <div className="voice-metric">
                  <span className="voice-metric-label">Motivation</span>
                  <span className="voice-metric-value">{Math.round(result.motivation_score)}%</span>
                </div>
              )}
              {typeof result.confidence_score === 'number' && (
                <div className="voice-metric">
                  <span className="voice-metric-label">Confidence</span>
                  <span className="voice-metric-value">{Math.round(result.confidence_score)}%</span>
                </div>
              )}
              {result.top_voice_career && (
                <div className="voice-metric">
                  <span className="voice-metric-label">Top career (voice)</span>
                  <span className="voice-metric-value voice-accent">{result.top_voice_career}</span>
                </div>
              )}
            </div>
            <p className="voice-success-msg">
              Your voice insight has been saved and will influence your career recommendations.
            </p>
            <button
              type="button"
              className="voice-btn voice-btn-secondary"
              onClick={() => {
                setResult(null);
                handleReRecord();
              }}
            >
              <i className="fas fa-redo" /> Record Again
            </button>
          </div>
        )}
      </div>
    </div>
  );
};

export default VoiceInsight;
