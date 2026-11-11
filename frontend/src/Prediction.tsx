import { useEffect, useState } from 'react';
import { ArrowRight, CheckCircle2, Cpu, FileImage, Info, RefreshCw, ScanSearch, ShieldCheck, UploadCloud, X, AlertTriangle } from 'lucide-react';
import { examinationLabel, fallbackTasks, type MedicalTaskId } from './medicalTasks';
import { runClinicalInference, recordPredictionMetadata, validateImageModality, type ClinicalInferenceResult } from './clinicalInference';
import { detectComputeHardware } from './webgpuTrainer';

export function Prediction() {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [task, setTask] = useState<MedicalTaskId>('PNEUMONIA');
  const [result, setResult] = useState<ClinicalInferenceResult | null>(null);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const [hardware, setHardware] = useState<string>('Detecting hardware…');
  const [metadataSynced, setMetadataSynced] = useState<boolean | null>(null);

  const selectedTask = fallbackTasks.find(item => item.taskId === task) || fallbackTasks[0];

  useEffect(() => {
    void detectComputeHardware().then(h => setHardware(h.deviceName));
  }, []);

  // Update image preview URL when file changes
  useEffect(() => {
    if (!file) {
      setPreviewUrl(null);
      return;
    }
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  const onFileChange = async (newFile: File | null) => {
    setFile(newFile);
    setResult(null);
    setError('');
    setMetadataSynced(null);

    if (newFile) {
      const check = await validateImageModality(newFile, task);
      if (!check.valid && check.error) {
        setError(check.error);
      }
    }
  };

  const onTaskChange = async (newTask: MedicalTaskId) => {
    setTask(newTask);
    setResult(null);
    setError('');
    if (file) {
      const check = await validateImageModality(file, newTask);
      if (!check.valid && check.error) {
        setError(check.error);
      }
    }
  };

  async function predict() {
    if (!file) return;
    setBusy(true);
    setError('');
    setMetadataSynced(null);

    try {
      // Strictly enforce examination modality
      const check = await validateImageModality(file, task);
      if (!check.valid) {
        throw new Error(check.error ?? 'The uploaded image does not match the selected examination modality.');
      }

      // Run in-browser client-side WebGPU inference
      const predictionResult = await runClinicalInference({
        file,
        task,
        modelVersion: 3,
      });

      setResult(predictionResult);

      // Record non-image metadata to central coordinator
      try {
        await recordPredictionMetadata({
          task,
          modelVersion: typeof predictionResult.model_version === 'number' ? predictionResult.model_version : 3,
          predictedClass: predictionResult.prediction,
          confidence: predictionResult.confidence,
        });
        setMetadataSynced(true);
      } catch {
        setMetadataSynced(false);
      }
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Inference failed on the selected image.');
    } finally {
      setBusy(false);
    }
  }

  return (
    <>
      <section className="page-heading">
        <div>
          <div className="eyebrow">CLINICAL DECISION SUPPORT</div>
          <h1>Clinical Prediction & Decision Support</h1>
          <p>
            Client-side image analysis powered by WebGPU. All diagnostic evaluations run strictly
            inside your hospital's browser sandbox—patient scans never leave this device.
          </p>
        </div>
      </section>

      <div className="privacy-shield-banner">
        <ShieldCheck size={18} />
        <span>100% Client-Side Privacy: Medical images are analyzed locally on {hardware}. Zero image bytes leave your hospital network.</span>
      </div>

      <div className="prediction-grid">
        {/* Left Panel: Upload & Task Configuration */}
        <section className="panel prediction-panel">
          <div className="panel-title">
            <div>
              <h2>Upload Medical Examination</h2>
              <p>Select diagnostic disease modality and upload the patient's radiologic image.</p>
            </div>
            <ScanSearch size={25} color="#168b78" />
          </div>

          <label>
            Examination / Disease Modality
            <select
              value={task}
              onChange={event => void onTaskChange(event.target.value as MedicalTaskId)}
            >
              {fallbackTasks.map(item => (
                <option key={item.taskId} value={item.taskId}>
                  {item.displayName} ({examinationLabel(item)})
                </option>
              ))}
            </select>
          </label>

          <div style={{ margin: '8px 0 14px 0', fontSize: '13px', color: 'var(--text-muted)' }}>
            Target Diagnostic Classes: <strong>{selectedTask.classLabels.join(', ')}</strong>
          </div>

          {!previewUrl ? (
            <label className="dropzone prediction-drop">
              <input
                type="file"
                accept="image/jpeg,image/png,image/bmp,image/webp,image/tiff"
                onChange={event => void onFileChange(event.target.files?.[0] ?? null)}
              />
              <UploadCloud size={36} />
              <b>Choose a medical image or drag & drop</b>
              <span>JPG, PNG, BMP, WEBP, TIFF · Local processing only</span>
            </label>
          ) : (
            <div className="image-preview-container">
              <img src={previewUrl} alt="Uploaded medical examination preview" className="image-preview-thumb" />
              <div className="image-preview-meta">
                <b>{file?.name}</b>
                <span>{file ? `${(file.size / 1024 / 1024).toFixed(2)} MB` : ''} · {selectedTask.modality}</span>
                <button
                  type="button"
                  className="button secondary"
                  style={{ width: 'fit-content', padding: '4px 10px', fontSize: '12px', marginTop: '6px' }}
                  onClick={() => void onFileChange(null)}
                >
                  <X size={14} /> Remove / Replace Image
                </button>
              </div>
            </div>
          )}

          {error && (
            <div className="form-error" style={{ fontSize: '13.5px', lineHeight: '1.5', marginTop: '12px', display: 'flex', gap: '8px', alignItems: 'flex-start' }}>
              <AlertTriangle size={18} style={{ flexShrink: 0, marginTop: '2px' }} />
              <div>{error}</div>
            </div>
          )}

          <button
            className="button primary full"
            disabled={!file || busy}
            onClick={() => void predict()}
            style={{ marginTop: '14px' }}
          >
            {busy ? (
              <>
                <RefreshCw size={16} className="spin" /> Analyzing Image with WebGPU…
              </>
            ) : (
              <>
                Run Clinical Inference <ArrowRight size={16} />
              </>
            )}
          </button>
        </section>

        {/* Right Panel: Model Prediction & Probability Breakdown */}
        <section className="panel result-panel">
          <div className="eyebrow">DIAGNOSTIC OUTPUT & CLASSIFICATION</div>

          {result ? (
            <>
              {/* Primary Winner Banner */}
              <div className="prediction-winner-card">
                <div className="prediction-winner-top">
                  <span style={{ fontSize: '12px', fontWeight: 750, color: '#0f766e', textTransform: 'uppercase' }}>
                    Top Clinical Prediction
                  </span>
                  <span className="prediction-confidence-pill">
                    {(result.confidence * 100).toFixed(1)}% Confidence
                  </span>
                </div>
                <div className="prediction-winner-class">{result.prediction}</div>
                <small style={{ color: '#0f766e', opacity: 0.85 }}>
                  {selectedTask.displayName} ({examinationLabel(selectedTask)}) · Processed in {result.processingTimeMs}ms
                </small>
              </div>

              {/* Multi-Class Probability Distribution */}
              <div className="class-distribution-box">
                <h3>Disease Class Probability Breakdown</h3>
                {selectedTask.classLabels.map(clsName => {
                  const prob = result.probabilities[clsName] ?? 0;
                  const isWinner = clsName.toLowerCase() === result.prediction.toLowerCase();
                  return (
                    <div className="class-bar-row" key={clsName}>
                      <span className="class-bar-label" title={clsName}>
                        {clsName}
                      </span>
                      <div className="class-bar-track">
                        <div
                          className={`class-bar-fill ${isWinner ? 'active' : ''}`}
                          style={{ width: `${Math.max(prob * 100, 2)}%` }}
                        />
                      </div>
                      <span className="class-bar-percent">{(prob * 100).toFixed(1)}%</span>
                    </div>
                  );
                })}
              </div>

              {/* Clinical Findings & Biomarkers */}
              {result.findings && result.findings.length > 0 && (
                <div className="findings-box">
                  <h3>
                    <Info size={15} color="#0f766e" /> Anatomical Observations & Biomarkers
                  </h3>
                  <ul className="findings-list">
                    {result.findings.map((finding, idx) => (
                      <li key={idx}>{finding}</li>
                    ))}
                  </ul>
                </div>
              )}

              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '12px', color: 'var(--text-muted)', marginBottom: '14px' }}>
                <Cpu size={14} />
                <span>Accelerator: {result.device} · Model v{String(result.model_version)}</span>
                {metadataSynced !== null && (
                  <span style={{ marginLeft: 'auto', color: metadataSynced ? '#0d8a68' : 'var(--amber)', display: 'flex', alignItems: 'center', gap: '4px' }}>
                    <CheckCircle2 size={13} /> {metadataSynced ? 'Metadata recorded' : 'Local metadata queued'}
                  </span>
                )}
              </div>
            </>
          ) : (
            <div style={{ padding: '36px 12px', textAlign: 'center', color: 'var(--text-muted)' }}>
              <FileImage size={48} strokeWidth={1.2} style={{ margin: '0 auto 12px', opacity: 0.6 }} />
              <p style={{ margin: '0 0 6px 0', fontSize: '16px', fontWeight: 600, color: 'var(--navy)' }}>
                {examinationLabel(selectedTask)} Selected
              </p>
              <p style={{ margin: 0, fontSize: '13.5px' }}>
                Upload a medical image and click "Run Clinical Inference" to view diagnostic classification,
                all-class probability breakdown, and biomarker analysis.
              </p>
            </div>
          )}

          <div className="clinical-note">
            <b>Institutional Decision-Support Tool</b>
            This AI-assisted classification is provided for clinical research and decision-support verification.
            It does not replace comprehensive medical evaluation by licensed healthcare professionals.
          </div>
        </section>
      </div>
    </>
  );
}
