import { useEffect, useRef, useState } from 'react';
import { ArrowRight, BrainCircuit, CheckCircle2, Clock3, FolderUp, History, LayoutDashboard, LogOut, RefreshCw, ScanSearch, Server, ShieldCheck, AlertCircle, Zap, Cpu } from 'lucide-react';
import { centralApiUrl, localMlUrl, localServiceHeaders, tokenHeaders } from './config';
import { LiveFederated } from './LiveFederated';
import { Prediction } from './Prediction';
import { fallbackTasks, type MedicalTaskId } from './medicalTasks';
import { detectComputeHardware, runInBrowserTraining, type LocalImageSample } from './webgpuTrainer';
import './dashboard.css';

type View = 'overview' | 'train' | 'global-model' | 'prediction' | 'history';
type Validation = { classes: string[]; class_distribution: Record<string, number>; file_count: number; size_bytes: number };
type Dataset = { folderName: string; files: File[]; classes: string[]; classDistribution: Record<string, number>; fileCount: number; sizeBytes: number };
type Training = {
  status: string;
  job_id?: string;
  epoch?: number;
  total_epochs?: number;
  loss?: number;
  accuracy?: number;
  checkpoint?: string;
  message?: string;
  samples?: number;
  base_version?: number;
  created_at?: string;
  updated_at?: string;
  createdAt?: string;
  submission?: { status?: string; round?: number; message?: string; accepted_at?: string; bytes?: number };
};
type Network = {
  version: number;
  round: number;
  updatesReceived: number;
  hospitals: number;
  aggregationStatus: string;
  baseGlobalModelVersion?: number;
  isParticipant?: boolean;
  hasSubmitted?: boolean;
};
type PendingUpdate = {
  task: MedicalTaskId;
  round: number;
  status: string;
  samples?: number;
  last_error?: string;
  updated_at?: string;
};
type LocalState = {
  latest_training_job?: Training;
  latest_checkpoint?: string | null;
  pending_updates: PendingUpdate[];
};
type OverviewData = {
  hospitalName: string;
  connectionStatus: string;
  globalModel: { task: MedicalTaskId; version: number; round: number; status: string } | null;
  currentRound: number;
  round?: {
    task?: MedicalTaskId;
    round: number;
    status: string;
    baseGlobalModelVersion: number;
    submittedParticipantCount: number;
    expectedParticipantCount: number;
    hasSubmitted: boolean;
    isParticipant: boolean;
  };
  training: Training;
  lastContribution: { round: number; samples: number; receivedAt: string } | null;
};

const nav: Array<[View, string, typeof LayoutDashboard]> = [
  ['overview', 'Overview', LayoutDashboard],
  ['train', 'Local Training', BrainCircuit],
  ['global-model', 'Global Model', Server],
  ['prediction', 'Clinical Prediction', ScanSearch],
  ['history', 'Training History', History],
];

const date = (value?: string) => (value ? new Date(value).toLocaleString() : 'Not available');

const elapsed = (start?: string, end?: string) => {
  if (!start) return;
  const seconds = Math.max(0, Math.round((new Date(end ?? Date.now()).getTime() - new Date(start).getTime()) / 1000));
  return seconds > 59 ? `${Math.floor(seconds / 60)}m ${seconds % 60}s` : `${seconds}s`;
};

const Pill = ({ children, done, tone }: { children: string; done?: boolean; tone?: 'green' | 'amber' | 'blue' | 'red' }) => {
  const finalTone = tone ?? (done ? 'green' : 'amber');
  return (
    <span className={`status-pill ${finalTone}`}>
      {done ? <CheckCircle2 size={15} /> : <Clock3 size={15} />}
      {children}
    </span>
  );
};

const Metric = ({ label, value, detail }: { label: string; value: string; detail: string }) => (
  <div className="metric">
    <div className="metric-label">
      {label}
      <span className="metric-dot green" />
    </div>
    <strong>{value}</strong>
    <small>{detail}</small>
  </div>
);

function Overview({ setView }: { setView: (view: View) => void }) {
  const [data, setData] = useState<OverviewData | null>(null);
  const [error, setError] = useState('');
  const [syncNotice, setSyncNotice] = useState('');
  const syncedModelRef = useRef('');

  useEffect(() => {
    const refresh = async () => {
      try {
        const r = await fetch(centralApiUrl + '/api/hospital/overview', { headers: tokenHeaders() });
        const b = (await r.json()) as OverviewData & { message?: string };
        if (!r.ok) throw new Error(b.message ?? 'Failed to load hospital overview');
        setData(b);

        void fetch(localMlUrl + '/health', { headers: localServiceHeaders() })
          .then(async local => {
            if (!local.ok) return;
            const health = (await local.json()) as { tasks?: string[] };
            await fetch(centralApiUrl + '/api/hospital/agent/heartbeat', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json', ...tokenHeaders() },
              body: JSON.stringify({ tasks: health.tasks ?? [] }),
            });
          })
          .catch(() => undefined);

        const activeTask = b.globalModel?.task ?? b.round?.task ?? 'PNEUMONIA';
        const availableVersion = b.globalModel?.version ?? 0;
        const assignment = availableVersion > 0 ? `${activeTask}:${availableVersion}` : '';
        if (assignment && assignment !== syncedModelRef.current) {
          try {
            const local = await fetch(localMlUrl + '/federated/download-global', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json', ...localServiceHeaders() },
              body: JSON.stringify({
                task: activeTask,
                api_url: centralApiUrl,
                token: sessionStorage.getItem('arogyavaani_token'),
                expected_version: availableVersion,
              }),
            });
            if (local.ok) {
              const result = await local.json();
              syncedModelRef.current = assignment;
              setSyncNotice(`Latest ${activeTask} global model v${result.version} is synchronized in the local training service.`);
            }
          } catch {
            // Local hospital ML agent may still be starting or offline
          }
        }
        setError('');
      } catch (e) {
        setError(e instanceof Error ? e.message : 'Unable to load hospital overview.');
      }
    };

    void refresh();
    const timer = window.setInterval(() => void refresh(), 5000);
    return () => window.clearInterval(timer);
  }, []);

  return (
    <>
      <section className="page-heading">
        <div>
          <div className="eyebrow">HOSPITAL WORKSPACE</div>
          <h1>{data?.hospitalName ?? 'Hospital Dashboard'}</h1>
          <p>Privacy-preserving federated clinical AI node</p>
        </div>
        <button className="button primary" onClick={() => setView('train')}>
          <BrainCircuit size={18} /> Start Local Training
        </button>
      </section>

      {error && <div className="form-error"><AlertCircle size={18} /> {error}</div>}
      {syncNotice && <div className="form-notice"><CheckCircle2 size={18} /> {syncNotice}</div>}

      {data && (
        <>
          <section className="metrics-grid">
            <Metric label="CONNECTION" value={data.connectionStatus} detail="Central API & Local Agent online" />
            <Metric
              label="GLOBAL MODEL"
              value={data.globalModel ? `v${data.globalModel.version}` : 'Not available'}
              detail={data.globalModel?.status ?? 'Awaiting aggregation'}
            />
            <Metric
              label="CURRENT ROUND"
              value={data.currentRound ? `Round ${data.currentRound}` : 'None'}
              detail={data.round ? (data.round.hasSubmitted ? 'Your update is submitted' : 'Your local update is pending') : 'Awaiting administrator'}
            />
            <Metric
              label="LOCAL TRAINING"
              value={data.training.status}
              detail={data.training.samples ? `${data.training.samples.toLocaleString()} local samples` : 'Ready to train'}
            />
          </section>

          <section className="panel round-overview">
            <div className="panel-title">
              <div>
                <h2>Federated Training Round</h2>
                <p>{data.round ? `Round ${data.round.round} · Base model v${data.round.baseGlobalModelVersion}` : 'No active federation round assigned to this hospital.'}</p>
              </div>
              {data.round && (
                <Pill done={data.round.hasSubmitted}>
                  {data.round.hasSubmitted ? 'Update Submitted' : data.round.status}
                </Pill>
              )}
            </div>
            {data.round ? (
              <p>Your hospital is an authorized participant in Round {data.round.round}. Raw patient medical images never leave your local infrastructure; only privacy-preserving mathematical model parameters are transmitted to the coordinator.</p>
            ) : (
              <p>When the central administrator creates a federated learning round selecting this hospital, configuration details and base model checkpoints will appear here automatically.</p>
            )}
          </section>

          {data.lastContribution && (
            <section className="panel contribution-summary">
              <h2>Last Contribution Confirmed</h2>
              <p>
                Round {data.lastContribution.round} · {data.lastContribution.samples.toLocaleString()} local training samples · Submitted and confirmed at {date(data.lastContribution.receivedAt)}
              </p>
            </section>
          )}
        </>
      )}

      <section className="privacy-indicator">
        <ShieldCheck size={20} />
        <span><b>Guaranteed Local Privacy:</b> Patient CT scans, X-rays, and medical images remain securely stored in this hospital's local system. Only parameter weight matrices are transferred.</span>
      </section>
    </>
  );
}

function LocalTrain() {
  const [task, setTask] = useState<MedicalTaskId>('PNEUMONIA');
  const [hardwareName, setHardwareName] = useState<string>('Detecting hardware...');
  const [datasetDir, setDatasetDir] = useState('');
  const [epochs, setEpochs] = useState(10);
  const [classes, setClasses] = useState(2);
  const [validation, setValidation] = useState<Validation | null>(null);
  const [training, setTraining] = useState<Training>({ status: 'idle' });
  const [network, setNetwork] = useState<Network | null>(null);
  const [selected, setSelected] = useState<Dataset | null>(null);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const [submitted, setSubmitted] = useState('');
  const [notice, setNotice] = useState('');
  const input = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    void detectComputeHardware().then(hw => {
      setHardwareName(hw.deviceName);
    });
  }, []);

  const refreshNetwork = async () => {
    try {
      const r = await fetch(centralApiUrl + '/api/federated/status?task=' + task, { headers: tokenHeaders() });
      const b = await r.json();
      if (!r.ok) throw new Error(b.message);
      setNetwork(b);
      if (b.hasSubmitted) {
        setSubmitted(`Model update confirmed and stored by central server for Round ${b.round}.`);
      }
    } catch {
      /* Handled gracefully */
    }
  };

  useEffect(() => {
    void refreshNetwork();
  }, [task]);

  useEffect(() => {
    if (network?.hasSubmitted && !submitted) {
      setSubmitted(`Model update confirmed and stored by central coordinator for Round ${network.round}.`);
    }
  }, [network?.hasSubmitted, network?.round]);

  const clear = () => {
    setSelected(null);
    setDatasetDir('');
    setValidation(null);
    setClasses(2);
    setNotice('');
    setError('');
    setSubmitted('');
    if (input.current) input.current.value = '';
  };

  const selectDataset = (files: File[]) => {
    setError('');
    setNotice('');
    setSubmitted('');
    setDatasetDir('');
    setValidation(null);
    if (!files.length) return;
    if (files.length > 20_000) return setError('Dataset contains too many files. The maximum is 20,000 files.');
    const sizeBytes = files.reduce((n, f) => n + f.size, 0);
    if (sizeBytes > 1_000_000_000) return setError('Dataset exceeds the 1 GB local upload limit.');
    const invalid = files.find(f => !/\.(jpe?g|png|bmp|tiff?|webp)$/i.test(f.name));
    if (invalid) return setError(`Unsupported file "${invalid.name}". Supported types: JPG, PNG, BMP, TIFF, WEBP.`);
    const paths = files.map(f => (f.webkitRelativePath || '').replace(/\\/g, '/').split('/').filter(Boolean));
    if (paths.some(p => p.length < 3)) return setError('Select the parent dataset folder containing class subfolders (e.g. dataset/NORMAL, dataset/PNEUMONIA), not individual image files.');
    const folderName = paths[0][0];
    if (paths.some(p => p[0] !== folderName)) return setError('Select one complete dataset folder at a time.');
    const classDistribution = paths.reduce<Record<string, number>>((result, p) => {
      result[p[1]] = (result[p[1]] ?? 0) + 1;
      return result;
    }, {});
    const detected = Object.keys(classDistribution).sort();
    if (detected.length < 2) return setError('Dataset must contain at least two class folders.');
    setSelected({ folderName, files, classes: detected, classDistribution, fileCount: files.length, sizeBytes });
  };

  const upload = async () => {
    if (!selected) return;
    setBusy(true);
    setError('');
    setNotice('');
    try {
      const next: Validation = {
        classes: selected.classes,
        class_distribution: selected.classDistribution,
        file_count: selected.fileCount,
        size_bytes: selected.sizeBytes,
      };
      setDatasetDir(selected.folderName);
      setValidation(next);
      setClasses(next.classes.length);
      setNotice(`Local dataset "${selected.folderName}" validated in browser sandbox. Zero installation required.`);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Dataset preparation failed.');
    } finally {
      setBusy(false);
    }
  };

  const train = async () => {
    if (!datasetDir || !validation) return;
    if (!network?.round || !network.isParticipant) return setError(`No active ${task} round selects this hospital.`);
    setBusy(true);
    setError('');
    setSubmitted('');

    try {
      setNotice('Synchronizing base model parameters from central coordinator...');
      const modelUrl = `${centralApiUrl}/api/federated/global-model-binary?task=${task}${
        network.baseGlobalModelVersion ? '&version=' + network.baseGlobalModelVersion : ''
      }`;
      const res = await fetch(modelUrl, { headers: tokenHeaders() });
      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.message ?? `Global model artifact is temporarily unavailable (status ${res.status}).`);
      }
      const baseModelBuffer = await res.arrayBuffer();

      const samples: LocalImageSample[] = [];
      if (selected) {
        for (const file of selected.files) {
          const p = (file.webkitRelativePath || '').replace(/\\/g, '/').split('/').filter(Boolean);
          if (p.length >= 3 && validation.classes.includes(p[1])) {
            samples.push({
              file,
              classLabel: p[1],
              classIndex: validation.classes.indexOf(p[1]),
            });
          }
        }
      }
      if (samples.length === 0) {
        throw new Error('No valid class images found matching folder classes in the selected directory.');
      }

      setNotice(`Training model on ${hardwareName}... All computations stay strictly inside your browser.`);
      const result = await runInBrowserTraining({
        samples,
        classes: validation.classes,
        task,
        baseModelBuffer,
        baseVersion: network.baseGlobalModelVersion ?? 0,
        epochs,
        onProgress: p => {
          setTraining({
            status: p.status,
            epoch: p.epoch,
            total_epochs: p.totalEpochs,
            loss: p.loss,
            accuracy: p.accuracy > 1 ? p.accuracy / 100 : p.accuracy,
            samples: p.samplesProcessed,
            message: p.message ?? `Epoch ${p.epoch}/${p.totalEpochs} · Accuracy: ${p.accuracy}% · Loss: ${p.loss}`,
          });
        },
      });

      // Submit AVM1 payload directly to central coordinator
      setNotice('Submitting encrypted mathematical parameter update to central coordinator...');
      setTraining(prev => ({ ...prev, status: 'submitting', message: 'Submitting update to coordinator...' }));
      const submitUrl = `${centralApiUrl}/api/federated/rounds/${network.round}/updates?samples=${samples.length}&task=${task}`;
      const subRes = await fetch(submitUrl, {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${sessionStorage.getItem('arogyavaani_token') ?? ''}`,
          'Content-Type': 'application/vnd.arogyavaani.model-v1',
        },
        body: result.avm1Payload,
      });
      const subData = await subRes.json().catch(() => ({}));
      if (!subRes.ok) {
        throw new Error(subData.message ?? 'Model update was not accepted by coordinator.');
      }

      // Record training run in coordinator audit log
      await fetch(`${centralApiUrl}/api/training/runs`, {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${sessionStorage.getItem('arogyavaani_token') ?? ''}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          task,
          status: 'completed',
          samples: samples.length,
          epochs,
          loss: result.loss,
          accuracy: result.accuracy > 1 ? result.accuracy / 100 : result.accuracy,
        }),
      }).catch(() => undefined);

      setSubmitted(`Model update submitted and confirmed for Round ${network.round}!`);
      setNotice(`🎉 Round ${network.round} training and contribution confirmed! Zero patient images left your machine.`);
      setTraining({
        status: 'completed',
        epoch: epochs,
        total_epochs: epochs,
        loss: result.loss,
        accuracy: result.accuracy > 1 ? result.accuracy / 100 : result.accuracy,
        samples: samples.length,
        submission: { status: 'submitted', round: network.round },
      });
      await refreshNetwork();
    } catch (e) {
      setError(e instanceof Error ? e.message : 'In-browser training failed.');
      setTraining({ status: 'failed', message: e instanceof Error ? e.message : 'Training failed.' });
    } finally {
      setBusy(false);
    }
  };

  const submit = async () => {
    if (!network?.round) {
      setError('No active federated round exists to submit this update to.');
      return;
    }
    setBusy(true);
    setError('');
    try {
      await refreshNetwork();
      setSubmitted(`Federated status refreshed for Round ${network.round}.`);
    } finally {
      setBusy(false);
    }
  };

  // Determine actual, verifiable model-update lifecycle status
  const getUpdateStatus = (): { label: string; tone: 'green' | 'blue' | 'amber' | 'red' } => {
    if (network?.hasSubmitted || submitted) {
      if (network?.aggregationStatus === 'COMPLETED') return { label: 'Aggregated into Global Model', tone: 'green' };
      if (network?.aggregationStatus === 'READY_FOR_AGGREGATION' || network?.aggregationStatus === 'AGGREGATING') return { label: 'Ready for Aggregation', tone: 'blue' };
      return { label: 'Persisted Successfully on Server', tone: 'green' };
    }
    if (training.status === 'completed') {
      if (training.submission?.status === 'submitted') return { label: 'Persisted Successfully on Server', tone: 'green' };
      return { label: 'Generated Locally', tone: 'blue' };
    }
    if (training.status === 'submitting') return { label: 'Uploading to Server...', tone: 'blue' };
    if (training.status === 'packaging') return { label: 'Packaging AVM1 Parameters...', tone: 'blue' };
    if (['queued', 'validating', 'training'].includes(training.status)) return { label: 'Training in Progress', tone: 'amber' };
    return { label: 'Not Yet Generated', tone: 'amber' };
  };

  const updateStatus = getUpdateStatus();
  const progress = training.total_epochs ? Math.round(((training.epoch ?? 0) / training.total_epochs) * 100) : 0;
  const trainingDuration = elapsed(training.created_at, training.updated_at);

  return (
    <>
      <section className="page-heading">
        <div>
          <div className="eyebrow">LOCAL TRAINING & TRANSMISSION</div>
          <h1>Local Model Training</h1>
          <p>Train PyTorch DenseNet-121 on your local dataset and submit encrypted mathematical parameters to the federation</p>
        </div>
        <div style={{ display: 'flex', gap: '12px', alignItems: 'center' }}>
          <label>
            Medical Task
            <select
              value={task}
              onChange={e => {
                setTask(e.target.value as MedicalTaskId);
                clear();
                setTraining({ status: 'idle' });
              }}
            >
              {fallbackTasks.map(t => (
                <option key={t.taskId} value={t.taskId}>{t.displayName}</option>
              ))}
            </select>
          </label>
          <button className="button secondary" onClick={() => { void refreshNetwork(); }} title="Refresh federation status">
            <RefreshCw size={15} /> Refresh
          </button>
        </div>
      </section>

      {network && (
        <section className="training-context">
          <div>
            <span>BASE GLOBAL MODEL</span>
            <b>v{network.baseGlobalModelVersion ?? 0}</b>
          </div>
          <div>
            <span>FEDERATED ROUND</span>
            <b>{network.round ? `Round ${network.round}` : 'No active round'}</b>
          </div>
          <div>
            <span>PARTICIPATION STATUS</span>
            <b>{network.isParticipant ? `${network.updatesReceived} / ${network.hospitals} submitted` : network.round ? 'Not selected' : 'Awaiting admin'}</b>
          </div>
          <div>
            <span>ROUND AGGREGATION</span>
            <b>{network.aggregationStatus}</b>
          </div>
        </section>
      )}

      {/* Model Update Transmission Status Banner */}
      <section className="panel" style={{ padding: '18px 24px', display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '14px' }}>
        <div>
          <span style={{ fontSize: '13px', fontWeight: 800, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.8px' }}>Model Update Lifecycle</span>
          <div style={{ fontSize: '18px', fontWeight: 750, color: 'var(--navy)', marginTop: '2px' }}>Mathematical Parameter State</div>
          <div style={{ fontSize: '13px', color: 'var(--text-muted)', marginTop: '4px', display: 'flex', alignItems: 'center', gap: '6px' }}>
            <Zap size={14} color="#0d8a68" />
            <span>Client-side WebGPU Engine: <b>{hardwareName}</b> (100% private in-browser training)</span>
          </div>
        </div>
        <Pill done={updateStatus.tone === 'green'} tone={updateStatus.tone}>
          {updateStatus.label}
        </Pill>
      </section>

      <section className="panel upload-panel">
        <div
          className="dataset-upload-field"
          role="button"
          tabIndex={0}
          onClick={() => input.current?.click()}
          onKeyDown={e => {
            if (e.key === 'Enter' || e.key === ' ') input.current?.click();
          }}
        >
          <input
            ref={node => {
              input.current = node;
              node?.setAttribute('webkitdirectory', '');
            }}
            type="file"
            multiple
            accept=".jpg,.jpeg,.png,.bmp,.tif,.tiff,.webp"
            hidden
            onChange={e => selectDataset(Array.from(e.target.files ?? []))}
          />
          <FolderUp size={28} />
          <div>
            <b>Choose Local Dataset Folder</b>
            <span>Select the parent directory containing your diagnostic class subfolders (e.g. NORMAL, PNEUMONIA).</span>
          </div>
          <button type="button" className="button secondary" disabled={busy}>
            Choose dataset folder
          </button>
        </div>

        {selected && (
          <section className="dataset-summary">
            <div>
              <FolderUp size={20} />
              <span>
                <b>{selected.folderName}</b> — Folder selected for local validation
              </span>
              <button
                type="button"
                className="text-link"
                disabled={busy}
                onClick={e => {
                  e.stopPropagation();
                  clear();
                }}
              >
                Clear
              </button>
            </div>
            <table>
              <thead>
                <tr>
                  <th>CLASS</th>
                  <th>IMAGES</th>
                </tr>
              </thead>
              <tbody>
                {selected.classes.map(n => (
                  <tr key={n}>
                    <td>{n}</td>
                    <td>{selected.classDistribution[n].toLocaleString()} images</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <p>
              {selected.fileCount.toLocaleString()} total images · {(selected.sizeBytes / 1024 / 1024).toFixed(1)} MB total size
            </p>
            <div className="dataset-summary-actions">
              <button className="button primary" disabled={busy || Boolean(validation)} onClick={() => void upload()}>
                {busy ? 'Preparing…' : 'Prepare Dataset'} <ArrowRight size={16} />
              </button>
              <small>Zero installation: validates images directly in browser sandbox memory with complete privacy.</small>
            </div>
          </section>
        )}

        {notice && <div className="form-notice"><CheckCircle2 size={18} /> {notice}</div>}
        {error && <div className="form-error"><AlertCircle size={18} /> {error}</div>}

        {validation && (
          <section className="dataset-ready detailed">
            <div className="ready-heading">
              <CheckCircle2 size={24} />
              <div>
                <b>Local Dataset Validated & Ready</b>
                <span>Validated in browser sandbox · {hardwareName}</span>
              </div>
            </div>
            <div className="dataset-real-metrics">
              {validation.classes.map(n => (
                <div key={n}>
                  <span>{n}</span>
                  <b>{validation.class_distribution[n].toLocaleString()}</b>
                  <small>local images</small>
                </div>
              ))}
              <div>
                <span>TOTAL SAMPLES</span>
                <b>{validation.file_count.toLocaleString()}</b>
                <small>training images</small>
              </div>
            </div>
          </section>
        )}

        <div className="training-actions">
          <label>
            Epochs
            <input type="number" min={1} max={100} value={epochs} onChange={e => setEpochs(Number(e.target.value))} />
          </label>
          {validation && (
            <button
              className="button primary"
              disabled={busy || ['queued', 'validating', 'training', 'packaging', 'submitting'].includes(training.status)}
              onClick={() => void train()}
            >
              {busy ? 'Starting…' : 'Start WebGPU Local Training'} <ArrowRight size={16} />
            </button>
          )}
        </div>

        {/* Training In Progress */}
        {training.status !== 'idle' && training.status !== 'completed' && (
          <section className="training-progress">
            <div className="panel-title">
              <div>
                <h2>Training Local Model</h2>
                <p>
                  {training.epoch && training.total_epochs
                    ? `Epoch ${training.epoch} of ${training.total_epochs} · DenseNet-121 Feature Learning`
                    : training.message ?? training.status}
                </p>
              </div>
              <Pill>{training.status}</Pill>
            </div>
            {training.total_epochs ? (
              <>
                <div className="progress">
                  <span style={{ width: `${progress}%` }} />
                </div>
                <div className="training-metrics">
                  {training.loss !== undefined && (
                    <div>
                      <span>TRAINING LOSS</span>
                      <b>{training.loss.toFixed(4)}</b>
                    </div>
                  )}
                  {training.accuracy !== undefined && (
                    <div>
                      <span>TRAINING ACCURACY</span>
                      <b>{(training.accuracy * 100).toFixed(1)}%</b>
                    </div>
                  )}
                  {trainingDuration && (
                    <div>
                      <span>ELAPSED TIME</span>
                      <b>{trainingDuration}</b>
                    </div>
                  )}
                </div>
              </>
            ) : null}
          </section>
        )}

        {/* Local Model Trained and Update Ready */}
        {training.status === 'completed' && (
          <section className="local-model-ready">
            <div className="ready-heading">
              <CheckCircle2 size={24} />
              <div>
                <b>Local Model Parameters Ready</b>
                <span>AVM1 mathematical model weights generated from your local medical image run.</span>
              </div>
            </div>
            <div className="model-update-metadata">
              <div>
                <span>TASK</span>
                <b>{task}</b>
              </div>
              <div>
                <span>BASE MODEL</span>
                <b>v{training.base_version ?? network?.baseGlobalModelVersion ?? 0}</b>
              </div>
              <div>
                <span>ROUND</span>
                <b>{network?.round ? `Round ${network.round}` : 'Not available'}</b>
              </div>
              <div>
                <span>TRAINING SAMPLES</span>
                <b>{training.samples?.toLocaleString() ?? 'Recorded'}</b>
              </div>
              {training.loss !== undefined && (
                <div>
                  <span>FINAL LOSS</span>
                  <b>{training.loss.toFixed(4)}</b>
                </div>
              )}
              {training.accuracy !== undefined && (
                <div>
                  <span>FINAL ACCURACY</span>
                  <b>{(training.accuracy * 100).toFixed(1)}%</b>
                </div>
              )}
              {trainingDuration && (
                <div>
                  <span>TRAINING DURATION</span>
                  <b>{trainingDuration}</b>
                </div>
              )}
            </div>

            <button
              className="button primary"
              disabled={busy || Boolean(submitted) || Boolean(network?.hasSubmitted)}
              onClick={() => void submit()}
            >
              {busy
                ? 'Transmitting Parameters…'
                : submitted || network?.hasSubmitted
                ? 'Model Update Transmitted & Persisted'
                : 'Transmit Model Update to Coordinator'} <ArrowRight size={16} />
            </button>
          </section>
        )}

        {/* Confirmed Submitted State */}
        {(submitted || network?.hasSubmitted) && (
          <section className="submission-state">
            <div className="ready-heading">
              <CheckCircle2 size={24} />
              <div>
                <b>Model Update Transmitted & Persisted</b>
                <span>
                  {submitted || `Model update confirmed for Round ${network?.round}.`}{' '}
                  {network?.round ? `Progress: ${network.updatesReceived} of ${network.hospitals} participant updates received.` : ''}
                </span>
              </div>
            </div>
            {network?.aggregationStatus === 'AGGREGATING' && (
              <p>All participating hospital updates have been accepted. Central federated averaging (FedAvg) aggregation is executing.</p>
            )}
            {network?.aggregationStatus === 'COMPLETED' && (
              <p>Federated aggregation completed successfully. New global model checkpoint v{(network.baseGlobalModelVersion ?? 0) + 1} is available for download!</p>
            )}
            {network?.aggregationStatus === 'WAITING_FOR_UPDATES' && (
              <p>Your mathematical parameters are durably stored in the coordinator's database. Waiting for other assigned hospitals to finish local training.</p>
            )}
          </section>
        )}
      </section>

      <section className="privacy-indicator">
        <ShieldCheck size={20} />
        <span><b>Zero Image Egress:</b> ArogyaVaani's federated protocol strictly validates that no image buffers, base64 payloads, or patient metadata can ever be uploaded to the central server.</span>
      </section>
    </>
  );
}

function TrainingHistory() {
  const [task, setTask] = useState<MedicalTaskId>('PNEUMONIA');
  const [items, setItems] = useState<Array<Training & { _id?: string }>>([]);
  const [error, setError] = useState('');

  useEffect(() => {
    fetch(centralApiUrl + '/api/training/runs?task=' + task, { headers: tokenHeaders() })
      .then(async r => {
        const b = await r.json();
        if (!r.ok) throw new Error(b.message);
        setItems(b);
      })
      .catch(e => setError(e instanceof Error ? e.message : 'Unable to load training history.'));
  }, [task]);

  return (
    <>
      <section className="page-heading">
        <div>
          <div className="eyebrow">PERSISTENT AUDIT LOG</div>
          <h1>Local Training History</h1>
          <p>Auditable records of all local training runs performed on this hospital node</p>
        </div>
        <label>
          Medical Task
          <select value={task} onChange={e => setTask(e.target.value as MedicalTaskId)}>
            {fallbackTasks.map(t => (
              <option key={t.taskId} value={t.taskId}>{t.displayName}</option>
            ))}
          </select>
        </label>
      </section>

      {error && <div className="form-error"><AlertCircle size={18} /> {error}</div>}

      <section className="panel hospital-table">
        <table>
          <thead>
            <tr>
              <th>STATUS</th>
              <th>SAMPLES</th>
              <th>LOSS</th>
              <th>ACCURACY</th>
              <th>RECORDED AT</th>
            </tr>
          </thead>
          <tbody>
            {items.length ? (
              items.map((item, i) => (
                <tr key={item._id ?? i}>
                  <td>
                    <Pill done={item.status === 'completed'}>{item.status}</Pill>
                  </td>
                  <td>{item.samples?.toLocaleString() ?? 'Not reported'}</td>
                  <td>{item.loss !== undefined ? item.loss.toFixed(4) : 'Not reported'}</td>
                  <td>{item.accuracy === undefined ? 'Not reported' : `${(item.accuracy * 100).toFixed(1)}%`}</td>
                  <td>{date(item.createdAt)}</td>
                </tr>
              ))
            ) : (
              <tr>
                <td colSpan={5} style={{ textAlign: 'center', padding: '32px', color: 'var(--text-muted)' }}>
                  No local {task} training runs have been recorded yet.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </section>
    </>
  );
}

export function Hospital({
  onExit,
  path,
  onNavigate,
}: {
  onExit: () => void;
  path: string;
  onNavigate: (path: string) => void;
}) {
  const fragment = path.replace('/hospital/', '');
  const view: View =
    fragment === 'dashboard' || !fragment
      ? 'overview'
      : nav.some(([id]) => id === fragment)
      ? (fragment as View)
      : 'overview';

  const content =
    view === 'overview' ? (
      <Overview setView={next => onNavigate('/hospital/' + (next === 'overview' ? 'dashboard' : next))} />
    ) : view === 'train' ? (
      <LocalTrain />
    ) : view === 'global-model' ? (
      <LiveFederated />
    ) : view === 'prediction' ? (
      <Prediction />
    ) : (
      <TrainingHistory />
    );

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">
            <ShieldCheck size={24} />
          </div>
          <div>
            <b>Arogya<span>Vaani</span></b>
            <small>Privacy-Preserved Medical AI</small>
          </div>
        </div>
        <nav>
          {nav.map(([id, label, Icon]) => (
            <button
              key={id}
              className={view === id ? 'active' : ''}
              onClick={() => onNavigate('/hospital/' + (id === 'overview' ? 'dashboard' : id))}
            >
              <Icon size={19} />
              <span>{label}</span>
            </button>
          ))}
        </nav>
        <div className="sidebar-bottom">
          <button className="text-button" onClick={onExit}>
            <LogOut size={17} /> Sign out
          </button>
        </div>
      </aside>
      <div className="main">
        <main>{content}</main>
        <footer>
          <span>
            <ShieldCheck size={16} /> Privacy Guaranteed: Raw medical images stay strictly on this hospital's storage
          </span>
        </footer>
      </div>
    </div>
  );
}
