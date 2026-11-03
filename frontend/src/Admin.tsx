import { useEffect, useState, type ReactNode } from 'react';
import { Activity, ClipboardList, Globe2, HeartPulse, History, Network, RefreshCw, ShieldCheck, Users, Play, AlertCircle, CheckCircle2 } from 'lucide-react';
import { centralApiUrl, tokenHeaders } from './config';
import { fallbackTasks, type MedicalTask, type MedicalTaskId } from './medicalTasks';

type View = 'dashboard' | 'hospitals' | 'federated-rounds' | 'models' | 'performance' | 'audit-logs';
type Hospital = { hospitalName?: string; email: string; createdAt?: string };
type Round = {
  task: MedicalTaskId;
  round: number;
  status: string;
  baseGlobalModelVersion: number;
  resultingGlobalModelVersion?: number;
  expectedParticipantCount: number;
  submittedParticipantCount: number;
  remainingParticipantCount: number;
  participantHospitalIds: string[];
  submittedHospitalIds: string[];
  completedAt?: string;
  aggregationError?: string;
};
type Dashboard = {
  totalHospitals: number;
  globalVersion: number;
  activeRound?: Round;
  updatesReceived?: number;
};
type Model = {
  task: MedicalTaskId;
  version: number;
  round: number;
  samples: number;
  hospitals: number;
  status: string;
  architecture?: string;
  checksum?: string;
  createdAt?: string;
};
type Update = {
  task: MedicalTaskId;
  hospitalId: string;
  round: number;
  samples: number;
  baseVersion?: number;
  receivedAt?: string;
  payloadBytes?: number;
  checksum?: string;
  status?: string;
  architecture?: string;
  metrics?: { accuracy?: number; loss?: number };
};
type Audit = {
  actorId: string;
  actorRole: string;
  action: string;
  target?: string;
  createdAt?: string;
  details?: { task?: MedicalTaskId };
};

const nav: Array<[View, string, typeof Activity]> = [
  ['dashboard', 'Overview', Activity],
  ['hospitals', 'Connected Hospitals', Users],
  ['federated-rounds', 'Federated Rounds', Network],
  ['models', 'Global Models', Globe2],
  ['performance', 'Confirmed Updates', ClipboardList],
  ['audit-logs', 'Audit Logs', History],
];

const api = async <T,>(path: string, options: RequestInit = {}) => {
  const r = await fetch(centralApiUrl + path, {
    ...options,
    headers: { ...options.headers, ...tokenHeaders() },
  });
  const b = r.status === 204 ? undefined : await r.json();
  if (!r.ok) throw new Error(b?.message ?? 'Request failed.');
  return b as T;
};

const date = (value?: string) => (value ? new Date(value).toLocaleString() : 'Not available');

const Table = ({ headings, rows }: { headings: string[]; rows: ReactNode[][] }) => (
  <section className="panel hospital-table">
    <table>
      <thead>
        <tr>
          {headings.map(h => (
            <th key={h}>{h}</th>
          ))}
        </tr>
      </thead>
      <tbody>
        {rows.map((row, i) => (
          <tr key={i}>
            {row.map((cell, j) => (
              <td key={j}>{cell}</td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  </section>
);

export function Admin({
  onExit,
  path,
  onNavigate,
}: {
  onExit: () => void;
  path: string;
  onNavigate: (path: string) => void;
}) {
  const requested = path.replace('/admin/', '') as View;
  const view = nav.some(([id]) => id === requested) ? requested : 'dashboard';
  const [task, setTask] = useState<MedicalTaskId>('PNEUMONIA');
  const [tasks, setTasks] = useState<MedicalTask[]>(fallbackTasks);
  const [hospitals, setHospitals] = useState<Hospital[]>([]);
  const [dashboard, setDashboard] = useState<Dashboard | null>(null);
  const [rounds, setRounds] = useState<Round[]>([]);
  const [models, setModels] = useState<Model[]>([]);
  const [updates, setUpdates] = useState<Update[]>([]);
  const [audits, setAudits] = useState<Audit[]>([]);
  const [selected, setSelected] = useState<string[]>([]);
  const [error, setError] = useState('');
  const [successNotice, setSuccessNotice] = useState('');
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const query = '?task=' + task;

  const hospitalName = (id: string) => hospitals.find(h => h.email === id)?.hospitalName?.trim() || id;

  const refresh = async (background = false) => {
    if (!background) {
      setLoading(true);
      setError('');
    }
    try {
      const [fetchedTasks, fetchedHospitals] = await Promise.all([
        api<MedicalTask[]>('/api/medical-tasks'),
        api<Hospital[]>('/api/admin/hospitals'),
      ]);
      setTasks(fetchedTasks);
      setHospitals(fetchedHospitals);

      if (view === 'dashboard') {
        const [d, r, u] = await Promise.all([
          api<Dashboard>('/api/admin/dashboard' + query),
          api<Round[]>('/api/admin/federated-rounds' + query),
          api<Update[]>('/api/admin/model-updates' + query),
        ]);
        setDashboard(d);
        setRounds(r);
        setUpdates(u);
      } else if (view === 'hospitals') {
        // Hospitals already loaded above
      } else if (view === 'federated-rounds') {
        setRounds(await api('/api/admin/federated-rounds' + query));
      } else if (view === 'models') {
        setModels(await api('/api/admin/models' + query));
      } else if (view === 'performance') {
        setUpdates(await api('/api/admin/model-updates' + query));
      } else {
        setAudits(await api('/api/admin/audit-logs' + query));
      }
    } catch (cause) {
      if (!background) setError(cause instanceof Error ? cause.message : 'Unable to load administrator data.');
    } finally {
      if (!background) setLoading(false);
    }
  };

  useEffect(() => {
    void refresh();
    const timer = ['dashboard', 'federated-rounds', 'performance', 'models'].includes(view)
      ? window.setInterval(() => void refresh(true), 5000)
      : undefined;
    return () => {
      if (timer) window.clearInterval(timer);
    };
  }, [view, task]);

  const createRound = async () => {
    if (!selected.length) return setError('Select at least one registered hospital to participate.');
    setBusy(true);
    setError('');
    setSuccessNotice('');
    try {
      await api('/api/federated/rounds', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ task, participantHospitalIds: selected }),
      });
      setSelected([]);
      setSuccessNotice(`Federated learning round created for ${task} with ${selected.length} hospital participants.`);
      await refresh();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Unable to create round.');
    } finally {
      setBusy(false);
    }
  };

  const cancel = async (round: number) => {
    setBusy(true);
    setError('');
    setSuccessNotice('');
    try {
      await api(`/api/federated/rounds/${round}/cancel${query}`, { method: 'POST' });
      setSuccessNotice(`Round ${round} cancelled.`);
      await refresh();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Unable to cancel round.');
    } finally {
      setBusy(false);
    }
  };

  const triggerAggregation = async (round: number) => {
    setBusy(true);
    setError('');
    setSuccessNotice('');
    try {
      await api(`/api/federated/rounds/${round}/aggregate${query}`, { method: 'POST' });
      setSuccessNotice(`Federated averaging aggregation triggered for Round ${round}.`);
      await refresh();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Unable to trigger aggregation.');
    } finally {
      setBusy(false);
    }
  };

  const active = rounds.find(round => !['COMPLETED', 'CANCELLED', 'CLOSED'].includes(round.status)) ?? dashboard?.activeRound;
  const progress = active?.expectedParticipantCount
    ? Math.round((active.submittedParticipantCount / active.expectedParticipantCount) * 100)
    : 0;

  const updateRows = updates.length
    ? updates.slice(0, 25).map(update => [
        <div key={update.hospitalId} style={{ display: 'flex', flexDirection: 'column' }}>
          <strong>{hospitalName(update.hospitalId)}</strong>
          <small>{update.hospitalId}</small>
        </div>,
        String(update.round),
        update.samples.toLocaleString(),
        `v${update.baseVersion ?? 0}`,
        date(update.receivedAt),
        update.payloadBytes ? `${(update.payloadBytes / 1024 / 1024).toFixed(1)} MB (AVM1)` : 'Stored',
        <span key="status" className="status-pill success">
          <CheckCircle2 size={14} /> Stored
        </span>,
      ])
    : [['No centrally confirmed parameter submissions for this task yet.', '—', '—', '—', '—', '—', '—']];

  let content: ReactNode;
  if (loading) {
    content = <section className="panel empty-state">Loading real-time federation data…</section>;
  } else if (view === 'dashboard') {
    content = (
      <>
        <section className="metrics-grid">
          <div className="metric">
            <div className="metric-label">REGISTERED HOSPITALS</div>
            <strong>{dashboard?.totalHospitals ?? hospitals.length}</strong>
            <small>Total network authorized accounts</small>
          </div>
          <div className="metric">
            <div className="metric-label">{task} ROUND</div>
            <strong>{active ? `Round ${active.round}` : 'None'}</strong>
            <small>{active?.status ?? 'No active round running'}</small>
          </div>
          <div className="metric">
            <div className="metric-label">GLOBAL MODEL</div>
            <strong>v{dashboard?.globalVersion ?? 0}</strong>
            <small>Immutable checkpoint per task</small>
          </div>
          <div className="metric">
            <div className="metric-label">ROUND PROGRESS</div>
            <strong>{active ? `${active.submittedParticipantCount} / ${active.expectedParticipantCount}` : '—'}</strong>
            <small>{active ? `${active.remainingParticipantCount} updates remaining` : 'Idle coordinator'}</small>
          </div>
        </section>

        <section className="panel">
          <div className="panel-title">
            <div>
              <h2>{active ? `${task} · Round ${active.round}` : `Create New Federated Round · ${task}`}</h2>
              <p>
                {active
                  ? `Base model v${active.baseGlobalModelVersion} · ${active.submittedParticipantCount} of ${active.expectedParticipantCount} updates submitted`
                  : 'Select participating hospitals to freeze participant roster for this round.'}
              </p>
            </div>
            {!active && (
              <button
                className="button primary"
                disabled={busy || !selected.length}
                onClick={() => void createRound()}
              >
                {busy ? 'Creating…' : `Start Round with ${selected.length} Hospitals`}
              </button>
            )}
          </div>

          {active ? (
            <>
              <div className="progress">
                <span style={{ width: `${progress}%` }} />
              </div>

              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', margin: '14px 0 8px' }}>
                <span style={{ fontSize: '15px', fontWeight: 700, color: 'var(--navy)' }}>Selected Round Participants ({active.participantHospitalIds.length})</span>
                <span style={{ fontSize: '14px', color: 'var(--text-muted)' }}>{active.submittedParticipantCount} Submitted, {active.remainingParticipantCount} Pending</span>
              </div>

              <div className="participant-list admin-participants">
                {active.participantHospitalIds.map(id => {
                  const hasSubmitted = active.submittedHospitalIds.includes(id);
                  return (
                    <div key={id}>
                      <div>
                        <strong>{hospitalName(id)}</strong>
                        <small style={{ display: 'block', color: 'var(--text-muted)', marginTop: '2px' }}>{id}</small>
                      </div>
                      <span className={`status-pill ${hasSubmitted ? 'success' : ''}`}>
                        {hasSubmitted ? <CheckCircle2 size={14} /> : null}
                        {hasSubmitted ? 'Parameters Submitted' : 'Waiting for Local Run'}
                      </span>
                    </div>
                  );
                })}
              </div>

              {active.aggregationError && (
                <div className="form-error" style={{ marginTop: '14px' }}>
                  <AlertCircle size={18} /> Aggregation notice: {active.aggregationError}
                </div>
              )}

              <div className="button-row" style={{ marginTop: '16px' }}>
                {active.status === 'READY_FOR_AGGREGATION' && (
                  <button className="button primary" disabled={busy} onClick={() => void triggerAggregation(active.round)}>
                    <Play size={16} /> Run Federated Aggregation Now
                  </button>
                )}
                {['ACTIVE', 'WAITING_FOR_UPDATES', 'READY_FOR_AGGREGATION'].includes(active.status) && (
                  <button className="button secondary" disabled={busy} onClick={() => void cancel(active.round)}>
                    Cancel Incomplete Round
                  </button>
                )}
              </div>
            </>
          ) : (
            <Table
              headings={['Select', 'Hospital Name', 'Email Account', 'Registered Date']}
              rows={hospitals.map(h => [
                <input
                  key={h.email}
                  type="checkbox"
                  style={{ width: '18px', height: '18px', cursor: 'pointer' }}
                  checked={selected.includes(h.email)}
                  onChange={() =>
                    setSelected(value =>
                      value.includes(h.email) ? value.filter(email => email !== h.email) : [...value, h.email]
                    )
                  }
                />,
                h.hospitalName ?? 'Not provided',
                h.email,
                date(h.createdAt),
              ])}
            />
          )}
        </section>

        <section className="panel">
          <div className="panel-title">
            <div>
              <h2>Confirmed Model Updates ({updates.length})</h2>
              <p>Durably persisted AVM1 mathematical parameter artifacts received by the central server</p>
            </div>
          </div>
          <Table headings={['Hospital', 'Round', 'Samples', 'Base Model', 'Received At', 'Artifact Size', 'Verification']} rows={updateRows} />
        </section>
      </>
    );
  } else if (view === 'hospitals') {
    content = (
      <Table
        headings={['Hospital Name', 'Email Account', 'Registration Timestamp']}
        rows={hospitals.map(h => [h.hospitalName ?? 'Not provided', h.email, date(h.createdAt)])}
      />
    );
  } else if (view === 'federated-rounds') {
    content = (
      <Table
        headings={['Medical Task', 'Round', 'Base Model', 'Submitted / Expected', 'Status', 'Completed / Result']}
        rows={rounds.map(round => [
          round.task,
          `Round ${round.round}`,
          `v${round.baseGlobalModelVersion}`,
          `${round.submittedParticipantCount} / ${round.expectedParticipantCount}`,
          <span key={round.round} className={`status-pill ${round.status === 'COMPLETED' ? 'success' : round.status === 'CANCELLED' ? 'red' : 'blue'}`}>
            {round.status}
          </span>,
          round.resultingGlobalModelVersion ? `Global Model v${round.resultingGlobalModelVersion}` : date(round.completedAt),
        ])}
      />
    );
  } else if (view === 'models') {
    content = (
      <Table
        headings={['Medical Task', 'Version', 'From Round', 'Contributing Hospitals', 'Total Samples', 'Model Status']}
        rows={models.map(model => [
          model.task,
          `v${model.version}`,
          `Round ${model.round}`,
          `${model.hospitals} hospitals`,
          model.samples.toLocaleString(),
          <span key={model.version} className={`status-pill ${model.status === 'ACTIVE' ? 'success' : ''}`}>
            {model.status}
          </span>,
        ])}
      />
    );
  } else if (view === 'performance') {
    content = (
      <section className="panel">
        <div className="panel-title">
          <div>
            <h2>Confirmed Model Updates ({updates.length})</h2>
            <p>Cryptographically validated mathematical model weights stored in MongoDB and object artifact storage</p>
          </div>
        </div>
        <Table headings={['Hospital', 'Round', 'Samples', 'Base Model', 'Received At', 'Artifact Size', 'Verification']} rows={updateRows} />
      </section>
    );
  } else {
    content = (
      <Table
        headings={['Timestamp', 'Task', 'Actor Email', 'Role', 'Action', 'Target Entity']}
        rows={audits.map(audit => [
          date(audit.createdAt),
          audit.details?.task ?? task,
          audit.actorId,
          audit.actorRole,
          audit.action,
          audit.target ?? '—',
        ])}
      />
    );
  }

  return (
    <div className="admin-shell">
      <header className="admin-top">
        <div className="brand">
          <div className="brand-mark">
            <HeartPulse size={24} />
          </div>
          <div>
            <b>Arogya<span>Vaani</span></b>
            <small>Federation Network Operations Center</small>
          </div>
        </div>
        <div className="admin-user">
          <b>Administrator Session</b>
          <button onClick={onExit}>Sign out</button>
        </div>
      </header>
      <div className="admin-tabs">
        {nav.map(([id, label, Icon]) => (
          <button className={view === id ? 'active' : ''} key={id} onClick={() => onNavigate(`/admin/${id}`)}>
            <Icon size={17} />
            <span>{label}</span>
          </button>
        ))}
      </div>
      <main className="admin-main">
        <section className="page-heading">
          <div>
            <div className="eyebrow">CENTRAL COORDINATOR</div>
            <h1>{task} Operations</h1>
            <p>Manage federation rounds, inspect received parameters, and trigger aggregation</p>
          </div>
          <div style={{ display: 'flex', gap: '14px', alignItems: 'center' }}>
            <label>
              Medical Task
              <select
                value={task}
                onChange={event => {
                  setTask(event.target.value as MedicalTaskId);
                  setSelected([]);
                }}
              >
                {tasks.map(item => (
                  <option value={item.taskId} key={item.taskId}>{item.displayName}</option>
                ))}
              </select>
            </label>
            <button className="button secondary" onClick={() => void refresh()}>
              <RefreshCw size={16} /> Refresh
            </button>
          </div>
        </section>

        {successNotice && <div className="form-notice"><CheckCircle2 size={18} /> {successNotice}</div>}
        {error && <div className="form-error"><AlertCircle size={18} /> {error}</div>}

        {content}

        <div className="admin-privacy">
          <ShieldCheck size={22} />
          <span>
            <b>Privacy-Preserved Monitoring:</b> Administrators inspect strictly mathematical updates (weight matrices, shapes, sample counts) and coordination audit trails. Raw patient images remain exclusively at hospital edge nodes.
          </span>
        </div>
      </main>
    </div>
  );
}
