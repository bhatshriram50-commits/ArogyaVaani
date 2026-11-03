import React from 'react';
import { CheckCircle2, Clock, AlertTriangle, AlertCircle, Info, Loader2 } from 'lucide-react';

export type Tone = 'green' | 'amber' | 'slate' | 'blue' | 'red';

export function Badge({ children, tone = 'green' }: { children: React.ReactNode; tone?: Tone }) {
  return (
    <span className={`badge ${tone}`}>
      <span className="badge-dot" />
      {children}
    </span>
  );
}

export function StatusPill({
  children,
  done,
  tone,
}: {
  children: React.ReactNode;
  done?: boolean;
  tone?: Tone;
}) {
  const finalTone = tone ?? (done ? 'green' : 'amber');
  return (
    <span className={`status-pill ${finalTone}`}>
      {finalTone === 'green' ? (
        <CheckCircle2 size={13} className="pill-icon" />
      ) : finalTone === 'red' ? (
        <AlertCircle size={13} className="pill-icon" />
      ) : (
        <Clock size={13} className="pill-icon" />
      )}
      <span>{children}</span>
    </span>
  );
}

export function StatCard({
  label,
  value,
  detail,
  icon: Icon,
  badge,
  indicator = 'green',
}: {
  label: string;
  value: React.ReactNode;
  detail?: string;
  icon?: React.ComponentType<{ size?: number | string; className?: string }>;
  badge?: React.ReactNode;
  indicator?: 'green' | 'blue' | 'amber' | 'red';
}) {
  return (
    <div className="stat-card">
      <div className="stat-card-header">
        <span className="stat-card-label">{label}</span>
        {Icon ? (
          <div className={`stat-card-icon-wrap icon-${indicator}`}>
            <Icon size={16} />
          </div>
        ) : (
          <span className={`metric-dot ${indicator}`} />
        )}
      </div>
      <div className="stat-card-body">
        <div className="stat-card-value">{value}</div>
        {detail && <div className="stat-card-detail">{detail}</div>}
      </div>
      {badge && <div className="stat-card-footer">{badge}</div>}
    </div>
  );
}

export function ProgressBar({
  progress,
  height = 8,
  tone = 'teal',
}: {
  progress: number;
  height?: number;
  tone?: 'teal' | 'blue' | 'amber';
}) {
  const clamped = Math.max(0, Math.min(100, Math.round(progress)));
  return (
    <div className={`progress-bar-track tone-${tone}`} style={{ height }}>
      <div
        className="progress-bar-fill"
        style={{ width: `${clamped}%` }}
        role="progressbar"
        aria-valuenow={clamped}
        aria-valuemin={0}
        aria-valuemax={100}
      />
    </div>
  );
}

export function Alert({
  type = 'info',
  title,
  children,
  onDismiss,
}: {
  type?: 'error' | 'notice' | 'success' | 'warning' | 'info';
  title?: string;
  children: React.ReactNode;
  onDismiss?: () => void;
}) {
  const Icon =
    type === 'error'
      ? AlertCircle
      : type === 'warning'
      ? AlertTriangle
      : type === 'success' || type === 'notice'
      ? CheckCircle2
      : Info;

  return (
    <div className={`alert-banner alert-${type}`} role="alert">
      <Icon size={18} className="alert-icon" />
      <div className="alert-content">
        {title && <div className="alert-title">{title}</div>}
        <div className="alert-message">{children}</div>
      </div>
      {onDismiss && (
        <button type="button" className="alert-close" onClick={onDismiss} aria-label="Dismiss alert">
          &times;
        </button>
      )}
    </div>
  );
}

export type StepItem = {
  id: string;
  label: string;
  description?: string;
  status: 'upcoming' | 'current' | 'completed' | 'error';
};

export function Stepper({ steps }: { steps: StepItem[] }) {
  return (
    <div className="workflow-stepper">
      {steps.map((step, idx) => (
        <div key={step.id} className={`stepper-item ${step.status}`}>
          <div className="stepper-track-line" />
          <div className="stepper-node">
            {step.status === 'completed' ? (
              <CheckCircle2 size={16} />
            ) : step.status === 'current' ? (
              <span className="stepper-pulse" />
            ) : step.status === 'error' ? (
              <AlertCircle size={16} />
            ) : (
              <span className="stepper-number">{idx + 1}</span>
            )}
          </div>
          <div className="stepper-meta">
            <div className="stepper-label">{step.label}</div>
            {step.description && <div className="stepper-desc">{step.description}</div>}
          </div>
        </div>
      ))}
    </div>
  );
}

export function EmptyState({
  icon: Icon = Info,
  title,
  description,
  action,
}: {
  icon?: React.ComponentType<{ size?: number | string; className?: string }>;
  title: string;
  description?: string;
  action?: React.ReactNode;
}) {
  return (
    <div className="empty-state-card">
      <div className="empty-state-icon">
        <Icon size={32} />
      </div>
      <h3 className="empty-state-title">{title}</h3>
      {description && <p className="empty-state-desc">{description}</p>}
      {action && <div className="empty-state-action">{action}</div>}
    </div>
  );
}

export function Spinner({ size = 18, className = '' }: { size?: number; className?: string }) {
  return <Loader2 size={size} className={`spin-animate ${className}`} />;
}
