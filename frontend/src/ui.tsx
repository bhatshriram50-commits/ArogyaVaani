export function Badge({ children, tone = 'green' }: { children: React.ReactNode; tone?: 'green' | 'amber' | 'slate' }) { return <span className={`badge ${tone}`}><span />{children}</span>; }
