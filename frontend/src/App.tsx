import { useEffect, useState } from 'react';
import { Admin } from './Admin';
import { Hospital } from './Hospital';
import { centralApiUrl, tokenHeaders } from './config';
import { Auth, Landing } from './Public';

type Role = 'hospital' | 'admin';
const go = (path: string) => { window.history.pushState({}, '', path); window.dispatchEvent(new PopStateEvent('popstate')); };
const authenticatedPath = (role: Role) => role === 'admin' ? '/admin/dashboard' : '/hospital/dashboard';

export default function App() {
  const storedRole = sessionStorage.getItem('arogyavaani_role') as Role | null;
  const [role, setRole] = useState<Role | null>(storedRole); const [path, setPath] = useState(window.location.pathname); const [checking, setChecking] = useState(Boolean(sessionStorage.getItem('arogyavaani_token')));
  const clearSession = () => { sessionStorage.removeItem('arogyavaani_token'); sessionStorage.removeItem('arogyavaani_role'); setRole(null); setChecking(false); };
  useEffect(() => { const updatePath = () => setPath(window.location.pathname); const expire = () => { clearSession(); go('/login'); }; window.addEventListener('popstate', updatePath); window.addEventListener('arogyavaani:session-expired', expire); return () => { window.removeEventListener('popstate', updatePath); window.removeEventListener('arogyavaani:session-expired', expire); }; }, []);
  useEffect(() => { const token = sessionStorage.getItem('arogyavaani_token'); if (!token) { setChecking(false); return; } fetch(centralApiUrl + '/api/auth/session', { headers: tokenHeaders() }).then(async response => { if (!response.ok) throw new Error(); const data = await response.json() as { role: Role }; sessionStorage.setItem('arogyavaani_role', data.role); setRole(data.role); if (!window.location.pathname.startsWith('/' + data.role)) go(authenticatedPath(data.role)); }).catch(clearSession).finally(() => setChecking(false)); }, []);
  const logout = () => { fetch(centralApiUrl + '/api/auth/logout', { method: 'POST', headers: tokenHeaders() }).catch(() => undefined).finally(() => { clearSession(); go('/'); }); };
  const signIn = (token: string, nextRole: Role) => { sessionStorage.setItem('arogyavaani_token', token); sessionStorage.setItem('arogyavaani_role', nextRole); setRole(nextRole); setChecking(false); go(authenticatedPath(nextRole)); };
  if (checking) return <div className="app-loading">Restoring your secure session…</div>;
  if (!role) { if (path === '/login') return <Auth onBack={() => go('/')} onSuccess={signIn} />; if (path === '/signup') return <Auth startInSignup onBack={() => go('/')} onSuccess={signIn} />; return <Landing onLogin={() => go('/login')} />; }
  if (!path.startsWith('/' + role)) { go(authenticatedPath(role)); return null; }
  return role === 'admin' ? <Admin onExit={logout} path={path} onNavigate={go} /> : <Hospital onExit={logout} path={path} onNavigate={go} />;
}
