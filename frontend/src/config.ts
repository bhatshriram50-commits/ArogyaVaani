export const centralApiUrl = (import.meta.env.VITE_API_URL as string | undefined) ?? 'http://localhost:4000';
export const localMlUrl = (import.meta.env.VITE_LOCAL_ML_URL as string | undefined) ?? 'http://127.0.0.1:8000';

export const tokenHeaders = () => ({ Authorization: `Bearer ${sessionStorage.getItem('arogyavaani_token') ?? ''}` });
// This is intended only for a protected hospital-network deployment. Never put
// a central API secret here: VITE_ values are deliberately visible to browsers.
export const localServiceHeaders = (): Record<string, string> => {
  const token = import.meta.env.VITE_LOCAL_SERVICE_TOKEN as string | undefined;
  return token ? { 'X-ArogyaVaani-Local-Token': token } : {};
};
