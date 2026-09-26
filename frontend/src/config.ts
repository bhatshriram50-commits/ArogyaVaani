export const centralApiUrl = (import.meta.env.VITE_API_URL as string | undefined) ?? 'http://localhost:4000';
export const localMlUrl = (import.meta.env.VITE_LOCAL_ML_URL as string | undefined) ?? 'http://127.0.0.1:8000';

export const tokenHeaders = () => ({ Authorization: `Bearer ${sessionStorage.getItem('arogyavaani_token') ?? ''}` });
