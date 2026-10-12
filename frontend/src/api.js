const BASE = import.meta.env.VITE_API_URL || "http://127.0.0.1:8001";
async function request(path, options={}) {
  const res = await fetch(`${BASE}${path}`, options);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || data.message || `Request failed (${res.status})`);
  return data;
}
export const registerStudent = (data) => request('/register-student', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});
export const loginStudent = (data) => request('/login-student', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});
export const getLetters = () => request('/letters');
export const submitLetter = (data, attachment) => { const f=new FormData(); Object.entries(data).forEach(([k,v])=>f.append(k,String(v ?? ''))); if(attachment) f.append('attachment', attachment); return request('/letters/submit',{method:'POST',body:f}); };
export const updateLetterStatus = (id,status) => { const f=new FormData(); f.append('status',status); return request(`/letters/${id}/status`,{method:'PATCH',body:f}); };
export const verifyIdentity = (data) => request('/verification/verify',{method:'POST',body:data});
