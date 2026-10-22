// Typed client for the NestJS backend (the UI never calls the AI service directly).

export interface Booking {
  bookingId: string
  patientId: string
  doctorId: string
  doctorName: string
  specialty: string
  treatmentId: string
  treatmentName: string
  clinicName: string
  bookingDate: string
  bookingTime: string
  status: string
  notes: string
  source: string
  createdAt: string
}

export interface Patient {
  patientId: string
  fullName: string
  dob: string
  email: string
  phone: string
  allergies: string[]
  emergencyContact: string
  treatmentHistory: { recordId: string; visitDate: string; diagnosis: string; doctorName: string }[]
}

export interface ToolCall {
  name: string
  transport: string
  args: Record<string, unknown>
  ok: boolean
  ms: number
}

export interface TraceStep {
  step: string
  detail: string
  ms: number
}

export interface ChatResponse {
  reply: string
  session_id: string
  intent: string
  confidence: number
  booking_draft: Record<string, string> | null
  awaiting_confirmation: boolean
  booking_created: Booking | null
  citations: string[]
  tool_calls: ToolCall[]
  trace: TraceStep[]
  usage: { input_tokens: number; output_tokens: number; llm_calls: number }
  latency_ms: number
}

export interface MedicalSummary {
  chief_complaint: string
  diagnosis: string
  medication: string[]
  doctor_instructions: string[]
  allergies_mentioned: string[]
  follow_up: string | null
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`/api${path}`, {
    ...init,
    headers: { 'content-type': 'application/json', ...(init?.headers ?? {}) },
  })
  const body = await res.json().catch(() => ({}))
  if (!res.ok) {
    throw new Error(body?.error?.message ?? `Request failed (${res.status})`)
  }
  return body as T
}

export const api = {
  patient: (id: string) => request<Patient>(`/patients/${id}`),
  bookings: (patientId: string) => request<Booking[]>(`/bookings?patientId=${encodeURIComponent(patientId)}`),
  chat: (message: string, sessionId: string, patientId: string) =>
    request<ChatResponse>('/chat', { method: 'POST', body: JSON.stringify({ message, sessionId, patientId }) }),
  summarize: (transcript: string) =>
    request<{ summary: MedicalSummary; latency_ms: number }>('/summarize', {
      method: 'POST',
      body: JSON.stringify({ transcript }),
    }),
}
