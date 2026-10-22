import { useCallback, useEffect, useRef, useState } from 'react'
import { api, type Booking, type ChatResponse, type MedicalSummary, type Patient } from './api'

const PATIENT_ID = 'P001' // single-patient demo, as specified in the assignment

const QUICK_PROMPTS = [
  'What are the opening hours of Metro Care?',
  "I've had an itchy rash on my arm for a week.",
  'When is my next appointment?',
  'Book me with her next Monday at 10am.',
  'Can I book an appointment for my peanut allergy?',
]

const SAMPLE_TRANSCRIPT = `Doctor: Good morning, what brings you in today?
Patient: I've had a sore throat and a mild fever for three days.
Doctor: Any allergies?
Patient: Yes, penicillin.
Doctor: Your throat looks red but there's no sign of a bacterial infection. It's most likely a viral pharyngitis.
Doctor: Take paracetamol 500mg every 6 hours as needed, drink plenty of fluids and rest.
Doctor: If the fever lasts more than five days or you have trouble swallowing, come back. Otherwise no follow-up needed.`

interface Message {
  role: 'user' | 'assistant'
  text: string
  meta?: ChatResponse
  error?: boolean
}

const newSessionId = () => `web-${Math.random().toString(36).slice(2, 10)}`

function renderText(text: string) {
  // Make source citations like [D103] visually distinct.
  return text.split(/(\[[CDTP]\d{3,}\])/g).map((part, i) =>
    /^\[[CDTP]\d{3,}\]$/.test(part) ? (
      <span key={i} className="cite">
        {part.slice(1, -1)}
      </span>
    ) : (
      <span key={i}>{part}</span>
    ),
  )
}

export default function App() {
  const [sessionId, setSessionId] = useState(newSessionId)
  const [patient, setPatient] = useState<Patient | null>(null)
  const [bookings, setBookings] = useState<Booking[]>([])
  const [messages, setMessages] = useState<Message[]>([
    {
      role: 'assistant',
      text: "Hi John, I'm ClinicPilot. I can answer questions about our clinics and doctors, suggest the right service, check your appointments and book a new one for you.",
    },
  ])
  const [input, setInput] = useState('')
  const [busy, setBusy] = useState(false)
  const [lastTurn, setLastTurn] = useState<ChatResponse | null>(null)
  const [highlight, setHighlight] = useState<string | null>(null)
  const [tab, setTab] = useState<'chat' | 'summary'>('chat')
  const [bookingsError, setBookingsError] = useState<string | null>(null)
  const endRef = useRef<HTMLDivElement>(null)

  const refreshBookings = useCallback(async () => {
    try {
      setBookings(await api.bookings())
      setBookingsError(null)
    } catch (e) {
      setBookingsError((e as Error).message)
    }
  }, [])

  useEffect(() => {
    api.patient(PATIENT_ID).then(setPatient).catch(() => setPatient(null))
    void refreshBookings()
    const timer = setInterval(refreshBookings, 3000) // live view of the database
    return () => clearInterval(timer)
  }, [refreshBookings])

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, busy])

  async function send(text: string) {
    const message = text.trim()
    if (!message || busy) return
    setInput('')
    setMessages((m) => [...m, { role: 'user', text: message }])
    setBusy(true)
    try {
      const res = await api.chat(message, sessionId, PATIENT_ID)
      setMessages((m) => [...m, { role: 'assistant', text: res.reply, meta: res }])
      setLastTurn(res)
      if (res.booking_created) {
        setHighlight(res.booking_created.bookingId)
        await refreshBookings()
        api.patient(PATIENT_ID).then(setPatient).catch(() => undefined)
      }
    } catch (e) {
      setMessages((m) => [...m, { role: 'assistant', text: (e as Error).message, error: true }])
    } finally {
      setBusy(false)
    }
  }

  function resetConversation() {
    setSessionId(newSessionId())
    setLastTurn(null)
    setMessages([{ role: 'assistant', text: 'New conversation started. How can I help?' }])
  }

  const upcoming = bookings.filter((b) => b.patientId === PATIENT_ID)

  return (
    <div className="app">
      <header>
        <h1>
          ClinicPilot <span>AI</span>
        </h1>
        <nav>
          <button className={tab === 'chat' ? 'active' : ''} onClick={() => setTab('chat')}>
            Assistant
          </button>
          <button className={tab === 'summary' ? 'active' : ''} onClick={() => setTab('summary')}>
            Transcript summarizer
          </button>
        </nav>
      </header>

      {tab === 'summary' ? (
        <Summarizer />
      ) : (
        <main className="grid">
          <aside className="panel">
            <h2>Patient</h2>
            {patient ? (
              <div className="patient">
                <strong>{patient.fullName}</strong> <span className="muted">{patient.patientId}</span>
                <div className="muted">Born {patient.dob}</div>
                <div className="allergies">
                  Allergies:{' '}
                  {patient.allergies.length ? patient.allergies.map((a) => <span key={a} className="tag warn">{a}</span>) : 'none'}
                </div>
                <div className="muted small">Emergency contact: {patient.emergencyContact}</div>
                {patient.treatmentHistory.length > 0 && (
                  <div className="small">
                    Last visit: {patient.treatmentHistory[0].visitDate} - {patient.treatmentHistory[0].diagnosis}
                  </div>
                )}
              </div>
            ) : (
              <div className="muted">Backend not reachable.</div>
            )}
            <h2>Try it</h2>
            <div className="quick">
              {QUICK_PROMPTS.map((q) => (
                <button key={q} onClick={() => void send(q)} disabled={busy}>
                  {q}
                </button>
              ))}
            </div>
            <button className="secondary" onClick={resetConversation}>
              Start a new conversation
            </button>
            <div className="muted small">Session: {sessionId}</div>
          </aside>

          <section className="panel chat">
            <h2>Assistant</h2>
            <div className="messages">
              {messages.map((m, i) => (
                <div key={i} className={`msg ${m.role} ${m.error ? 'error' : ''}`}>
                  <div className="bubble">{renderText(m.text)}</div>
                  {m.meta && (
                    <div className="meta">
                      <span className="tag">{m.meta.intent}</span>
                      {m.meta.tool_calls
                        .filter((c) => c.name !== 'get_patient_profile' && c.name !== 'get_catalog')
                        .map((c, j) => (
                          <span key={j} className={`tag ${c.ok ? 'tool' : 'warn'}`}>
                            {c.name} via {c.transport}
                          </span>
                        ))}
                      {m.meta.awaiting_confirmation && <span className="tag warn">awaiting confirmation</span>}
                      {m.meta.booking_created && <span className="tag ok">saved {m.meta.booking_created.bookingId}</span>}
                      <span className="muted small">{(m.meta.latency_ms / 1000).toFixed(1)}s</span>
                    </div>
                  )}
                </div>
              ))}
              {busy && <div className="msg assistant"><div className="bubble typing">Thinking...</div></div>}
              <div ref={endRef} />
            </div>
            <form
              onSubmit={(e) => {
                e.preventDefault()
                void send(input)
              }}
            >
              <input
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Ask a question or book an appointment..."
                disabled={busy}
                aria-label="Message"
              />
              <button type="submit" disabled={busy || !input.trim()}>
                Send
              </button>
            </form>
          </section>

          <aside className="panel right">
            <div className="row">
              <h2>Bookings (live from database)</h2>
              <button className="secondary small" onClick={() => void refreshBookings()}>
                Refresh
              </button>
            </div>
            {bookingsError && <div className="error small">{bookingsError}</div>}
            <table>
              <thead>
                <tr>
                  <th>ID</th>
                  <th>Date</th>
                  <th>Time</th>
                  <th>Doctor / Service</th>
                  <th>Source</th>
                </tr>
              </thead>
              <tbody>
                {upcoming.map((b) => (
                  <tr key={b.bookingId} className={b.bookingId === highlight ? 'new' : ''}>
                    <td>{b.bookingId}</td>
                    <td>{b.bookingDate}</td>
                    <td>{b.bookingTime.slice(0, 5)}</td>
                    <td>
                      {b.doctorName}
                      <div className="muted small">{b.treatmentName}</div>
                    </td>
                    <td>
                      <span className={`tag ${b.source === 'chatbot' ? 'ok' : ''}`}>{b.source}</span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>

            <h2>Agent trace (last turn)</h2>
            {lastTurn ? (
              <div className="trace">
                <div className="small">
                  Intent <b>{lastTurn.intent}</b> ({Math.round(lastTurn.confidence * 100)}%) - {lastTurn.usage.llm_calls} LLM
                  call(s), {lastTurn.usage.input_tokens + lastTurn.usage.output_tokens} tokens, {lastTurn.latency_ms} ms
                </div>
                <ol>
                  {lastTurn.trace.map((t, i) => (
                    <li key={i}>
                      <code>{t.step}</code> <span className="muted">{t.ms} ms</span>
                      {t.detail && <div className="small">{t.detail}</div>}
                    </li>
                  ))}
                </ol>
                {lastTurn.tool_calls.length > 0 && (
                  <>
                    <div className="small"><b>Tool calls</b></div>
                    <ul>
                      {lastTurn.tool_calls.map((c, i) => (
                        <li key={i} className="small">
                          <code>{c.name}</code> via {c.transport} {c.ok ? 'ok' : 'failed'} ({c.ms} ms)
                        </li>
                      ))}
                    </ul>
                  </>
                )}
                {lastTurn.booking_draft && (
                  <div className="small">
                    <b>Booking draft:</b> <code>{JSON.stringify(lastTurn.booking_draft)}</code>
                  </div>
                )}
                {lastTurn.citations.length > 0 && (
                  <div className="small">
                    <b>Sources:</b> {lastTurn.citations.map((c) => <span key={c} className="cite">{c}</span>)}
                  </div>
                )}
              </div>
            ) : (
              <div className="muted small">Send a message to see intent, steps, tool calls and sources.</div>
            )}
          </aside>
        </main>
      )}
    </div>
  )
}

function Summarizer() {
  const [text, setText] = useState(SAMPLE_TRANSCRIPT)
  const [result, setResult] = useState<MedicalSummary | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  async function run() {
    setBusy(true)
    setError(null)
    try {
      setResult((await api.summarize(text)).summary)
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setBusy(false)
    }
  }

  return (
    <main className="grid two">
      <section className="panel">
        <h2>Doctor-patient transcript</h2>
        <textarea value={text} onChange={(e) => setText(e.target.value)} rows={16} aria-label="Transcript" />
        <button onClick={() => void run()} disabled={busy || text.trim().length < 20}>
          {busy ? 'Summarizing...' : 'Generate structured summary'}
        </button>
        {error && <div className="error small">{error}</div>}
      </section>
      <section className="panel">
        <h2>Structured summary (JSON)</h2>
        {result ? <pre>{JSON.stringify(result, null, 2)}</pre> : <div className="muted">The summary appears here.</div>}
      </section>
    </main>
  )
}
