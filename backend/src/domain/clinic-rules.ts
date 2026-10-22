/**
 * Deterministic business rules for bookings.
 * These live in code (not in the LLM prompt) so the model can never book something invalid:
 * the AI proposes, the backend validates.
 */

const DAYS = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'] as const;
type Day = (typeof DAYS)[number];

export interface OpeningWindow {
  open: number; // minutes since midnight
  close: number;
}
export type WeeklySchedule = Partial<Record<Day, OpeningWindow | null>>;

/**
 * Which doctor specialties may perform a treatment, keyed by treatment category.
 * The CSVs do not link treatments to doctors, so this mapping is an explicit, documented assumption.
 */
export const CATEGORY_TO_SPECIALTIES: Record<string, string[]> = {
  'Preventative Care': ['General Practice'],
  Pediatrics: ['Pediatrics'],
  Dermatology: ['Dermatology'],
  Immunization: ['General Practice', 'Pediatrics'],
  Orthopedics: ['Orthopedics'],
};

/** Clinic timezone inferred from the US state in the address (data is US-based). */
const STATE_TIMEZONES: Record<string, string> = {
  NY: 'America/New_York',
  CA: 'America/Los_Angeles',
};

export function clinicTimezone(address: string): string {
  const match = address.match(/,\s*([A-Z]{2})\s+\d{5}/);
  return (match && STATE_TIMEZONES[match[1]]) || 'UTC';
}

function parseClock(text: string): number {
  const m = text.trim().match(/^(\d{1,2}):(\d{2})\s*(AM|PM)$/i);
  if (!m) throw new Error(`Unrecognised time: "${text}"`);
  let hours = Number(m[1]) % 12;
  if (m[3].toUpperCase() === 'PM') hours += 12;
  return hours * 60 + Number(m[2]);
}

function expandDays(spec: string): Day[] {
  const [start, end] = spec.split('-').map((d) => d.trim() as Day);
  const i = DAYS.indexOf(start);
  if (i < 0) throw new Error(`Unrecognised day: "${start}"`);
  if (!end) return [start];
  const j = DAYS.indexOf(end);
  if (j < 0) throw new Error(`Unrecognised day: "${end}"`);
  const out: Day[] = [];
  for (let k = i; ; k = (k + 1) % 7) {
    out.push(DAYS[k]);
    if (k === j) break;
  }
  return out;
}

/** Parses strings like "Mon-Fri: 8:00 AM - 6:00 PM, Sat: 9:00 AM - 1:00 PM, Sun: Closed". */
export function parseOperatingHours(text: string): WeeklySchedule {
  const schedule: WeeklySchedule = {};
  for (const part of text.split(',')) {
    const idx = part.indexOf(':');
    if (idx < 0) continue;
    const days = expandDays(part.slice(0, idx));
    const hours = part.slice(idx + 1).trim();
    let window: OpeningWindow | null = null;
    if (!/^closed$/i.test(hours)) {
      const [open, close] = hours.split(/\s+-\s+/);
      window = { open: parseClock(open), close: parseClock(close) };
    }
    for (const d of days) schedule[d] = window;
  }
  return schedule;
}

export function dayOfWeek(isoDate: string): Day {
  const [y, m, d] = isoDate.split('-').map(Number);
  return DAYS[new Date(Date.UTC(y, m - 1, d)).getUTCDay()];
}

export function minutesOf(time: string): number {
  const [h, m] = time.split(':').map(Number);
  return h * 60 + m;
}

/** Today's date (YYYY-MM-DD) in a given timezone. CLINIC_TODAY can pin it for tests and demos. */
export function todayIn(timezone: string, now: Date = new Date()): string {
  if (process.env.CLINIC_TODAY) return process.env.CLINIC_TODAY;
  return new Intl.DateTimeFormat('en-CA', { timeZone: timezone }).format(now);
}

/** Current clinic-local time in minutes, or null when "today" is pinned for tests/demos. */
export function nowMinutesIn(timezone: string, now: Date = new Date()): number | null {
  if (process.env.CLINIC_TODAY) return null;
  const parts = new Intl.DateTimeFormat('en-GB', { timeZone: timezone, hour: '2-digit', minute: '2-digit', hourCycle: 'h23' })
    .formatToParts(now)
    .reduce<Record<string, string>>((acc, p) => ({ ...acc, [p.type]: p.value }), {});
  return Number(parts.hour) * 60 + Number(parts.minute);
}

export interface SlotCheck {
  ok: boolean;
  reason?: string;
}

/** The whole appointment (start + duration) must fit inside the clinic's opening window. */
export function checkWithinOpeningHours(
  operatingHours: string,
  date: string,
  time: string,
  durationMinutes: number,
): SlotCheck {
  const schedule = parseOperatingHours(operatingHours);
  const day = dayOfWeek(date);
  const window = schedule[day];
  if (!window) return { ok: false, reason: `The clinic is closed on ${day}. Opening hours: ${operatingHours}.` };
  const start = minutesOf(time);
  if (start < window.open || start + durationMinutes > window.close) {
    return {
      ok: false,
      reason: `${time.slice(0, 5)} on ${day} is outside opening hours for a ${durationMinutes}-minute appointment. Opening hours: ${operatingHours}.`,
    };
  }
  return { ok: true };
}

export function isSpecialtyAllowed(treatmentCategory: string, doctorSpecialty: string): boolean {
  const allowed = CATEGORY_TO_SPECIALTIES[treatmentCategory];
  return !!allowed && allowed.includes(doctorSpecialty);
}
