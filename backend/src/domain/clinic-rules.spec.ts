import {
  checkWithinOpeningHours,
  clinicTimezone,
  dayOfWeek,
  isSpecialtyAllowed,
  parseOperatingHours,
} from './clinic-rules';

const METRO = 'Mon-Fri: 8:00 AM - 6:00 PM, Sat: 9:00 AM - 1:00 PM, Sun: Closed';
const SUNSET = 'Mon-Fri: 9:00 AM - 5:00 PM, Sat-Sun: Closed';

describe('clinic rules', () => {
  it('parses weekday ranges, single days and closed days', () => {
    const s = parseOperatingHours(METRO);
    expect(s.Mon).toEqual({ open: 480, close: 1080 });
    expect(s.Sat).toEqual({ open: 540, close: 780 });
    expect(s.Sun).toBeNull();
    const t = parseOperatingHours(SUNSET);
    expect(t.Sat).toBeNull();
    expect(t.Sun).toBeNull();
  });

  it('computes the weekday of an ISO date', () => {
    expect(dayOfWeek('2026-10-25')).toBe('Sun');
    expect(dayOfWeek('2026-10-26')).toBe('Mon');
  });

  it('accepts a slot that fits and rejects closed days or overruns', () => {
    expect(checkWithinOpeningHours(METRO, '2026-10-26', '10:00:00', 30).ok).toBe(true);
    expect(checkWithinOpeningHours(METRO, '2026-10-25', '10:00:00', 30).ok).toBe(false);
    expect(checkWithinOpeningHours(METRO, '2026-10-31', '12:45:00', 30).ok).toBe(false); // Sat closes 13:00
    expect(checkWithinOpeningHours(METRO, '2026-10-26', '07:30:00', 30).ok).toBe(false);
  });

  it('maps treatments to compatible specialties', () => {
    expect(isSpecialtyAllowed('Dermatology', 'Dermatology')).toBe(true);
    expect(isSpecialtyAllowed('Immunization', 'Pediatrics')).toBe(true);
    expect(isSpecialtyAllowed('Orthopedics', 'General Practice')).toBe(false);
  });

  it('infers the clinic timezone from the address', () => {
    expect(clinicTimezone('123 Medical Center Way, Suite 400, New York, NY 10001')).toBe('America/New_York');
    expect(clinicTimezone('456 Ocean Ave, Building B, Santa Monica, CA 90401')).toBe('America/Los_Angeles');
  });
});
