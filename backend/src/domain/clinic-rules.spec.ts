import {
  checkWithinOpeningHours,
  findOverlap,
  freeStarts,
  interval,
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

  it('detects overlapping appointments using each treatment duration (back-to-back is fine)', () => {
    const busy = [interval('11:00:00', 45, 'B1')]; // 11:00-11:45
    expect(findOverlap(busy, '11:30:00', 30)?.label).toBe('B1, 11:00-11:45');
    expect(findOverlap(busy, '10:30:00', 45)).toBeDefined(); // ends inside
    expect(findOverlap(busy, '11:45:00', 30)).toBeUndefined(); // starts when the other ends
    expect(findOverlap(busy, '10:30:00', 30)).toBeUndefined(); // ends when the other starts
  });

  it('suggests free start times inside opening hours', () => {
    const busy = [interval('09:00:00', 30, 'B1'), interval('09:30:00', 30, 'B2')];
    expect(freeStarts(SUNSET, '2026-10-26', 30, busy)).toEqual(['10:00', '10:30', '11:00']);
    expect(freeStarts(SUNSET, '2026-10-31', 30, [])).toEqual([]); // Saturday: closed
    expect(freeStarts(SUNSET, '2026-10-26', 30, [], { notBefore: 15 * 60 + 59 })).toEqual(['16:00', '16:30']);
  });
});
