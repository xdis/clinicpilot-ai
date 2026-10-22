import { INestApplication } from '@nestjs/common';
import { Test } from '@nestjs/testing';
import request from 'supertest';
import { AppModule } from '../src/app.module';
import { configureApp } from '../src/main';

// Runs against prisma/test.db, freshly migrated and seeded by `npm run test:e2e`,
// with CLINIC_TODAY=2026-10-22 so date rules are deterministic.
describe('ClinicPilot backend (e2e)', () => {
  let app: INestApplication;

  beforeAll(async () => {
    const moduleRef = await Test.createTestingModule({ imports: [AppModule] }).compile();
    app = configureApp(moduleRef.createNestApplication());
    await app.init();
  });

  afterAll(async () => {
    await app.close();
  });

  it('GET /api/health', async () => {
    const res = await request(app.getHttpServer()).get('/api/health').expect(200);
    expect(res.body.status).toBe('ok');
  });

  it('GET /api/bookings returns the seeded bookings enriched with names', async () => {
    const res = await request(app.getHttpServer()).get('/api/bookings?patientId=P001').expect(200);
    expect(res.body.map((b: { bookingId: string }) => b.bookingId)).toEqual(['B1001', 'B1004']);
    expect(res.body[0]).toMatchObject({ doctorName: 'Dr. Sarah Jenkins', treatmentName: 'General Health Checkup' });
  });

  it('GET /api/patients/P001 includes allergies and history', async () => {
    const res = await request(app.getHttpServer()).get('/api/patients/P001').expect(200);
    expect(res.body.allergies).toEqual(['Penicillin', 'Peanuts']);
    expect(res.body.treatmentHistory).toHaveLength(1);
  });

  it('POST /api/bookings creates a booking that GET then returns', async () => {
    const res = await request(app.getHttpServer())
      .post('/api/bookings')
      .send({
        patientId: 'P001',
        doctorId: 'D103',
        treatmentId: 'T003',
        bookingDate: '2026-10-26',
        bookingTime: '10:00',
        notes: 'Itchy rash',
        source: 'chatbot',
        idempotencyKey: 'e2e-1',
      })
      .expect(201);
    expect(res.body.booking).toMatchObject({ bookingId: 'B1005', status: 'Confirmed', source: 'chatbot' });
    const list = await request(app.getHttpServer()).get('/api/bookings?patientId=P001').expect(200);
    expect(list.body.map((b: { bookingId: string }) => b.bookingId)).toContain('B1005');
  });

  it('replays an idempotent request instead of creating a duplicate', async () => {
    const res = await request(app.getHttpServer())
      .post('/api/bookings')
      .send({
        patientId: 'P001',
        doctorId: 'D103',
        treatmentId: 'T003',
        bookingDate: '2026-10-26',
        bookingTime: '10:00',
        idempotencyKey: 'e2e-1',
      })
      .expect(200);
    expect(res.body).toMatchObject({ idempotentReplay: true, booking: { bookingId: 'B1005' } });
  });

  it('rejects a specialty mismatch, a closed day and a past date with a clear message', async () => {
    const res = await request(app.getHttpServer())
      .post('/api/bookings')
      .send({ patientId: 'P001', doctorId: 'D101', treatmentId: 'T005', bookingDate: '2026-10-18', bookingTime: '10:00' })
      .expect(422);
    expect(res.body.error.code).toBe('BOOKING_INVALID');
    expect(res.body.error.details).toHaveLength(3);
  });

  it('POST /api/bookings/validate dry-runs the same rules without saving', async () => {
    const ok = await request(app.getHttpServer())
      .post('/api/bookings/validate')
      .send({ patientId: 'P001', doctorId: 'D104', treatmentId: 'T005', bookingDate: '2026-10-27', bookingTime: '11:00' })
      .expect(200);
    expect(ok.body).toEqual({ valid: true, problems: [] });
    const bad = await request(app.getHttpServer())
      .post('/api/bookings/validate')
      .send({ patientId: 'P001', doctorId: 'D103', treatmentId: 'T003', bookingDate: '2026-10-31', bookingTime: '10:00' })
      .expect(200);
    expect(bad.body.valid).toBe(false);
    expect(bad.body.problems[0]).toContain('closed on Sat');
  });

  it('validates the request body', async () => {
    const res = await request(app.getHttpServer())
      .post('/api/bookings')
      .send({ patientId: 'P001', doctorId: 'D103', treatmentId: 'T003', bookingDate: '26/10/2026', bookingTime: '10am' })
      .expect(400);
    expect(res.body.error.message).toBe('Request validation failed.');
  });

  it('returns 404 for an unknown doctor', async () => {
    const res = await request(app.getHttpServer())
      .post('/api/bookings')
      .send({ patientId: 'P001', doctorId: 'D999', treatmentId: 'T003', bookingDate: '2026-10-26', bookingTime: '10:00' })
      .expect(404);
    expect(res.body.error.code).toBe('DOCTOR_NOT_FOUND');
  });
});
