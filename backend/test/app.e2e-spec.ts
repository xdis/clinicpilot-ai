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

  it('rejects reusing an idempotency key with different details, and accepts the Idempotency-Key header', async () => {
    const conflict = await request(app.getHttpServer())
      .post('/api/bookings')
      .send({ patientId: 'P001', doctorId: 'D103', treatmentId: 'T003', bookingDate: '2026-10-27', bookingTime: '10:00', idempotencyKey: 'e2e-1' })
      .expect(409);
    expect(conflict.body.error.code).toBe('IDEMPOTENCY_KEY_REUSED');
    const body = { patientId: 'P001', doctorId: 'D104', treatmentId: 'T005', bookingDate: '2026-10-28', bookingTime: '11:00' };
    const first = await request(app.getHttpServer()).post('/api/bookings').set('Idempotency-Key', 'hdr-1').send(body).expect(201);
    const again = await request(app.getHttpServer()).post('/api/bookings').set('Idempotency-Key', 'hdr-1').send(body).expect(200);
    expect(again.body.booking.bookingId).toBe(first.body.booking.bookingId);
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

  it('rejects an impossible calendar date', async () => {
    const res = await request(app.getHttpServer())
      .post('/api/bookings')
      .send({ patientId: 'P001', doctorId: 'D103', treatmentId: 'T003', bookingDate: '2026-02-30', bookingTime: '10:00' })
      .expect(400);
    expect(JSON.stringify(res.body.error.details)).toContain('real calendar date');
  });

  it('accepts the brief\'s snake_case patient_id on /api/chat (gateway validation)', async () => {
    process.env.AI_SERVICE_URL = 'http://127.0.0.1:9'; // nothing listens: proves validation passed and the call was forwarded
    const res = await request(app.getHttpServer()).post('/api/chat').send({ message: 'hi', patient_id: 'P001' });
    expect(res.status).toBe(503);
    expect(res.body.error.code).toBe('AI_SERVICE_UNAVAILABLE');
  });

  it('returns 404 for an unknown doctor', async () => {
    const res = await request(app.getHttpServer())
      .post('/api/bookings')
      .send({ patientId: 'P001', doctorId: 'D999', treatmentId: 'T003', bookingDate: '2026-10-26', bookingTime: '10:00' })
      .expect(404);
    expect(res.body.error.code).toBe('DOCTOR_NOT_FOUND');
  });

  describe('appointment conflicts (beyond the brief)', () => {
    const book = (body: Record<string, string>) => request(app.getHttpServer()).post('/api/bookings').send(body);

    it('rejects an overlapping slot for the same doctor and suggests free times', async () => {
      await book({ patientId: 'P002', doctorId: 'D104', treatmentId: 'T005', bookingDate: '2026-11-02', bookingTime: '11:00' }).expect(201);
      const res = await book({ patientId: 'P003', doctorId: 'D104', treatmentId: 'T005', bookingDate: '2026-11-02', bookingTime: '11:30' }).expect(422);
      expect(res.body.error.message).toContain('Dr. James Wilson is already booked at that time');
      expect(res.body.error.message).toContain('Free times that day: 08:00, 08:30, 09:00');
      expect(res.body.error.message).toMatch(/already booked at that time \(11:00-11:45\)/); // no other patient's booking id
      await book({ patientId: 'P003', doctorId: 'D104', treatmentId: 'T005', bookingDate: '2026-11-02', bookingTime: '11:45' }).expect(201); // back-to-back
    });

    it('rejects a second appointment for the same patient at the same time', async () => {
      const res = await book({ patientId: 'P002', doctorId: 'D101', treatmentId: 'T001', bookingDate: '2026-11-02', bookingTime: '11:15' }).expect(422);
      expect(res.body.error.message).toContain('You already have an appointment at that time');
    });

    it('says so clearly when the patient asks for an appointment they already have', async () => {
      const res = await book({ patientId: 'P001', doctorId: 'D101', treatmentId: 'T001', bookingDate: '2026-11-05', bookingTime: '09:00' }).expect(422);
      expect(res.body.error.message).toBe('You already have this appointment (B1001, 09:00-09:30).');
    });

    it('a retried request with the same idempotency key gets its booking back, even when sent concurrently', async () => {
      const body = { patientId: 'P002', doctorId: 'D101', treatmentId: 'T004', bookingDate: '2026-11-04', bookingTime: '10:00', idempotencyKey: 'retry-1' };
      const results = await Promise.all([book(body), book(body)]);
      expect(results.map((r) => r.status).sort()).toEqual([200, 201]);
      expect(results[0].body.booking.bookingId).toBe(results[1].body.booking.bookingId);
    });

    it('lets only one of two simultaneous requests take the same slot', async () => {
      const body = { doctorId: 'D101', treatmentId: 'T004', bookingDate: '2026-11-03', bookingTime: '09:00' };
      const results = await Promise.all([book({ ...body, patientId: 'P002' }), book({ ...body, patientId: 'P003' })]);
      const statuses = results.map((r) => r.status).sort();
      expect(statuses[0]).toBe(201);
      expect([409, 422]).toContain(statuses[1]);
    });
  });

  describe('consultation recording upload', () => {
    it('requires a file', async () => {
      const res = await request(app.getHttpServer()).post('/api/summarize/audio').expect(400);
      expect(res.body.error.code).toBe('AUDIO_REQUIRED');
    });

    it('accepts multipart audio and forwards it to the AI service', async () => {
      process.env.AI_SERVICE_URL = 'http://127.0.0.1:9'; // nothing listens: 503 proves parsing + forwarding happened
      const res = await request(app.getHttpServer())
        .post('/api/summarize/audio')
        .attach('file', Buffer.from('ID3fake'), { filename: 'visit.mp3', contentType: 'audio/mpeg' });
      expect(res.status).toBe(503);
      expect(res.body.error.code).toBe('AI_SERVICE_UNAVAILABLE');
    });
  });
});
