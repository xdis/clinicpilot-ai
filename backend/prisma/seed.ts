/**
 * Seeds the database from the CSV files in ../mock-data.
 * Idempotent: every row is upserted by its primary key, so re-running never duplicates data
 * and never deletes bookings created by the chatbot.
 */
import { PrismaClient } from '@prisma/client';
import { parse } from 'csv-parse/sync';
import { readFileSync } from 'fs';
import { join } from 'path';

const prisma = new PrismaClient();
const DATA_DIR = process.env.MOCK_DATA_DIR ?? join(__dirname, '..', '..', 'mock-data');

function readCsv(name: string): Record<string, string>[] {
  return parse(readFileSync(join(DATA_DIR, name), 'utf8'), {
    columns: true,
    skip_empty_lines: true,
    trim: true,
  });
}

async function main() {
  for (const r of readCsv('clinics.csv')) {
    const data = {
      name: r.clinic_name,
      address: r.address,
      phone: r.phone,
      operatingHours: r.operating_hours,
      cancellationPolicy: r.cancellation_policy,
      emergencyInstructions: r.emergency_instructions,
    };
    await prisma.clinic.upsert({ where: { id: r.clinic_id }, create: { id: r.clinic_id, ...data }, update: data });
  }
  for (const r of readCsv('doctors.csv')) {
    const data = {
      clinicId: r.clinic_id,
      fullName: r.full_name,
      specialty: r.specialty,
      languagesSpoken: r.languages_spoken,
      consultationFeeUsd: Number(r.consultation_fee_usd),
      bio: r.bio,
    };
    await prisma.doctor.upsert({ where: { id: r.doctor_id }, create: { id: r.doctor_id, ...data }, update: data });
  }
  for (const r of readCsv('treatments.csv')) {
    const data = {
      name: r.treatment_name,
      category: r.category,
      durationMinutes: Number(r.duration_minutes),
      priceUsd: Number(r.price_usd),
      preTreatmentInstructions: r.pre_treatment_instructions,
    };
    await prisma.treatment.upsert({ where: { id: r.treatment_id }, create: { id: r.treatment_id, ...data }, update: data });
  }
  for (const r of readCsv('patients.csv')) {
    const data = {
      firstName: r.first_name,
      lastName: r.last_name,
      dob: r.dob,
      gender: r.gender,
      phone: r.phone,
      email: r.email,
      knownAllergies: r.known_allergies,
      emergencyContact: r.emergency_contact,
    };
    await prisma.patient.upsert({ where: { id: r.patient_id }, create: { id: r.patient_id, ...data }, update: data });
  }
  for (const r of readCsv('patient_bookings.csv')) {
    const data = {
      patientId: r.patient_id,
      doctorId: r.doctor_id,
      treatmentId: r.treatment_id,
      bookingDate: r.booking_date,
      bookingTime: r.booking_time,
      status: r.status,
      notes: r.notes ?? '',
      source: 'seed',
    };
    await prisma.patientBooking.upsert({ where: { id: r.booking_id }, create: { id: r.booking_id, ...data }, update: data });
  }
  for (const r of readCsv('patient_treatment_history.csv')) {
    const data = {
      patientId: r.patient_id,
      doctorId: r.doctor_id,
      visitDate: r.visit_date,
      chiefComplaint: r.chief_complaint,
      diagnosis: r.diagnosis,
      prescribedMedication: r.prescribed_medication,
      doctorNotes: r.doctor_notes,
    };
    await prisma.treatmentHistory.upsert({ where: { id: r.record_id }, create: { id: r.record_id, ...data }, update: data });
  }
  const counts = {
    clinics: await prisma.clinic.count(),
    doctors: await prisma.doctor.count(),
    treatments: await prisma.treatment.count(),
    patients: await prisma.patient.count(),
    bookings: await prisma.patientBooking.count(),
    history: await prisma.treatmentHistory.count(),
  };
  console.log('Seed complete:', counts);
}

main()
  .catch((err) => {
    console.error(err);
    process.exit(1);
  })
  .finally(() => prisma.$disconnect());
