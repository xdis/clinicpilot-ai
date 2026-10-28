import { ConflictException, Injectable, NotFoundException, UnprocessableEntityException } from '@nestjs/common';
import { Prisma } from '@prisma/client';
import { PrismaService } from '../prisma/prisma.service';
import {
  CATEGORY_TO_SPECIALTIES,
  checkWithinOpeningHours,
  clinicTimezone,
  isSpecialtyAllowed,
  minutesOf,
  nowMinutesIn,
  todayIn,
  findOverlap,
  freeStarts,
  interval,
} from '../domain/clinic-rules';
import { CreateBookingDto, ListBookingsQuery } from './dto/booking.dto';

const bookingInclude = {
  doctor: { include: { clinic: true } },
  treatment: true,
  patient: true,
} satisfies Prisma.PatientBookingInclude;

type BookingRow = Prisma.PatientBookingGetPayload<{ include: typeof bookingInclude }>;

export interface BookingView {
  bookingId: string;
  patientId: string;
  patientName: string;
  doctorId: string;
  doctorName: string;
  specialty: string;
  treatmentId: string;
  treatmentName: string;
  clinicId: string;
  clinicName: string;
  bookingDate: string;
  bookingTime: string;
  durationMinutes: number;
  priceUsd: number;
  consultationFeeUsd: number;
  status: string;
  notes: string;
  source: string;
  preTreatmentInstructions: string;
  createdAt: string;
}

function normalizeTime(time: string): string {
  return time.length === 5 ? `${time}:00` : time;
}

function toView(b: BookingRow): BookingView {
  return {
    bookingId: b.id,
    patientId: b.patientId,
    patientName: `${b.patient.firstName} ${b.patient.lastName}`,
    doctorId: b.doctorId,
    doctorName: b.doctor.fullName,
    specialty: b.doctor.specialty,
    treatmentId: b.treatmentId,
    treatmentName: b.treatment.name,
    clinicId: b.doctor.clinicId,
    clinicName: b.doctor.clinic.name,
    bookingDate: b.bookingDate,
    bookingTime: b.bookingTime,
    durationMinutes: b.treatment.durationMinutes,
    priceUsd: b.treatment.priceUsd,
    consultationFeeUsd: b.doctor.consultationFeeUsd,
    status: b.status,
    notes: b.notes,
    source: b.source,
    preTreatmentInstructions: b.treatment.preTreatmentInstructions,
    createdAt: b.createdAt.toISOString(),
  };
}

@Injectable()
export class BookingsService {
  constructor(private readonly prisma: PrismaService) {}

  async list(query: ListBookingsQuery): Promise<BookingView[]> {
    const rows = await this.prisma.patientBooking.findMany({
      where: query.patientId ? { patientId: query.patientId } : undefined,
      include: bookingInclude,
      orderBy: [{ bookingDate: 'asc' }, { bookingTime: 'asc' }],
    });
    const views = rows.map(toView);
    if (!query.when || query.when === 'all') return views;
    return views.filter((v) => {
      const today = todayIn(clinicTimezone(rows.find((r) => r.id === v.bookingId)!.doctor.clinic.address));
      return query.when === 'upcoming' ? v.bookingDate >= today : v.bookingDate < today;
    });
  }

  /** Dry run used by the AI agent before it asks the patient to confirm. Same rules as create(). */
  async validate(dto: CreateBookingDto): Promise<{ valid: boolean; problems: string[] }> {
    const problems = await this.findProblems(dto);
    return { valid: problems.length === 0, problems };
  }

  private async findProblems(dto: CreateBookingDto): Promise<string[]> {
    const [patient, doctor, treatment] = await Promise.all([
      this.prisma.patient.findUnique({ where: { id: dto.patientId } }),
      this.prisma.doctor.findUnique({ where: { id: dto.doctorId }, include: { clinic: true } }),
      this.prisma.treatment.findUnique({ where: { id: dto.treatmentId } }),
    ]);
    if (!patient) throw new NotFoundException({ code: 'PATIENT_NOT_FOUND', message: `Patient ${dto.patientId} not found.` });
    if (!doctor) throw new NotFoundException({ code: 'DOCTOR_NOT_FOUND', message: `Doctor ${dto.doctorId} not found.` });
    if (!treatment)
      throw new NotFoundException({ code: 'TREATMENT_NOT_FOUND', message: `Treatment ${dto.treatmentId} not found.` });

    const problems: string[] = [];
    if (!isSpecialtyAllowed(treatment.category, doctor.specialty)) {
      const allowed = CATEGORY_TO_SPECIALTIES[treatment.category] ?? [];
      problems.push(
        `${treatment.name} is provided by ${allowed.join(' or ') || 'no listed specialty'}, but ${doctor.fullName} is ${doctor.specialty}.`,
      );
    }
    const tz = clinicTimezone(doctor.clinic.address);
    const today = todayIn(tz);
    if (dto.bookingDate < today) problems.push(`${dto.bookingDate} is in the past (today is ${today} at the clinic).`);
    const nowMin = nowMinutesIn(tz);
    if (dto.bookingDate === today && nowMin !== null && minutesOf(normalizeTime(dto.bookingTime)) <= nowMin) {
      problems.push(`${dto.bookingTime.slice(0, 5)} today has already passed at the clinic.`);
    }
    const slot = checkWithinOpeningHours(
      doctor.clinic.operatingHours,
      dto.bookingDate,
      normalizeTime(dto.bookingTime),
      treatment.durationMinutes,
    );
    if (!slot.ok) problems.push(slot.reason!);
    if (!problems.length) problems.push(...(await this.findConflicts(dto, doctor, treatment.durationMinutes, today, nowMin)));
    return problems;
  }

  /**
   * Beyond the brief (which allows assuming every slot is free): a doctor cannot see two patients at once and a
   * patient cannot be in two appointments at once. Intervals use each treatment's duration, so a 45-minute
   * consultation at 11:00 also blocks 11:30. The database backs this up for identical start times
   * (partial unique index, see the slot_unique migration).
   */
  private async findConflicts(
    dto: CreateBookingDto,
    doctor: { id: string; fullName: string; clinic: { operatingHours: string } },
    durationMinutes: number,
    today: string,
    nowMin: number | null,
  ): Promise<string[]> {
    const time = normalizeTime(dto.bookingTime);
    const sameDay = await this.prisma.patientBooking.findMany({
      where: {
        bookingDate: dto.bookingDate,
        status: { not: 'Cancelled' },
        OR: [{ doctorId: dto.doctorId }, { patientId: dto.patientId }],
      },
      include: { treatment: true },
    });
    const toInterval = (b: (typeof sameDay)[number]) => interval(b.bookingTime, b.treatment.durationMinutes, b.id);
    const doctorBusy = sameDay.filter((b) => b.doctorId === dto.doctorId).map(toInterval);
    const patientBusy = sameDay.filter((b) => b.patientId === dto.patientId).map(toInterval);
    const problems: string[] = [];
    const clash = findOverlap(doctorBusy, time, durationMinutes);
    const own = findOverlap(patientBusy, time, durationMinutes);
    if (own && clash && own.id === clash.id) {
      return [`You already have this appointment (${own.label}).`]; // same booking: one clear message
    }
    if (clash) {
      const free = freeStarts(doctor.clinic.operatingHours, dto.bookingDate, durationMinutes, doctorBusy, {
        notBefore: dto.bookingDate === today && nowMin !== null ? nowMin : -1,
      });
      problems.push(
        `${doctor.fullName} is already booked at that time (${clash.label}).` +
          (free.length ? ` Free times that day: ${free.join(', ')}.` : ' No other free time that day.'),
      );
    }
    if (own) problems.push(`You already have an appointment at that time (${own.label}).`);
    return problems;
  }

  async create(dto: CreateBookingDto): Promise<{ booking: BookingView; idempotentReplay: boolean }> {
    if (dto.idempotencyKey) {
      const existing = await this.prisma.patientBooking.findUnique({
        where: { idempotencyKey: dto.idempotencyKey },
        include: bookingInclude,
      });
      if (existing) return { booking: toView(this.assertSameRequest(existing, dto)), idempotentReplay: true };
    }

    const problems = await this.findProblems(dto);
    if (problems.length) {
      throw new UnprocessableEntityException({
        code: 'BOOKING_INVALID',
        message: problems.join(' '),
        details: problems,
      });
    }
    const time = normalizeTime(dto.bookingTime);

    for (let attempt = 0; ; attempt++) {
      try {
        const created = await this.insert(dto, time);
        return { booking: toView(created), idempotentReplay: false };
      } catch (err) {
        if (!(err instanceof Prisma.PrismaClientKnownRequestError) || err.code !== 'P2002' || attempt >= 3) throw err;
        // SQLite may name any violated unique index, so the idempotent replay is checked first:
        // a client retrying while its first request is still in flight must get its booking back, not a 409
        if (dto.idempotencyKey) {
          const existing = await this.prisma.patientBooking.findUnique({
            where: { idempotencyKey: dto.idempotencyKey },
            include: bookingInclude,
          });
          if (existing) return { booking: toView(this.assertSameRequest(existing, dto)), idempotentReplay: true };
        }
        if (JSON.stringify(err.meta ?? {}).includes('doctor_id')) {
          // a concurrent request took the same doctor slot between our check and our insert
          throw new ConflictException({ code: 'SLOT_TAKEN', message: 'That time was just booked by someone else.' });
        }
        // otherwise a concurrent insert took the same booking id: retry with the next one
      }
    }
  }

  /** An idempotency key may only be replayed with the same booking details. */
  private assertSameRequest(existing: BookingRow, dto: CreateBookingDto): BookingRow {
    const same =
      existing.patientId === dto.patientId &&
      existing.doctorId === dto.doctorId &&
      existing.treatmentId === dto.treatmentId &&
      existing.bookingDate === dto.bookingDate &&
      existing.bookingTime === normalizeTime(dto.bookingTime);
    if (!same) {
      throw new ConflictException({
        code: 'IDEMPOTENCY_KEY_REUSED',
        message: `Idempotency key already used for booking ${existing.id} with different details.`,
      });
    }
    return existing;
  }

  private insert(dto: CreateBookingDto, time: string) {
    return this.prisma.$transaction(async (tx) => {
      const ids = await tx.patientBooking.findMany({ select: { id: true } });
      const next = Math.max(1000, ...ids.map((r) => Number(r.id.replace(/\D/g, '')) || 0)) + 1;
      return tx.patientBooking.create({
        data: {
          id: `B${next}`,
          patientId: dto.patientId,
          doctorId: dto.doctorId,
          treatmentId: dto.treatmentId,
          bookingDate: dto.bookingDate,
          bookingTime: time,
          status: 'Confirmed',
          notes: dto.notes ?? '',
          source: dto.source ?? 'api',
          idempotencyKey: dto.idempotencyKey,
        },
        include: bookingInclude,
      });
    });
  }
}
