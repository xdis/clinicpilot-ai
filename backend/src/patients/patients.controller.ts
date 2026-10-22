import { Controller, Get, Module, NotFoundException, Param } from '@nestjs/common';
import { ApiOperation, ApiTags } from '@nestjs/swagger';
import { BookingsModule } from '../bookings/bookings.controller';
import { BookingsService } from '../bookings/bookings.service';
import { PrismaService } from '../prisma/prisma.service';

@ApiTags('patients')
@Controller('patients')
export class PatientsController {
  constructor(
    private readonly prisma: PrismaService,
    private readonly bookings: BookingsService,
  ) {}

  @Get(':id')
  @ApiOperation({ summary: 'Patient profile with allergies, bookings and visit history (context for the AI agent).' })
  async get(@Param('id') id: string) {
    const patient = await this.prisma.patient.findUnique({
      where: { id },
      include: { history: { include: { doctor: true }, orderBy: { visitDate: 'desc' } } },
    });
    if (!patient) throw new NotFoundException({ code: 'PATIENT_NOT_FOUND', message: `Patient ${id} not found.` });
    const allergies = patient.knownAllergies
      .split(',')
      .map((a) => a.trim())
      .filter((a) => a && a.toLowerCase() !== 'none');
    return {
      patientId: patient.id,
      fullName: `${patient.firstName} ${patient.lastName}`,
      firstName: patient.firstName,
      lastName: patient.lastName,
      dob: patient.dob,
      gender: patient.gender,
      phone: patient.phone,
      email: patient.email,
      allergies,
      emergencyContact: patient.emergencyContact,
      bookings: await this.bookings.list({ patientId: id }),
      treatmentHistory: patient.history.map((h) => ({
        recordId: h.id,
        visitDate: h.visitDate,
        doctorName: h.doctor.fullName,
        chiefComplaint: h.chiefComplaint,
        diagnosis: h.diagnosis,
        prescribedMedication: h.prescribedMedication,
        doctorNotes: h.doctorNotes,
      })),
    };
  }
}

@Module({ imports: [BookingsModule], controllers: [PatientsController] })
export class PatientsModule {}
