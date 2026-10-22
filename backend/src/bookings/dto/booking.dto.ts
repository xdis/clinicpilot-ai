import { ApiProperty, ApiPropertyOptional } from '@nestjs/swagger';
import { IsIn, IsOptional, IsString, Matches, MaxLength } from 'class-validator';

export class CreateBookingDto {
  @ApiProperty({ example: 'P001' })
  @Matches(/^P\d{3,}$/, { message: 'patientId must look like P001' })
  patientId!: string;

  @ApiProperty({ example: 'D103' })
  @Matches(/^D\d{3,}$/, { message: 'doctorId must look like D101' })
  doctorId!: string;

  @ApiProperty({ example: 'T003' })
  @Matches(/^T\d{3,}$/, { message: 'treatmentId must look like T001' })
  treatmentId!: string;

  @ApiProperty({ example: '2026-10-26', description: 'Clinic-local date, YYYY-MM-DD' })
  @Matches(/^\d{4}-\d{2}-\d{2}$/, { message: 'bookingDate must be YYYY-MM-DD' })
  bookingDate!: string;

  @ApiProperty({ example: '10:00', description: 'Clinic-local time, HH:MM or HH:MM:SS (24h)' })
  @Matches(/^([01]\d|2[0-3]):[0-5]\d(:[0-5]\d)?$/, { message: 'bookingTime must be HH:MM (24h)' })
  bookingTime!: string;

  @ApiPropertyOptional({ example: 'Itchy rash on the left arm for a week.' })
  @IsOptional()
  @IsString()
  @MaxLength(500)
  notes?: string;

  @ApiPropertyOptional({ enum: ['chatbot', 'api'], default: 'api' })
  @IsOptional()
  @IsIn(['chatbot', 'api'])
  source?: 'chatbot' | 'api';

  @ApiPropertyOptional({ description: 'Repeated requests with the same key return the original booking.' })
  @IsOptional()
  @IsString()
  @MaxLength(128)
  idempotencyKey?: string;
}

export class ListBookingsQuery {
  @ApiPropertyOptional({ example: 'P001' })
  @IsOptional()
  @Matches(/^P\d{3,}$/)
  patientId?: string;

  @ApiPropertyOptional({ enum: ['upcoming', 'past', 'all'], default: 'all' })
  @IsOptional()
  @IsIn(['upcoming', 'past', 'all'])
  when?: 'upcoming' | 'past' | 'all';
}
