import { Controller, Get, Module } from '@nestjs/common';
import { ApiTags } from '@nestjs/swagger';
import { PrismaService } from '../prisma/prisma.service';

@ApiTags('health')
@Controller('health')
export class HealthController {
  constructor(private readonly prisma: PrismaService) {}

  @Get()
  async health() {
    const bookings = await this.prisma.patientBooking.count();
    return { status: 'ok', database: 'ok', bookings };
  }
}

@Module({ controllers: [HealthController] })
export class HealthModule {}
