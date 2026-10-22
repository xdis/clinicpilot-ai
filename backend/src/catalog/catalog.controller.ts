import { Controller, Get, Module } from '@nestjs/common';
import { ApiOperation, ApiTags } from '@nestjs/swagger';
import { CATEGORY_TO_SPECIALTIES } from '../domain/clinic-rules';
import { PrismaService } from '../prisma/prisma.service';

@ApiTags('catalog')
@Controller('catalog')
export class CatalogController {
  constructor(private readonly prisma: PrismaService) {}

  @Get()
  @ApiOperation({
    summary: 'Clinics, doctors and treatments in one payload (the AI service indexes this for retrieval).',
  })
  async all() {
    const [clinics, doctors, treatments] = await Promise.all([
      this.prisma.clinic.findMany({ orderBy: { id: 'asc' } }),
      this.prisma.doctor.findMany({ orderBy: { id: 'asc' } }),
      this.prisma.treatment.findMany({ orderBy: { id: 'asc' } }),
    ]);
    return {
      clinics,
      doctors,
      treatments: treatments.map((t) => ({ ...t, providedBySpecialties: CATEGORY_TO_SPECIALTIES[t.category] ?? [] })),
    };
  }
}

@Module({ controllers: [CatalogController] })
export class CatalogModule {}
