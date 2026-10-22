import { Body, Controller, Get, Headers, HttpCode, Module, Post, Query, Res } from '@nestjs/common';
import { ApiHeader, ApiOperation, ApiTags } from '@nestjs/swagger';
import type { Response } from 'express';
import { BookingsService } from './bookings.service';
import { CreateBookingDto, ListBookingsQuery } from './dto/booking.dto';

@ApiTags('bookings')
@Controller('bookings')
export class BookingsController {
  constructor(private readonly bookings: BookingsService) {}

  @Get()
  @ApiOperation({ summary: 'List bookings (optionally for one patient, upcoming or past).' })
  list(@Query() query: ListBookingsQuery) {
    return this.bookings.list(query);
  }

  @Post('validate')
  @HttpCode(200)
  @ApiOperation({ summary: 'Dry-run validation of a booking (same rules as create, nothing is saved).' })
  validate(@Body() dto: CreateBookingDto) {
    return this.bookings.validate(dto);
  }

  @Post()
  @HttpCode(201)
  @ApiOperation({ summary: 'Create a booking. Called by the AI agent tool (REST or MCP) and the API.' })
  @ApiHeader({ name: 'Idempotency-Key', required: false, description: 'Alternative to the idempotencyKey body field.' })
  async create(
    @Body() dto: CreateBookingDto,
    @Res({ passthrough: true }) res: Response,
    @Headers('idempotency-key') headerKey?: string,
  ) {
    if (!dto.idempotencyKey && headerKey) dto.idempotencyKey = headerKey.slice(0, 128);
    const result = await this.bookings.create(dto);
    if (result.idempotentReplay) res.status(200);
    return result;
  }
}

@Module({
  controllers: [BookingsController],
  providers: [BookingsService],
  exports: [BookingsService],
})
export class BookingsModule {}
