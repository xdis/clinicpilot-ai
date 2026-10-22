import { ArgumentsHost, Catch, ExceptionFilter, HttpException, HttpStatus, Logger } from '@nestjs/common';
import { Prisma } from '@prisma/client';
import type { Request, Response } from 'express';

/**
 * Single error envelope for every failure:
 *   { "error": { "code": "BOOKING_INVALID", "message": "...", "details": ..., "requestId": "..." } }
 * The AI service relies on this shape to explain failures to the patient in plain language.
 */
@Catch()
export class AllExceptionsFilter implements ExceptionFilter {
  private readonly logger = new Logger('HTTP');

  catch(exception: unknown, host: ArgumentsHost) {
    const ctx = host.switchToHttp();
    const res = ctx.getResponse<Response>();
    const req = ctx.getRequest<Request & { requestId?: string }>();

    let status = HttpStatus.INTERNAL_SERVER_ERROR;
    let code = 'INTERNAL_ERROR';
    let message = 'Unexpected server error.';
    let details: unknown;

    if (exception instanceof HttpException) {
      status = exception.getStatus();
      const body = exception.getResponse() as string | Record<string, unknown>;
      if (typeof body === 'string') {
        message = body;
      } else {
        message = Array.isArray(body.message) ? 'Request validation failed.' : String(body.message ?? message);
        details = Array.isArray(body.message) ? body.message : body.details;
        code = typeof body.code === 'string' ? body.code : code;
      }
      if (code === 'INTERNAL_ERROR') code = HttpStatus[status] ?? 'ERROR';
    } else if (exception instanceof Prisma.PrismaClientKnownRequestError) {
      status = exception.code === 'P2025' ? HttpStatus.NOT_FOUND : HttpStatus.CONFLICT;
      code = `DB_${exception.code}`;
      message = 'Database constraint error.';
    }

    if (status >= 500) this.logger.error(`${req.method} ${req.url} failed`, exception as Error);
    res.status(status).json({ error: { code, message, details, requestId: req.requestId } });
  }
}
