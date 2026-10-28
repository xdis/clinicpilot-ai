import {
  BadRequestException,
  Body,
  Controller,
  Delete,
  Get,
  HttpCode,
  HttpException,
  Logger,
  Module,
  Param,
  Post,
  Query,
  Req,
  ServiceUnavailableException,
  UploadedFile,
  UseInterceptors,
} from '@nestjs/common';
import { FileInterceptor } from '@nestjs/platform-express';
import { ConfigService } from '@nestjs/config';
import { ApiBody, ApiConsumes, ApiOperation, ApiProperty, ApiPropertyOptional, ApiTags } from '@nestjs/swagger';
import { IsOptional, IsString, Matches, MaxLength, MinLength } from 'class-validator';
import type { Request } from 'express';

export class ChatRequestDto {
  @ApiProperty({ example: 'I have had an itchy rash on my arm for a week.' })
  @IsString()
  @MinLength(1)
  @MaxLength(2000)
  message!: string;

  @ApiPropertyOptional({ example: 'demo-session-1', description: 'Conversation id; keeps multi-turn booking state.' })
  @IsOptional()
  @Matches(/^[A-Za-z0-9_-]{1,64}$/)
  sessionId?: string;

  @ApiPropertyOptional({ example: 'P001', default: 'P001' })
  @IsOptional()
  @Matches(/^P\d{3,}$/)
  patientId?: string;

  @ApiPropertyOptional({ example: 'P001', description: 'snake_case alias of patientId (as written in the brief)' })
  @IsOptional()
  @Matches(/^P\d{3,}$/)
  patient_id?: string;

  @ApiPropertyOptional({ description: 'snake_case alias of sessionId' })
  @IsOptional()
  @Matches(/^[A-Za-z0-9_-]{1,64}$/)
  session_id?: string;
}

/** The parts of a multer upload this controller uses (kept local instead of adding @types/multer). */
interface UploadedAudio {
  buffer: Buffer;
  mimetype: string;
  originalname: string;
  size: number;
}

export class SummarizeRequestDto {
  @ApiProperty({ description: 'Raw doctor-patient conversation transcript.' })
  @IsString()
  @MinLength(20)
  @MaxLength(20000)
  transcript!: string;
}

/**
 * Thin gateway to the Python AI service: validates input, applies the single-patient
 * demo default (P001), propagates the request id and converts AI-service outages into a clean 503.
 */
@ApiTags('chat')
@Controller()
export class ChatController {
  private readonly logger = new Logger(ChatController.name);

  constructor(private readonly config: ConfigService) {}

  private async forward(path: string, body: unknown, requestId?: string, method: 'POST' | 'GET' | 'DELETE' = 'POST') {
    const base = this.config.get<string>('AI_SERVICE_URL', 'http://localhost:8000');
    try {
      const res = await fetch(`${base}${path}`, {
        method,
        // FormData (file upload) sets its own multipart content type and boundary
        headers:
          body instanceof FormData
            ? { 'x-request-id': requestId ?? '' }
            : { 'content-type': 'application/json', 'x-request-id': requestId ?? '' },
        body: method !== 'POST' ? undefined : body instanceof FormData ? body : JSON.stringify(body),
        signal: AbortSignal.timeout(Number(this.config.get('AI_TIMEOUT_MS', 90000))),
      });
      const payload = await res.json().catch(() => ({}));
      if (!res.ok) {
        const detail = payload?.detail;
        const message =
          typeof detail === 'string' ? detail : (detail?.message ?? 'The AI service returned an error.');
        if (res.status >= 400 && res.status < 500) {
          throw new BadRequestException({ code: 'AI_REQUEST_INVALID', message, details: detail });
        }
        throw new ServiceUnavailableException({ code: detail?.code ?? 'AI_SERVICE_ERROR', message });
      }
      return payload;
    } catch (err) {
      if (err instanceof HttpException) throw err;
      this.logger.error(`AI service unreachable at ${base}: ${(err as Error).message}`);
      throw new ServiceUnavailableException({
        code: 'AI_SERVICE_UNAVAILABLE',
        message: 'The assistant is temporarily unavailable. Please try again in a moment.',
      });
    }
  }

  @Post('chat')
  @HttpCode(200)
  @ApiOperation({ summary: 'Send a patient message; returns the assistant reply plus an agent trace.' })
  chat(@Body() dto: ChatRequestDto, @Req() req: Request & { requestId?: string }) {
    return this.forward(
      '/v1/chat',
      {
        message: dto.message,
        session_id: dto.sessionId ?? dto.session_id ?? 'default',
        patient_id: dto.patientId ?? dto.patient_id ?? 'P001',
      },
      req.requestId,
    );
  }

  @Post('chat/sessions/:sessionId/end')
  @HttpCode(200)
  @ApiOperation({ summary: 'End a conversation: the AI service turns it into one long-term memory episode (background).' })
  endSession(
    @Param('sessionId') sessionId: string,
    @Query('patientId') patientId: string | undefined,
    @Req() req: Request & { requestId?: string },
  ) {
    const patient = patientId ?? 'P001';
    if (!/^[A-Za-z0-9_-]{1,64}$/.test(sessionId) || !/^P\d{3,}$/.test(patient)) {
      throw new BadRequestException({ code: 'INVALID_ID', message: 'invalid session or patient id' });
    }
    return this.forward(`/v1/sessions/${sessionId}/end?patient_id=${patient}`, {}, req.requestId);
  }

  @Get('patients/:id/memory')
  @ApiOperation({ summary: 'What the assistant remembers about a patient: preferences, past-conversation episodes, audit log.' })
  memory(@Param('id') id: string, @Req() req: Request & { requestId?: string }) {
    if (!/^P\d{3,}$/.test(id)) throw new BadRequestException({ code: 'INVALID_ID', message: 'invalid patient id' });
    return this.forward(`/v1/patients/${id}/memory`, undefined, req.requestId, 'GET');
  }

  @Delete('patients/:id/memory')
  @ApiOperation({ summary: "Right to erasure: delete the assistant's long-term memory and chat sessions for a patient." })
  forget(@Param('id') id: string, @Req() req: Request & { requestId?: string }) {
    if (!/^P\d{3,}$/.test(id)) throw new BadRequestException({ code: 'INVALID_ID', message: 'invalid patient id' });
    return this.forward(`/v1/patients/${id}/memory`, undefined, req.requestId, 'DELETE');
  }

  @Post('summarize/audio')
  @HttpCode(200)
  @ApiOperation({ summary: 'Bonus: consultation recording -> local Whisper transcript -> structured medical summary.' })
  @ApiConsumes('multipart/form-data')
  @ApiBody({ schema: { type: 'object', properties: { file: { type: 'string', format: 'binary' } } } })
  @UseInterceptors(FileInterceptor('file', { limits: { fileSize: 15 * 1024 * 1024 } }))
  summarizeAudio(@UploadedFile() file: UploadedAudio | undefined, @Req() req: Request & { requestId?: string }) {
    if (!file) throw new BadRequestException({ code: 'AUDIO_REQUIRED', message: 'Upload a recording in the "file" field.' });
    const form = new FormData();
    form.append('file', new Blob([new Uint8Array(file.buffer)], { type: file.mimetype }), file.originalname || 'audio');
    return this.forward('/v1/summarize/audio', form, req.requestId);
  }

  @Post('summarize')
  @HttpCode(200)
  @ApiOperation({ summary: 'Bonus: turn a doctor-patient transcript into a structured medical summary.' })
  summarize(@Body() dto: SummarizeRequestDto, @Req() req: Request & { requestId?: string }) {
    return this.forward('/v1/summarize', { transcript: dto.transcript }, req.requestId);
  }
}

@Module({ controllers: [ChatController] })
export class ChatModule {}
