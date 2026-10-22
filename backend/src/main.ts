import { INestApplication, ValidationPipe } from '@nestjs/common';
import { NestFactory } from '@nestjs/core';
import { DocumentBuilder, SwaggerModule } from '@nestjs/swagger';
import { AppModule } from './app.module';
import { AllExceptionsFilter } from './common/http-exception.filter';

/** Shared app setup so e2e tests run exactly the same pipeline as production. */
export function configureApp(app: INestApplication) {
  app.setGlobalPrefix('api');
  app.enableCors();
  app.useGlobalPipes(new ValidationPipe({ whitelist: true, forbidNonWhitelisted: true, transform: true }));
  app.useGlobalFilters(new AllExceptionsFilter());
  return app;
}

async function bootstrap() {
  const app = configureApp(await NestFactory.create(AppModule));
  const doc = new DocumentBuilder()
    .setTitle('ClinicPilot AI Backend')
    .setDescription('Bookings, patients, clinic catalog and the chat gateway to the AI service.')
    .setVersion('1.0.0')
    .build();
  SwaggerModule.setup('api/docs', app, SwaggerModule.createDocument(app, doc));
  const port = Number(process.env.PORT ?? 3000);
  await app.listen(port);
  console.log(`Backend listening on http://localhost:${port}/api  (Swagger: /api/docs)`);
}

if (require.main === module) {
  void bootstrap();
}
