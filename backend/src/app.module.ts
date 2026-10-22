import { MiddlewareConsumer, Module, NestModule } from '@nestjs/common';
import { ConfigModule } from '@nestjs/config';
import { BookingsModule } from './bookings/bookings.controller';
import { CatalogModule } from './catalog/catalog.controller';
import { ChatModule } from './chat/chat.controller';
import { RequestIdMiddleware } from './common/request-id.middleware';
import { HealthModule } from './health/health.controller';
import { PatientsModule } from './patients/patients.controller';
import { PrismaModule } from './prisma/prisma.service';

@Module({
  imports: [
    ConfigModule.forRoot({ isGlobal: true }),
    PrismaModule,
    BookingsModule,
    PatientsModule,
    CatalogModule,
    ChatModule,
    HealthModule,
  ],
})
export class AppModule implements NestModule {
  configure(consumer: MiddlewareConsumer) {
    consumer.apply(RequestIdMiddleware).forRoutes('*');
  }
}
