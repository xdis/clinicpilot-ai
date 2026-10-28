/**
 * OpenTelemetry for the NestJS gateway. Imported as the first line of main.ts so the instrumentations patch
 * http / express / Nest / fetch before they are loaded. Enabled only when OTEL_EXPORTER_OTLP_ENDPOINT is set.
 *
 * Spans: incoming /api requests, Nest handlers, and outgoing fetch (undici) calls to the AI service. The fetch
 * span injects the W3C `traceparent` header, so the Python service (and its Langfuse agent trace) continue the
 * same trace id.
 */
import { OTLPTraceExporter } from '@opentelemetry/exporter-trace-otlp-http';
import { ExpressInstrumentation } from '@opentelemetry/instrumentation-express';
import { HttpInstrumentation } from '@opentelemetry/instrumentation-http';
import { NestInstrumentation } from '@opentelemetry/instrumentation-nestjs-core';
import { UndiciInstrumentation } from '@opentelemetry/instrumentation-undici';
import { NodeSDK } from '@opentelemetry/sdk-node';

if (
  process.env.OTEL_EXPORTER_OTLP_ENDPOINT &&
  process.env.OTEL_SDK_DISABLED !== 'true'
) {
  const sdk = new NodeSDK({
    serviceName: process.env.OTEL_SERVICE_NAME ?? 'clinicpilot-backend',
    traceExporter: new OTLPTraceExporter(), // endpoint and headers from the standard OTEL_EXPORTER_OTLP_* env vars
    instrumentations: [
      new HttpInstrumentation({
        ignoreIncomingRequestHook: (req) => (req.url ?? '').includes('/health'),
      }),
      new ExpressInstrumentation(),
      new NestInstrumentation(),
      new UndiciInstrumentation(),
    ],
  });
  sdk.start();
  const stop = () => {
    void sdk.shutdown().finally(() => process.exit(0));
  };
  process.once('SIGTERM', stop);
  process.once('SIGINT', stop);
}
