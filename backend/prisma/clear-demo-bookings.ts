/**
 * Removes bookings created during demos (source != 'seed') and re-applies the CSV seed,
 * so a live demo always starts from the original dataset. Seed rows are never deleted.
 */
import { PrismaClient } from '@prisma/client';

const prisma = new PrismaClient();

async function main() {
  const { count } = await prisma.patientBooking.deleteMany({ where: { NOT: { source: 'seed' } } });
  console.log(`Removed ${count} demo booking(s); seed data untouched.`);
}

main()
  .catch((err) => {
    console.error(err);
    process.exit(1);
  })
  .finally(() => prisma.$disconnect());
