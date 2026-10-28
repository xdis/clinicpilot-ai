-- A doctor cannot have two active appointments starting at the same time (cancelled ones do not count).
-- Partial index: written by hand because Prisma's schema cannot express WHERE clauses. Overlaps with different
-- start times are checked in BookingsService.findConflicts; on PostgreSQL an exclusion constraint
-- (tsrange && with btree_gist) would enforce those in the database too.
CREATE UNIQUE INDEX "patient_bookings_doctor_slot_active"
ON "patient_bookings"("doctor_id", "booking_date", "booking_time")
WHERE "status" <> 'Cancelled';
