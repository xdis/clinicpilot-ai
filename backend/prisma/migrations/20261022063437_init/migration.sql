-- CreateTable
CREATE TABLE "clinics" (
    "clinic_id" TEXT NOT NULL PRIMARY KEY,
    "clinic_name" TEXT NOT NULL,
    "address" TEXT NOT NULL,
    "phone" TEXT NOT NULL,
    "operating_hours" TEXT NOT NULL,
    "cancellation_policy" TEXT NOT NULL,
    "emergency_instructions" TEXT NOT NULL
);

-- CreateTable
CREATE TABLE "doctors" (
    "doctor_id" TEXT NOT NULL PRIMARY KEY,
    "clinic_id" TEXT NOT NULL,
    "full_name" TEXT NOT NULL,
    "specialty" TEXT NOT NULL,
    "languages_spoken" TEXT NOT NULL,
    "consultation_fee_usd" REAL NOT NULL,
    "bio" TEXT NOT NULL,
    CONSTRAINT "doctors_clinic_id_fkey" FOREIGN KEY ("clinic_id") REFERENCES "clinics" ("clinic_id") ON DELETE RESTRICT ON UPDATE CASCADE
);

-- CreateTable
CREATE TABLE "treatments" (
    "treatment_id" TEXT NOT NULL PRIMARY KEY,
    "treatment_name" TEXT NOT NULL,
    "category" TEXT NOT NULL,
    "duration_minutes" INTEGER NOT NULL,
    "price_usd" REAL NOT NULL,
    "pre_treatment_instructions" TEXT NOT NULL
);

-- CreateTable
CREATE TABLE "patients" (
    "patient_id" TEXT NOT NULL PRIMARY KEY,
    "first_name" TEXT NOT NULL,
    "last_name" TEXT NOT NULL,
    "dob" TEXT NOT NULL,
    "gender" TEXT NOT NULL,
    "phone" TEXT NOT NULL,
    "email" TEXT NOT NULL,
    "known_allergies" TEXT NOT NULL,
    "emergency_contact" TEXT NOT NULL
);

-- CreateTable
CREATE TABLE "patient_bookings" (
    "booking_id" TEXT NOT NULL PRIMARY KEY,
    "patient_id" TEXT NOT NULL,
    "doctor_id" TEXT NOT NULL,
    "treatment_id" TEXT NOT NULL,
    "booking_date" TEXT NOT NULL,
    "booking_time" TEXT NOT NULL,
    "status" TEXT NOT NULL,
    "notes" TEXT NOT NULL DEFAULT '',
    "source" TEXT NOT NULL DEFAULT 'seed',
    "idempotency_key" TEXT,
    "created_at" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT "patient_bookings_patient_id_fkey" FOREIGN KEY ("patient_id") REFERENCES "patients" ("patient_id") ON DELETE RESTRICT ON UPDATE CASCADE,
    CONSTRAINT "patient_bookings_doctor_id_fkey" FOREIGN KEY ("doctor_id") REFERENCES "doctors" ("doctor_id") ON DELETE RESTRICT ON UPDATE CASCADE,
    CONSTRAINT "patient_bookings_treatment_id_fkey" FOREIGN KEY ("treatment_id") REFERENCES "treatments" ("treatment_id") ON DELETE RESTRICT ON UPDATE CASCADE
);

-- CreateTable
CREATE TABLE "patient_treatment_history" (
    "record_id" TEXT NOT NULL PRIMARY KEY,
    "patient_id" TEXT NOT NULL,
    "doctor_id" TEXT NOT NULL,
    "visit_date" TEXT NOT NULL,
    "chief_complaint" TEXT NOT NULL,
    "diagnosis" TEXT NOT NULL,
    "prescribed_medication" TEXT NOT NULL,
    "doctor_notes" TEXT NOT NULL,
    CONSTRAINT "patient_treatment_history_patient_id_fkey" FOREIGN KEY ("patient_id") REFERENCES "patients" ("patient_id") ON DELETE RESTRICT ON UPDATE CASCADE,
    CONSTRAINT "patient_treatment_history_doctor_id_fkey" FOREIGN KEY ("doctor_id") REFERENCES "doctors" ("doctor_id") ON DELETE RESTRICT ON UPDATE CASCADE
);

-- CreateIndex
CREATE UNIQUE INDEX "patient_bookings_idempotency_key_key" ON "patient_bookings"("idempotency_key");

-- CreateIndex
CREATE INDEX "patient_bookings_patient_id_booking_date_idx" ON "patient_bookings"("patient_id", "booking_date");
